"""
revision/select_examples.py - worked examples drawn from actual labelled records by fixed rules
(review s7: "Choose cases by a stated rule, not only by how favorable they are").

Rules (applied in family_id, then unit_id order; first match wins):
  E1 matched family: first family in the locked intervention sample with labelled B0 outputs in all
     three conditions for the same model and draw, preferring one whose incidental B0 output is a
     human positive.
  E2 useful-but-optional: first representative single-turn unit whose majority least_relevant is
     useful_optional.
  E3 unsupported attribution or pronoun: first labelled unit with majority least_supported =
     unsupported and majority unsupported_personal_attribution or gendered pronoun = yes.
  E4 revision that loses required identity: first locked comparator pair (any arm) where B0 retained
     and the arm did not (majority labels), by arm order SYS2_IMPL, CAI_REVISE, REPROMPT, IF_MSC, B1.
  E5 RGR restore: first locked labelled RGR output where an executed restore preceded a human
     positive that B0 did not have.
Writes revision/results/worked_examples.json
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

REV = Path(__file__).resolve().parent
L = pd.read_csv(REV / "unit_ledger.csv", dtype=str)
T = {}
for line in open(REV / "data" / "labelled_texts.jsonl", encoding="utf-8"):
    r = json.loads(line)
    T[r["request_key"]] = r


def rec(row) -> dict:
    t = T.get(row["request_key"], {})
    msgs = t.get("messages") or []
    return {"unit_id": row["unit_id"], "family_id": row["family_id"], "category": row["category"],
            "condition": row["condition"], "model": row["model_key"], "arm": row["arm"], "draw": row["draw"],
            "request": msgs[-1]["content"] if msgs else None, "response": t.get("text"),
            "labels": {k: row.get(f"final_{k}") for k in ("unnecessary", "required_identity_retained",
                                                           "task_adequacy")} |
            {"least_supported": row.get("maj_least_supported"), "least_relevant": row.get("maj_least_relevant"),
             "raters_unnecessary_yes": row.get("n_raters_unnecessary_yes"),
             "rater_spans": [row.get(f"R{i}_identity_spans") for i in (1, 2, 3)]}}


out = {}
iv = L[(L["sample"] == "intervention") & (L.arm == "B0")].copy()
iv["draw_i"] = iv.draw.astype(float).astype(int)
cands = []
for (fam, model, draw), g in iv.groupby(["family_id", "model_key", "draw_i"]):
    if set(g.condition) >= {"absent", "incidental", "required"}:
        pos = (g[g.condition == "incidental"].final_unnecessary == "yes").any()
        cands.append((not pos, fam, model, draw))
cands.sort()
if cands:
    _, fam, model, draw = cands[0]
    g = iv[(iv.family_id == fam) & (iv.model_key == model) & (iv.draw_i == draw)]
    out["E1_matched_family"] = [rec(r) for _, r in g.sort_values("condition").iterrows()]
else:
    # fall back to representative units of one family (different models/draws)
    rep = L[L["sample"] == "representative_single"]
    fams = rep.groupby("family_id").condition.nunique()
    fam = sorted(fams[fams == 3].index)[0]
    out["E1_matched_family"] = [rec(r) for _, r in rep[rep.family_id == fam].sort_values("condition").iterrows()]
    out["E1_note"] = "no intervention family with all three conditions for one model/draw; representative units"

rep = L[L["sample"] == "representative_single"].sort_values(["family_id", "unit_id"])
e2 = rep[rep.maj_least_relevant == "useful_optional"]
out["E2_useful_optional"] = rec(e2.iloc[0]) if len(e2) else None

e3 = L[(L.maj_least_supported == "unsupported") &
       ((L.maj_unsupported_personal_attribution == "yes") | (L.maj_person_gendered_pronoun == "yes"))] \
    .sort_values(["family_id", "unit_id"])
out["E3_unsupported"] = rec(e3.iloc[0]) if len(e3) else None

m = L[L["sample"].isin(["intervention", "comparator_review"]) & (L.split == "locked")].copy()
m["draw_i"] = m.draw.astype(float).astype(int)
b0 = m[m.arm == "B0"].set_index(["family_id", "condition", "model_key", "draw_i"])
for arm in ["SYS2_IMPL", "CAI_REVISE", "REPROMPT", "IF_MSC", "B1"]:
    a = m[(m.arm == arm) & (m.condition == "required") & (m.final_required_identity_retained == "no")] \
        .sort_values(["family_id", "unit_id"])
    hit = None
    for _, r in a.iterrows():
        k = (r.family_id, r.condition, r.model_key, r.draw_i)
        if k in b0.index and b0.loc[k].final_required_identity_retained == "yes":
            hit = (r, b0.loc[k])
            break
    if hit:
        r, b = hit
        brow = L[L.request_key == b.request_key].iloc[0]
        out["E4_retention_loss"] = {"arm": arm, "b0": rec(brow), "remedy": rec(r)}
        break

g = m[(m.arm == "RGR") & (m.final_unnecessary == "yes") & (m.rgr_gate == "revised") &
      m.rgr_restore.fillna("[]").ne("[]")].sort_values(["family_id", "unit_id"])
for _, r in g.iterrows():
    k = (r.family_id, r.condition, r.model_key, r.draw_i)
    if k in b0.index and b0.loc[k].final_unnecessary == "no":
        brow = L[L.request_key == b0.loc[k].request_key].iloc[0]
        out["E5_rgr_restore"] = {"restore": r.rgr_restore, "b0": rec(brow), "rgr": rec(r)}
        break

(REV / "results").mkdir(exist_ok=True)
json.dump(out, open(REV / "results" / "worked_examples.json", "w", encoding="utf-8"), indent=1, ensure_ascii=False)
for k, v in out.items():
    print("=====", k)
    print(json.dumps(v, ensure_ascii=False)[:2500])
