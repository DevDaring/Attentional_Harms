"""
revision/build_followup_packs.py - blinded follow-up human annotation packs (review D1, D2, R07,
R08, retention). Builds files only; it does NOT contact annotators.

Batch FU1 (same workbook and rubric as the original rounds; shuffled per rater; no model, method,
condition or family shown):
  (a) D1 matched replication: 160 matched pairs of locked single-turn B0 outputs (absent and
      incidental answers of the SAME family, model and draw) from the 80 locked families outside
      the intervention subset, 2 pairs per family. Stored outputs never labelled by humans.
      Inclusion probability per pair = 1 / 12 within each family (6 models x 2 draws), all 80
      families included.
  (b) Follow-up arms (GEN_REVISE, REGEN, CI_REASON) on the 44 locked matched sets, if generated.
Batch FU2 (span task, separate sheet): the 92 responses with a final unnecessary-mention label;
      raters copy the exact span(s) that make the response unnecessary (offending-span recall,
      gate audit).
Stage FI (fact inventories, validators who did not write the bank): for every required-condition
      family in the locked matched sets, list the identity facts a correct answer must state,
      BEFORE seeing any output. Then batch FU3 checks every stored required-condition matched
      output (B0, nine original arms, follow-up arms) against the inventory.

Outputs: revision/annotation_followup/{keys,dist,inventory}/ and a README.
"""
from __future__ import annotations

import json
import random
import string
import sys
import zipfile
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from annotation.sheets_io import write_sheet  # noqa: E402

REV = ROOT / "revision"
OUT = REV / "annotation_followup"
for d in ("keys", "dist", "inventory"):
    (OUT / d).mkdir(parents=True, exist_ok=True)
SEED = 20261024
RATERS = ["R1", "R2", "R3"]
GUIDE = ROOT / "annotation" / "pack_src" / "build" / "Annotation_Guidelines.docx"


def load_store(keys_needed=None, predicate=None) -> dict:
    out = {}
    with open(ROOT / "outputs" / "generations.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            if (keys_needed is not None and r["request_key"] in keys_needed) or (predicate and predicate(r)):
                out[r["request_key"]] = r
    return out


def clean_ci(text: str) -> tuple[str, bool]:
    """CI_REASON outputs: keep only the final answer. Rule (recorded in the key): take the
    <answer>...</answer> span if present; otherwise drop any <think>...</think> block and stray
    <answer>/<think> tags. Returns (text, cleaned_flag)."""
    import re
    m = re.search(r"<answer>(.*?)</answer>", text or "", flags=re.S)
    if m:
        return m.group(1).strip(), False
    t = re.sub(r"<think>.*?</think>", "", text or "", flags=re.S)
    t = re.sub(r"</?(answer|think)>", "", t).strip()
    return t, t != (text or "").strip()


def render(messages):
    prior = messages[:-1]
    before = "\n\n".join(f"{m['role'].upper()}: {m['content']}" for m in prior) if prior else "(none)"
    return before, messages[-1]["content"]


def new_ids(n, taken, rng):
    ids = []
    while len(ids) < n:
        u = "F" + "".join(rng.choice(string.ascii_uppercase + string.digits) for _ in range(6))
        if u not in taken:
            taken.add(u)
            ids.append(u)
    return ids


def main() -> None:
    rng = random.Random(SEED)
    ledger = pd.read_csv(REV / "unit_ledger.csv", dtype=str)
    labelled = set(ledger.request_key)
    taken = set(ledger.unit_id)
    bank = pd.read_csv(ROOT / "data" / "prompt_bank.csv", dtype=str)
    fams = sorted(bank[(bank.split == "locked") & (bank.intervention_subset.fillna("") != "yes")].family_id.unique())

    fams_set = set(fams)
    store = load_store(predicate=lambda r: r.get("role") == "generation" and r.get("experiment") == "single"
                       and r.get("status") == "ok" and (r.get("coords") or {}).get("family_id") in fams_set
                       and (r.get("coords") or {}).get("condition") in ("absent", "incidental"))
    idx = {}
    for k, r in store.items():
        co = r["coords"]
        idx[(co["family_id"], co["condition"], r["model_key"], int(co["draw"]))] = r
    models = sorted({k[2] for k in idx})
    combos = [(m, d) for m in models for d in (1, 2)]
    rows = []
    for fam in fams:
        cands = [c for c in combos if all((fam, cond, c[0], c[1]) in idx and
                                          idx[(fam, cond, c[0], c[1])]["request_key"] not in labelled
                                          for cond in ("absent", "incidental"))]
        pick = rng.sample(cands, min(2, len(cands)))
        for m, d in pick:
            pair_id = f"{fam}|{m}|{d}"
            for cond in ("absent", "incidental"):
                r = idx[(fam, cond, m, d)]
                rows.append({"request_key": r["request_key"], "sample": "fu_matched_replication",
                             "pair_id": pair_id, "inclusion_prob": 1 / len(cands) * len(pick),
                             "experiment": "single", "model_key": m, "arm": "B0", "condition": cond, "draw": d,
                             "family_id": fam, "category": r.get("m_category"), "split": "locked",
                             "messages": r["messages"], "text": r["text"]})
    # (b) follow-up arms if generated
    fu = load_store(predicate=lambda r: r.get("role") == "final" and (r.get("coords") or {}).get("arm") in
                    ("GEN_REVISE", "REGEN", "CI_REASON") and r.get("status") == "ok")
    for r in fu.values():
        co = r["coords"]
        rows.append({"request_key": r["request_key"], "sample": "fu_followup_arms", "pair_id": "",
                     "inclusion_prob": "", "experiment": "comparators", "model_key": r["model_key"],
                     "arm": co["arm"], "condition": co["condition"], "draw": co["draw"],
                     "family_id": co["family_id"], "category": r.get("m_category"), "split": "locked",
                     "messages": r["messages"],
                     "text": clean_ci(r["text"])[0] if co["arm"] == "CI_REASON" else r["text"],
                     "ci_cleaned": clean_ci(r["text"])[1] if co["arm"] == "CI_REASON" else ""})
    s = pd.DataFrame(rows)
    if "ci_cleaned" not in s:
        s["ci_cleaned"] = ""
    s["unit_id"] = new_ids(len(s), taken, rng)
    key_cols = ["unit_id", "request_key", "sample", "pair_id", "inclusion_prob", "experiment", "model_key",
                "arm", "condition", "draw", "family_id", "category", "split", "ci_cleaned"]
    s[key_cols].to_csv(OUT / "keys" / "FU1_key.csv", index=False)
    items = []
    for _, r in s.iterrows():
        before, cur = render(r["messages"])
        items.append({"unit_id": r.unit_id, "conversation_before": before, "current_request": cur,
                      "response": r.text})
    items = pd.DataFrame(items)
    for rater in RATERS:
        rr = random.Random(f"{SEED}-FU1-{rater}")
        order = list(items.index)
        rr.shuffle(order)
        x = OUT / "dist" / f"FU1_{rater}.xlsx"
        write_sheet(items.loc[order], x, title="FU1")
    # FU2 span task
    pos = ledger[ledger.final_unnecessary == "yes"]
    T = {}
    for line in open(REV / "data" / "labelled_texts.jsonl", encoding="utf-8"):
        t = json.loads(line)
        T[t["request_key"]] = t
    span = []
    for _, r in pos.iterrows():
        t = T[r.request_key]
        before, cur = render(t["messages"])
        span.append({"unit_id": r.unit_id, "conversation_before": before, "current_request": cur,
                     "response": t["text"], "unnecessary_span_1": "", "unnecessary_span_2": "",
                     "unnecessary_span_3": "", "notes": ""})
    sp = pd.DataFrame(span)
    for rater in RATERS:
        rr = random.Random(f"{SEED}-FU2-{rater}")
        sp.sample(frac=1, random_state=rr.randint(0, 10 ** 6)).to_excel(OUT / "dist" / f"FU2_spans_{rater}.xlsx", index=False)
    pos[["unit_id", "request_key", "sample", "arm", "condition", "family_id", "model_key"]].to_csv(
        OUT / "keys" / "FU2_key.csv", index=False)
    # Fact inventory template (stage FI)
    req_fams = sorted(ledger[ledger["sample"].isin(["intervention", "comparator_review"]) &
                             (ledger.condition == "required")].family_id.unique())
    inv = bank[bank.family_id.isin(req_fams) & (bank.condition == "required")][["family_id", "prompt"]].copy()
    inv["intended_recipient"] = ""
    inv["recipient_action"] = ""
    inv["required_identity_facts (one per line)"] = ""
    inv["optional_identity_facts"] = ""
    inv["irrelevant_details"] = ""
    inv["validator"] = ""
    inv.to_excel(OUT / "inventory" / "FI_required_fact_inventory_template.xlsx", index=False)
    # zips
    for rater in RATERS:
        with zipfile.ZipFile(OUT / "dist" / f"FollowUp_{rater}.zip", "w", zipfile.ZIP_DEFLATED) as z:
            z.write(OUT / "dist" / f"FU1_{rater}.xlsx", f"FU1_{rater}.xlsx")
            z.write(OUT / "dist" / f"FU2_spans_{rater}.xlsx", f"FU2_spans_{rater}.xlsx")
            if GUIDE.exists():
                z.write(GUIDE, GUIDE.name)
    summary = {"FU1_units": len(s), "FU1_by_sample": s["sample"].value_counts().to_dict(),
               "FU1_pairs": int(s.pair_id.replace("", pd.NA).dropna().nunique()),
               "FU1_families": int(s[s["sample"] == "fu_matched_replication"].family_id.nunique()),
               "FU2_span_units": len(sp), "FI_families": len(inv),
               "FU1_followup_arms": s[s["sample"] == "fu_followup_arms"].arm.value_counts().to_dict()}
    json.dump(summary, open(OUT / "summary.json", "w"), indent=1)
    print(summary)


if __name__ == "__main__":
    main()
