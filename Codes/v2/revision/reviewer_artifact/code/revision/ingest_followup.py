"""
revision/ingest_followup.py - ingest the returned FU1 and FU2 workbooks (prospective_extension_plan.md).

    python revision/ingest_followup.py

Reads revision/annotation_followup/dist/Annotated/{FU1,FU2_spans}_R{1,2,3}__*.xlsx, validates
every FU1 row with the shared schema, applies the plan's stopping rules, reports reliability on
raw labels BEFORE any adjudication, forms 2-of-3 majority labels and writes a condition-blind
adjudication pack for units with no majority on the primary label. Raw labels are copied
unchanged; nothing here edits a rater value. As for the main batches (deviation log, 10 Oct 2026),
annotator free-text notes and workbook names (institutional codes) are not copied into outputs.

Outputs: revision/annotation_followup/reports/{FU1,FU2}/
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config_v2 as C  # noqa: E402
from annotation.reliability import majority_labels, reliability_report  # noqa: E402
from annotation.schema import LABEL_FIELDS, RELIABILITY_FIELDS, primary_unnecessary, validate_labels  # noqa: E402
from annotation.sheets_io import read_sheet  # noqa: E402

FU = ROOT / "revision" / "annotation_followup"
SRC = FU / "dist" / "Annotated"
OUT = FU / "reports"
MISSING_LIMIT = 0.05                       # plan, stopping rules
ALPHA_FLOOR = 0.60                         # plan, stopping rules


def _rater_files(prefix: str) -> dict[str, Path]:
    files = {}
    for r in C.ANNOTATION["raters"]:
        hit = sorted(SRC.glob(f"{prefix}_{r}__*.xlsx")) or sorted(SRC.glob(f"{prefix}_{r}.xlsx"))
        if not hit:
            raise FileNotFoundError(f"{prefix}: no returned workbook for {r} in {SRC}")
        files[r] = hit[-1]
    return files


# ------------------------------------------------------------------ FU1
def load_fu1() -> pd.DataFrame:
    rows = []
    for rater, path in _rater_files("FU1").items():
        df = read_sheet(path)
        for _, r in df.iterrows():
            clean, problems = validate_labels({f: r.get(f) for f in LABEL_FIELDS})
            rows.append({"unit_id": str(r["unit_id"]).strip(), "rater": rater, **clean,
                         "problems": "; ".join(problems),
                         "unnecessary": primary_unnecessary(clean.get("least_relevant"))})
    return pd.DataFrame(rows)


def completeness(long: pd.DataFrame, key: pd.DataFrame) -> dict:
    out, coded = {}, [f for f in RELIABILITY_FIELDS]
    for r, g in long.groupby("rater"):
        missing_units = sorted(set(key.unit_id) - set(g.unit_id))
        extra_units = sorted(set(g.unit_id) - set(key.unit_id))
        miss_cells = int(g[coded].isna().sum().sum())
        miss_primary = int(g["least_relevant"].isna().sum())
        out[r] = {"rows": len(g), "duplicate_unit_ids": int(g.unit_id.duplicated().sum()),
                  "units_missing_vs_key": len(missing_units), "units_not_in_key": len(extra_units),
                  "missing_or_invalid_coded_cells": miss_cells,
                  "missing_share_coded_cells": miss_cells / (len(g) * len(coded)) if len(g) else 1.0,
                  "missing_or_invalid_primary": miss_primary,
                  "rows_with_consistency_problems": int((g.problems != "").sum())}
        out[r]["passes_5pct_rule"] = (out[r]["missing_share_coded_cells"] <= MISSING_LIMIT
                                      and out[r]["units_missing_vs_key"] <= MISSING_LIMIT * len(key))
    return out


def per_rater_profile(long: pd.DataFrame) -> dict:
    prof = {}
    for f in ("least_relevant", "unnecessary", "required_identity_retained", "task_adequacy",
              "unsupported_personal_attribution", "least_supported"):
        prof[f] = {r: g[f].fillna("missing").astype(str).value_counts().to_dict()
                   for r, g in long.groupby("rater")}
    return prof


def subset_reliability(long: pd.DataFrame, key: pd.DataFrame) -> dict:
    """Primary-label agreement inside each FU1 sample, and with each rater left out."""
    from analysis.agreement import summarise_field
    out = {}
    lk = long.merge(key[["unit_id", "sample"]], on="unit_id", how="left")
    for s, g in [("all", lk)] + list(lk.groupby("sample")):
        wide = g.pivot_table(index="unit_id", columns="rater", values="unnecessary", aggfunc="first")
        res = {"all_raters": summarise_field(wide.where(wide.notna(), None).values.tolist(),
                                             ["yes", "no", "unclear"], "nominal")}
        for drop in wide.columns:
            sub = wide.drop(columns=drop)
            res[f"without_{drop}"] = summarise_field(sub.where(sub.notna(), None).values.tolist(),
                                                     ["yes", "no", "unclear"], "nominal")
        out[s] = {k: {kk: v[kk] for kk in ("n_units", "alpha", "raw_agreement", "gwet_ac1", "prevalence")}
                  for k, v in res.items()}
    return out


def adjudication_pack(maj: pd.DataFrame, long: pd.DataFrame, path: Path) -> int:
    """Condition-blind pack for units with no 2-of-3 majority on the primary label (plan S1).
    Shows request and response text from the rater workbook, the three raw primary labels in a
    shuffled order, and an empty decision column. No key column (arm, condition, model) is shown."""
    todo = maj[maj.maj_unnecessary == "no_majority"].unit_id.tolist()
    if not todo:
        return 0
    text = read_sheet(_rater_files("FU1")["R1"])
    text = text.set_index(text.unit_id.astype(str).str.strip())
    rng = np.random.default_rng(20261025)
    rows = []
    for uid in rng.permutation(todo):
        g = long[long.unit_id == uid]
        labs = list(g.least_relevant.fillna("missing"))
        rng.shuffle(labs)
        t = text.loc[uid]
        rows.append({"unit_id": uid, "conversation_before": t.get("conversation_before"),
                     "current_request": t.get("current_request"), "response": t.get("response"),
                     "rater_labels_least_relevant_shuffled": "; ".join(labs),
                     "adjudicated_least_relevant": "", "adjudicator_note": ""})
    pd.DataFrame(rows).to_excel(path, index=False)
    return len(rows)


def fu1() -> dict:
    key = pd.read_csv(FU / "keys" / "FU1_key.csv", dtype=str)
    long = load_fu1()
    out = OUT / "FU1"
    out.mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "raw_labels_long.csv", index=False, encoding="utf-8-sig")
    comp = completeness(long, key)
    rep = reliability_report(long)
    maj = majority_labels(long)
    maj["n_raters_unnecessary_yes"] = maj.unit_id.map(
        long[long.unnecessary == "yes"].groupby("unit_id").size()).fillna(0).astype(int)
    maj.to_csv(out / "majority_labels.csv", index=False, encoding="utf-8-sig")
    maj[maj.maj_unnecessary == "no_majority"].to_csv(out / "adjudication_list.csv", index=False,
                                                     encoding="utf-8-sig")
    long[long.problems != ""][["unit_id", "rater", "problems"]].to_csv(
        out / "entry_problems.csv", index=False, encoding="utf-8-sig")
    n_adj = adjudication_pack(maj, long, out / "FU1_adjudication_pack_blind.xlsx")
    with open(out / "reliability.json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, default=str)
    alpha = rep["unnecessary"]["alpha"]
    summary = {
        "units": int(long.unit_id.nunique()), "raters": sorted(long.rater.unique()),
        "completeness": comp, "all_pass_5pct_rule": all(v["passes_5pct_rule"] for v in comp.values()),
        "primary_alpha": alpha, "primary_ac1": rep["unnecessary"]["gwet_ac1"],
        "primary_raw_agreement": rep["unnecessary"]["raw_agreement"],
        "alpha_below_floor": bool(alpha < ALPHA_FLOOR),
        "field_reliability": {f: {k: rep[f][k] for k in ("n_units", "alpha", "alpha_level",
                                                          "raw_agreement", "gwet_ac1", "prevalence")}
                              for f in rep},
        "subset_reliability": subset_reliability(long, key),
        "per_rater_profile": per_rater_profile(long),
        "majority_primary_counts": maj.maj_unnecessary.value_counts().to_dict(),
        "units_any_disagreement_primary": int(maj.needs_adjudication.sum()),
        "units_no_majority_primary": n_adj,
        "adjudication_status": "pending" if n_adj else "not_needed",
        "adjudication_note": "Plan S1: no-majority items go to an author blind to condition "
                             "(FU1_adjudication_pack_blind.xlsx). Until returned they are missing "
                             "and are bounded in the analysis; no adjudicated value is invented.",
    }
    (out / "ingest_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


# ------------------------------------------------------------------ FU2
def _spans(v) -> list[str]:
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return []
    s = str(v).strip()
    return [] if s.lower() in ("", "nan", "none", "na", "n/a", "-") else [s]


def fu2() -> dict:
    key = pd.read_csv(FU / "keys" / "FU2_key.csv", dtype=str)
    rows, comp = [], {}
    for rater, path in _rater_files("FU2_spans").items():
        df = read_sheet(path)
        df["unit_id"] = df.unit_id.astype(str).str.strip()
        n_any = 0
        for _, r in df.iterrows():
            sp = [x for c in ("unnecessary_span_1", "unnecessary_span_2", "unnecessary_span_3")
                  for x in _spans(r.get(c))]
            n_any += bool(sp)
            rows.append({"unit_id": r.unit_id, "rater": rater, "n_spans": len(sp),
                         "spans": json.dumps(sp, ensure_ascii=False),
                         "response": r.get("response")})
        comp[rater] = {"rows": len(df), "units_missing_vs_key": len(set(key.unit_id) - set(df.unit_id)),
                       "units_with_at_least_one_span": n_any, "units_with_no_span": len(df) - n_any,
                       "blank_rows_with_note": int(sum(1 for _, r in df.iterrows()
                                                       if not _spans(r.get("unnecessary_span_1"))
                                                       and _spans(r.get("notes"))))}
    long = pd.DataFrame(rows)
    out = OUT / "FU2"
    out.mkdir(parents=True, exist_ok=True)
    long.drop(columns=["response"]).to_csv(out / "raw_spans_long.csv", index=False, encoding="utf-8-sig")
    summary = {"units": int(long.unit_id.nunique()), "completeness": comp,
               "note": "A blank row means the rater found no unnecessary span in the displayed "
                       "response. Blank rows are kept as data, not treated as missing; the "
                       "stopping rule does not apply because 'no span' is a valid answer."}
    (out / "ingest_summary.json").write_text(json.dumps(summary, indent=2, default=str), encoding="utf-8")
    return summary


def main() -> None:
    s1 = fu1()
    s2 = fu2()
    print(json.dumps({"FU1": {k: s1[k] for k in ("units", "all_pass_5pct_rule", "primary_alpha",
                                                 "primary_ac1", "primary_raw_agreement",
                                                 "alpha_below_floor", "majority_primary_counts",
                                                 "units_any_disagreement_primary",
                                                 "units_no_majority_primary")},
                      "FU1_completeness": s1["completeness"], "FU2": s2}, indent=1, default=str))


if __name__ == "__main__":
    main()
