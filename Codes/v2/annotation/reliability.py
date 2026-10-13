"""
annotation/reliability.py - read returned rater workbooks for a batch, validate
them, report reliability BEFORE adjudication, and produce majority labels plus
an adjudication list (Future_Plan.md s6). Raw labels are preserved.

Expected layout: annotation/returned/<batch>/<anything>_R1.xlsx, ..._R2.xlsx, ..._R3.xlsx
"""
from __future__ import annotations

import json
from collections import Counter
from itertools import combinations
from pathlib import Path

import pandas as pd

import config_v2 as C
from analysis.agreement import pairwise_confusion, summarise_field
from annotation.schema import LABEL_FIELDS, RELIABILITY_FIELDS, primary_unnecessary, validate_labels
from annotation.sheets_io import read_sheet

RETURNED_DIR = C.ANNOTATION_DIR / "returned"
REPORT_DIR = C.ANNOTATION_DIR / "reports"


def load_returned(batch: str) -> pd.DataFrame:
    rows = []
    folder = RETURNED_DIR / batch
    for rater in C.ANNOTATION["raters"]:
        files = sorted(folder.glob(f"*{rater}*.xlsx"))
        if not files:
            continue
        df = read_sheet(files[-1])
        for _, r in df.iterrows():
            clean, problems = validate_labels({f: r.get(f) for f in LABEL_FIELDS})
            row = {"unit_id": str(r["unit_id"]).strip(), "rater": rater, **clean,
                   "notes": r.get("notes"), "problems": "; ".join(problems),
                   "source_file": files[-1].name}
            row["unnecessary"] = primary_unnecessary(clean.get("least_relevant"))
            rows.append(row)
    return pd.DataFrame(rows)


def reliability_report(long: pd.DataFrame) -> dict:
    report = {}
    fields = RELIABILITY_FIELDS + ["unnecessary"]
    for f in fields:
        wide = long.pivot_table(index="unit_id", columns="rater", values=f, aggfunc="first")
        units = wide.where(wide.notna(), None).values.tolist()
        if f == "unnecessary":
            cats, level = ["yes", "no", "unclear"], "nominal"
        else:
            cats, level = LABEL_FIELDS[f][0], LABEL_FIELDS[f][1]
        summ = summarise_field(units, cats, level)
        conf = {}
        for a, b in combinations([c for c in wide.columns], 2):
            both = wide[[a, b]].dropna()
            conf[f"{a}_vs_{b}"] = pairwise_confusion(list(both[a]), list(both[b]), cats)
        summ["confusion"] = conf
        report[f] = summ
    return report


def majority_labels(long: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for uid, g in long.groupby("unit_id"):
        row = {"unit_id": uid, "n_raters": len(g)}
        disagree = []
        for f in RELIABILITY_FIELDS + ["unnecessary"]:
            vals = [v for v in g[f] if v is not None and v == v]
            cnt = Counter(vals).most_common()
            if cnt and cnt[0][1] >= 2:
                row[f"maj_{f}"] = cnt[0][0]
            else:
                row[f"maj_{f}"] = "no_majority"
            if len(set(vals)) > 1:
                disagree.append(f)
        row["fields_disagreeing"] = ";".join(disagree)
        row["needs_adjudication"] = "unnecessary" in disagree or row["maj_unnecessary"] == "no_majority"
        rows.append(row)
    return pd.DataFrame(rows)


def run(batch: str) -> dict:
    long = load_returned(batch)
    if long.empty:
        raise FileNotFoundError(f"No returned workbooks in {RETURNED_DIR / batch}")
    out = REPORT_DIR / batch
    out.mkdir(parents=True, exist_ok=True)
    long.to_csv(out / "raw_labels_long.csv", index=False, encoding="utf-8-sig")
    rep = reliability_report(long)
    maj = majority_labels(long)
    maj.to_csv(out / "majority_labels.csv", index=False, encoding="utf-8-sig")
    maj[maj["needs_adjudication"]].to_csv(out / "adjudication_list.csv", index=False,
                                          encoding="utf-8-sig")
    problems = long[long["problems"] != ""][["unit_id", "rater", "problems"]]
    problems.to_csv(out / "entry_problems.csv", index=False, encoding="utf-8-sig")
    with open(out / "reliability.json", "w", encoding="utf-8") as fh:
        json.dump(rep, fh, indent=2, default=str)
    lines = [f"# Reliability report: {batch}", "",
             f"Raters: {sorted(long['rater'].unique())}; units: {long['unit_id'].nunique()}; "
             f"entries with problems: {len(problems)}", "",
             "| Field | Units | Alpha (level) | Raw agreement | AC1 | Prevalence |",
             "|---|---:|---:|---:|---:|---|"]
    for f, s in rep.items():
        lines.append(f"| {f} | {s['n_units']} | {s['alpha']:.3f} ({s['alpha_level']}) | "
                     f"{s['raw_agreement']:.3f} | {s['gwet_ac1']:.3f} | {s['prevalence']} |")
    alpha = rep["unnecessary"]["alpha"]
    lines += ["", f"Primary outcome alpha = {alpha:.3f}; provisional pilot screening target "
              f"{C.GATE_TARGETS['pilot_alpha']} (interpret with prevalence and disagreement).",
              f"Units needing adjudication: {int(maj['needs_adjudication'].sum())}"]
    (out / "reliability.md").write_text("\n".join(lines), encoding="utf-8")
    return {"report_dir": str(out), "primary_alpha": alpha,
            "units": int(long["unit_id"].nunique()), "entry_problems": len(problems)}
