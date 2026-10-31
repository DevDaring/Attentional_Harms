"""
analysis/robustness.py - post-hoc robustness checks and RGR ablation on existing human labels.

All checks reuse the confirmatory estimators in analysis/effects.py; none of them enters the
Holm-corrected primary family. Outputs: outputs/analysis/robustness/robustness.json

  1. Headline contrast (incidental - absent, representative B0 sample) under alternative
     label aggregation (lenient, strict, each rater alone), without inverse-probability
     weights, leaving out one model or one identity category at a time.
  2. Per-category and per-model incidental prevalence (descriptive).
  3. RGR ablation on the development split (rgr2 batch): RGR with and without the restore
     step (RGR_RO), each paired with B0 on the same matched sets.
  4. RGR mechanism: gate outcomes and restore activity by task condition (locked traces),
     and lexical detector coverage of human-marked unnecessary mentions.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config_v2 as C
from analysis.bootstrap import Bootstrap, CellMean
from analysis.effects import CELLS, _pairs, _paired, build_frame

PRIMARY = "y_unnecessary"


def _rater_variants(ann_root: Path) -> pd.DataFrame:
    """unit_id x {lenient, strict, R1, R2, R3} labels for the primary outcome (yes/no/NaN)."""
    rows = []
    for f in sorted((ann_root / "reports").glob("*/raw_labels_long.csv")):
        rows.append(pd.read_csv(f, dtype=str, usecols=["unit_id", "rater", "unnecessary"]))
    long = pd.concat(rows, ignore_index=True)
    wide = long.pivot_table(index="unit_id", columns="rater", values="unnecessary", aggfunc="first")
    yes = (wide == "yes")
    no = (wide == "no")
    out = pd.DataFrame(index=wide.index)
    out["lenient"] = np.where(yes.any(axis=1), "yes", np.where(no.any(axis=1), "no", None))
    out["strict"] = np.where(yes.all(axis=1), "yes", np.where(no.any(axis=1), "no", None))
    for r in wide.columns:
        out[r] = wide[r].where(wide[r].isin(["yes", "no"]))
    return out.reset_index()


def _single(df: pd.DataFrame) -> pd.DataFrame:
    s = df[(df["experiment"] == "single") & (df["arm"] == "B0") & (df["split"] == "locked")
           & (df["sample"] == "representative_single")]
    return s.reset_index(drop=True)


def _headline(single: pd.DataFrame, weight: str | None = "ipw") -> dict:
    bs = Bootstrap(single)
    inc = CellMean(single, PRIMARY, CELLS, single["condition"] == "incidental", weight)
    ab = CellMean(single, PRIMARY, CELLS, single["condition"] == "absent", weight)
    out = bs.run(lambda w: inc(w) - ab(w))
    out["incidental"] = inc(np.ones(len(single)))
    out["absent"] = ab(np.ones(len(single)))
    return out


def headline_checks(df: pd.DataFrame, variants: pd.DataFrame) -> dict:
    base = _single(df)
    res = {"main": _headline(base), "unweighted": _headline(base, None)}
    v = base[["unit_id"]].merge(variants, on="unit_id", how="left")
    for col in [c for c in variants.columns if c != "unit_id"]:
        alt = base.copy()
        alt[PRIMARY] = v[col].map({"yes": 1.0, "no": 0.0}).to_numpy()
        res[f"labels_{col}"] = _headline(alt)
    for m in sorted(base["model_key"].unique()):
        res[f"drop_model_{m}"] = _headline(base[base["model_key"] != m].reset_index(drop=True))
    for k in sorted(base["category"].unique()):
        res[f"drop_category_{k}"] = _headline(base[base["category"] != k].reset_index(drop=True))
    return res


def breakdown(df: pd.DataFrame) -> dict:
    """Incidental and absent prevalence within each category and each model (IPW, unpooled)."""
    base = _single(df)
    out = {}
    for col in ("category", "model_key"):
        for g, sub in base.groupby(col):
            sub = sub.reset_index(drop=True)
            other = [c for c in CELLS if c != col]
            bs = Bootstrap(sub)
            inc = CellMean(sub, PRIMARY, other, sub["condition"] == "incidental", "ipw")
            ab = CellMean(sub, PRIMARY, other, sub["condition"] == "absent", "ipw")
            r = bs.run(lambda w: inc(w) - ab(w))
            r.update({"incidental": inc(np.ones(len(sub))), "absent": ab(np.ones(len(sub))),
                      "n_incidental": int(((sub["condition"] == "incidental") & sub[PRIMARY].notna()).sum()),
                      "n_pos_incidental": int(((sub["condition"] == "incidental") & (sub[PRIMARY] == 1)).sum())})
            out[f"{col}={g}"] = r
    return out


def _clean(e: dict) -> dict:
    return {k: v for k, v in e.items() if not k.startswith("_")}


def rgr_ablation(df: pd.DataFrame) -> dict:
    """Matched-set contrasts on the development split (rgr2 batch, 44 sets x 5 arms)."""
    m = df[(df["sample"] == "rgr2_review")].copy()
    m["draw"] = pd.to_numeric(m["draw"], errors="coerce").astype("Int64")
    b0 = m[m["arm"] == "B0"]
    out = {"design": {"matched_sets": int(len(b0)), "families": int(b0["family_id"].nunique())}}
    for a, b in (("RGR", "B0"), ("RGR_RO", "B0"), ("RGR", "RGR_RO"), ("B1", "B0"), ("B2", "B0")):
        p = _pairs(m, a, b)
        for y, cond in ((PRIMARY, "nonreq"), ("y_retained", "req"), ("y_adequacy", None)):
            out[f"{a}_vs_{b}__{y}"] = _clean(_paired(p, y, cond)[0])
        sel = p[(p["condition"] != "required") & p[f"{PRIMARY}_a"].notna() & p[f"{PRIMARY}_b"].notna()]
        out[f"level_{a}_paired_{b}"] = {"a": float(sel[f"{PRIMARY}_a"].mean()),
                                        "b": float(sel[f"{PRIMARY}_b"].mean()), "n": int(len(sel))}
    return out


def rgr_mechanism(store_path: Path, gen: pd.DataFrame) -> dict:
    """Gate outcomes and restore activity of RGR and RGR_RO by split and condition."""
    cond = gen.set_index("request_key")["condition"].to_dict()
    rows = []
    with open(store_path, encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            c = r.get("coords") or {}
            if r.get("role") != "final" or c.get("arm") not in ("RGR", "RGR_RO") or r.get("status") != "ok":
                continue
            t = r.get("trace") or {}
            rows.append({"arm": c.get("arm"), "split": r.get("m_split"),
                         "condition": c.get("condition") or cond.get(r["request_key"]),
                         "gate": t.get("gate"), "n_remove": len(t.get("remove") or []),
                         "n_restore": len(t.get("restore") or []),
                         "n_restore_suppressed": len(t.get("restore_suppressed") or [])})
    tr = pd.DataFrame(rows)
    out = {}
    for (arm, split), g in tr.groupby(["arm", "split"]):
        for cnd, h in g.groupby("condition"):
            n = len(h)
            out[f"{arm}|{split}|{cnd}"] = {
                "n": n,
                "skipped_no_identity": float((h["gate"] == "skipped_no_identity").mean()),
                "no_change": float((h["gate"] == "no_change").mean()),
                "revised": float((h["gate"] == "revised").mean()),
                "guard_kept_draft": float((h["gate"] == "retention_guard_keep_draft").mean()),
                "remove_any": float((h["n_remove"] > 0).mean()),
                "restore_any": float(((h["n_restore"] + h["n_restore_suppressed"]) > 0).mean()),
                "revised_with_restore_only": float(((h["gate"] == "revised") & (h["n_remove"] == 0)
                                                    & (h["n_restore"] > 0)).mean())}
    return out


def detector_coverage(df: pd.DataFrame) -> dict:
    """Share of human-labelled unnecessary (and any-identity) responses in which the RGR lexical
    detector finds at least one identity term; missed responses pass through RGR unchanged."""
    from remedies import identity_terms
    d = df[df[PRIMARY].notna()].copy()
    d["n_terms"] = d["text"].fillna("").map(lambda t: len(identity_terms(t)))
    out = {}
    for name, sel in (("unnecessary", d[PRIMARY] == 1), ("not_unnecessary", d[PRIMARY] == 0)):
        s = d[sel]
        out[name] = {"n": int(len(s)), "detected": float((s["n_terms"] > 0).mean()) if len(s) else float("nan"),
                     "n_missed": int((s["n_terms"] == 0).sum())}
    return out


def run(gen: pd.DataFrame, labels: pd.DataFrame, bank: pd.DataFrame, ann_root: Path,
        store_path: Path, out_dir: Path) -> dict:
    df = build_frame(gen, labels, "human", bank)
    df = df.merge(labels[["request_key", "unit_id"]].drop_duplicates("request_key"), on="request_key", how="left")
    variants = _rater_variants(ann_root)
    res = {"headline": headline_checks(df, variants), "breakdown": breakdown(df),
           "rgr_ablation_dev": rgr_ablation(df), "rgr_mechanism": rgr_mechanism(store_path, gen),
           "detector_coverage": detector_coverage(df)}
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "robustness.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=2, default=float)
    return res
