"""
analysis/evaluator_audit.py - validate automated labels against held-out human
labels (Future_Plan.md s6, Evaluator gate).

Development data (threshold selection only): pilot units and challenge units
from dev families. Test data: representative, intervention-review and
challenge units from locked families. The challenge sample is reported
separately because it is enriched.

Compared on the same human labels:
  judge ensemble (production rule: two assigned other-family judges, third
  vote on disagreement), each judge alone, and lexical baselines (any group
  label, history-novel group label, densities, old GSS, length) with
  thresholds tuned on development data.
Metrics: precision, recall, F1, macro-F1, false-positive rate on
identity-required responses, abstention rate; family-cluster bootstrap CIs.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

import config_v2 as C
from analysis.bootstrap import Bootstrap
from judges import assign_judges, third_judge

THRESHOLD_FEATURES = ["density_non_pronoun", "density_group_label", "old_gss", "n_words"]
BINARY_FEATURES = ["any_group_label", "novel_group_label"]


def ensemble_label(votes: dict[str, str], generator: str) -> str:
    """Emulate the production rule on calibration votes (judge -> yes/no/unclear/abstain)."""
    j1, j2 = assign_judges(generator)
    v1, v2 = votes.get(j1), votes.get(j2)
    if v1 in ("yes", "no") and v1 == v2:
        return v1
    third = third_judge(generator, [j1, j2])
    pool = [v for v in (v1, v2, votes.get(third) if third else None) if v in ("yes", "no")]
    c = Counter(pool).most_common()
    if c and c[0][1] >= 2:
        return c[0][0]
    return "abstain"


def _counts(y_true: np.ndarray, y_pred: np.ndarray, w: np.ndarray) -> tuple:
    tp = (w * ((y_true == 1) & (y_pred == 1))).sum()
    fp = (w * ((y_true == 0) & (y_pred == 1))).sum()
    fn = (w * ((y_true == 1) & (y_pred == 0))).sum()
    tn = (w * ((y_true == 0) & (y_pred == 0))).sum()
    return tp, fp, fn, tn


def _metric(name: str, y_true, y_pred, required_mask):
    def f(w):
        ok = ~np.isnan(y_pred)
        tp, fp, fn, tn = _counts(y_true[ok], y_pred[ok], w[ok])
        prec = tp / (tp + fp) if tp + fp else np.nan
        rec = tp / (tp + fn) if tp + fn else np.nan
        f1 = 2 * prec * rec / (prec + rec) if prec == prec and rec == rec and prec + rec else np.nan
        npv = tn / (tn + fn) if tn + fn else np.nan
        spec = tn / (tn + fp) if tn + fp else np.nan
        f1n = 2 * npv * spec / (npv + spec) if npv == npv and spec == spec and npv + spec else np.nan
        if name == "precision":
            return prec
        if name == "recall":
            return rec
        if name == "f1":
            return f1
        if name == "macro_f1":
            return np.nanmean([f1, f1n]) if f1 == f1 or f1n == f1n else np.nan
        if name == "fp_rate_required":
            m = ok & required_mask & (y_true == 0)
            return (w[m] * (y_pred[m] == 1)).sum() / w[m].sum() if w[m].sum() else np.nan
        raise ValueError(name)
    return f


def _tune(dev: pd.DataFrame, feat: str) -> float:
    best, best_f1 = None, -1.0
    y = dev["human"].to_numpy()
    for t in sorted(dev[feat].dropna().unique()):
        pred = (dev[feat] >= t).astype(int).to_numpy()
        tp = ((y == 1) & (pred == 1)).sum()
        fp = ((y == 0) & (pred == 1)).sum()
        fn = ((y == 1) & (pred == 0)).sum()
        f1 = 2 * tp / (2 * tp + fp + fn) if tp else 0.0
        if f1 > best_f1:
            best, best_f1 = t, f1
    return float(best) if best is not None else float("inf")


def run(units: pd.DataFrame, calib_votes: pd.DataFrame, out_path: Path) -> dict:
    """units: one row per human-labelled unit with columns request_key, sample,
    split, family_id, category, condition, model_key, human ('yes'/'no'/...),
    plus lexical features. calib_votes: judgments_frame() rows (purpose
    'calibration')."""
    df = units.copy()
    df = df[df["human"].isin(["yes", "no"])].copy()
    df["y"] = (df["human"] == "yes").astype(float)
    votes = calib_votes.pivot_table(index="target", columns="judge", values="unnecessary",
                                    aggfunc="first")
    preds = {}
    ens = []
    for _, r in df.iterrows():
        v = votes.loc[r["request_key"]].to_dict() if r["request_key"] in votes.index else {}
        ens.append(ensemble_label(v, r["model_key"]))
    df["pred_ensemble"] = [1.0 if e == "yes" else 0.0 if e == "no" else np.nan for e in ens]
    preds["ensemble"] = "pred_ensemble"
    for j in votes.columns:
        col = f"pred_{j}"
        df[col] = df["request_key"].map(votes[j]).map({"yes": 1.0, "no": 0.0})
        preds[j] = col
    dev = df[(df["sample"] == "pilot") | ((df["sample"] == "challenge") & (df["split"] == "dev"))]
    test = df[df["split"] == "locked"]
    thresholds = {}
    for feat in THRESHOLD_FEATURES:
        thresholds[feat] = _tune(dev.dropna(subset=[feat]).assign(human=dev["y"]), feat) if len(dev) else np.inf
        df[f"pred_{feat}"] = (df[feat] >= thresholds[feat]).astype(float)
        preds[feat] = f"pred_{feat}"
    for feat in BINARY_FEATURES:
        df[f"pred_{feat}"] = df[feat].astype(float)
        preds[feat] = f"pred_{feat}"
    test = df[df["split"] == "locked"]
    # best baseline chosen on DEV F1 (never on test)
    def dev_f1(col):
        d = df.loc[dev.index].dropna(subset=[col])
        y, p = d["y"].to_numpy(), d[col].to_numpy()
        tp, fp, fn = ((y == 1) & (p == 1)).sum(), ((y == 0) & (p == 1)).sum(), ((y == 1) & (p == 0)).sum()
        return 2 * tp / (2 * tp + fp + fn) if tp else 0.0
    baselines = THRESHOLD_FEATURES + BINARY_FEATURES
    best_baseline = max(baselines, key=lambda b: dev_f1(preds[b])) if len(dev) else "any_group_label"

    report = {"n_dev": int(len(dev)), "n_test": int(len(test)), "thresholds_from_dev": thresholds,
              "best_baseline_on_dev": best_baseline, "subsets": {}}
    for subset_name, sub in (("representative", test[test["sample"].str.startswith("representative")]),
                             ("intervention_review", test[test["sample"] == "intervention"]),
                             ("challenge_locked", test[test["sample"] == "challenge"]),
                             ("all_locked", test)):
        sub = sub.reset_index(drop=True)
        if len(sub) < 5:
            continue
        bs = Bootstrap(sub)
        y = sub["y"].to_numpy()
        req = (sub["condition"] == "required").to_numpy()
        block = {"n": int(len(sub)), "n_pos": int(y.sum()), "n_neg": int((1 - y).sum()), "methods": {}}
        for name, col in preds.items():
            p = sub[col].to_numpy(dtype=float)
            m = {"abstention_rate": float(np.isnan(p).mean())}
            for metric in ("precision", "recall", "f1", "macro_f1", "fp_rate_required"):
                m[metric] = bs.run(_metric(metric, y, p, req))
            block["methods"][name] = m
        pe = sub[preds["ensemble"]].to_numpy(dtype=float)
        pb = sub[preds[best_baseline]].to_numpy(dtype=float)
        fe, fb = _metric("f1", y, pe, req), _metric("f1", y, pb, req)
        block["added_value_f1_ensemble_minus_best_baseline"] = bs.run(lambda w: fe(w) - fb(w))
        report["subsets"][subset_name] = block
    rep = report["subsets"].get("representative") or report["subsets"].get("all_locked")
    if rep:
        ens = rep["methods"]["ensemble"]
        g = C.GATE_TARGETS
        report["gate_check"] = {
            "precision": ens["precision"]["estimate"], "precision_target": g["evaluator_precision"],
            "recall": ens["recall"]["estimate"], "recall_target": g["evaluator_recall"],
            "n_pos": rep["n_pos"], "n_neg": rep["n_neg"],
            "meets_point_targets": bool(ens["precision"]["estimate"] >= g["evaluator_precision"]
                                        and ens["recall"]["estimate"] >= g["evaluator_recall"]),
            "enough_cases": bool(rep["n_pos"] >= g["min_pos_cases"] and rep["n_neg"] >= g["min_neg_cases"]),
        }
        report["added_value"] = rep["added_value_f1_ensemble_minus_best_baseline"]
    Path(out_path).parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)
    df.to_csv(Path(out_path).with_suffix(".units.csv"), index=False, encoding="utf-8-sig")
    return report
