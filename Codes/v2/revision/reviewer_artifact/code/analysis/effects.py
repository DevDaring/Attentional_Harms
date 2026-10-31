"""
analysis/effects.py - primary and secondary estimands (Future_Plan.md s7).

All estimands are equal-weight averages over model x category cells, using
locked families for confirmation (dev reported separately). Labels come from
ONE source per run: 'judge' (consensus of automated judges) or 'human'
(majority labels, inverse-probability weighted within cells). The two are
never mixed.

Primary contrast family (Holm):
  H_evaluator     F1 gain of the judge ensemble over the best dev-selected baseline (directional)
  H_dialogue      first switched task: identity history - neutral history
  H_remedy        RGR - B0 unnecessary-mention prevalence on non-required conditions, with
                  retention and adequacy non-inferiority reported alongside
Secondary: every remedy vs B0, RGR vs every comparator, B2 vs B1.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

import config_v2 as C
from analysis.bootstrap import Bootstrap, CellMean, holm

CELLS = ["model_key", "category"]


def _binary(series: pd.Series) -> pd.Series:
    return series.map({"yes": 1.0, "no": 0.0}).astype(float)


def build_frame(gen: pd.DataFrame, labels: pd.DataFrame, source: str, bank: pd.DataFrame) -> pd.DataFrame:
    """gen: generation_frame(); labels: judge consensus (target=...) or human
    majority joined to request_key."""
    if source == "judge":
        lab = labels.rename(columns={"target": "request_key"})
        lab = lab.assign(y_unnecessary=_binary(lab["judge_unnecessary"]),
                         y_attribution=_binary(lab["judge_unsupported_personal_attribution"]),
                         y_retained=_binary(lab["judge_required_identity_retained"]),
                         y_refusal=_binary(lab["judge_refusal"]),
                         y_adequacy=pd.to_numeric(lab["judge_task_adequacy"], errors="coerce") / 2.0,
                         ipw=1.0)
    else:
        lab = labels.assign(y_unnecessary=_binary(labels["maj_unnecessary"]),
                            y_attribution=_binary(labels["maj_unsupported_personal_attribution"]),
                            y_retained=_binary(labels["maj_required_identity_retained"]),
                            y_refusal=_binary(labels["maj_refusal"]),
                            y_adequacy=pd.to_numeric(labels["maj_task_adequacy"], errors="coerce") / 2.0)
        lab["ipw"] = 1.0 / pd.to_numeric(lab.get("inclusion_prob"), errors="coerce").replace(0, np.nan)
        lab["ipw"] = lab["ipw"].fillna(1.0)
        lab["retained_raw"] = labels["maj_required_identity_retained"]
    keep = ["request_key", "y_unnecessary", "y_attribution", "y_retained", "y_refusal",
            "y_adequacy", "ipw"] + [c for c in ("sample", "cell", "retained_raw") if c in lab.columns]
    df = gen.merge(lab[keep], on="request_key", how="inner")
    df["n_words"] = df["text"].fillna("").str.split().str.len()
    subset = bank.drop_duplicates("family_id").set_index("family_id")["intervention_subset"]
    df["intervention_subset"] = df["family_id"].map(subset).fillna(False).astype(bool)
    return df


def _contrast(bs: Bootstrap, a: CellMean, b: CellMean) -> dict:
    return bs.run(lambda w: a(w) - b(w))


REMEDY_SAMPLES = ("intervention", "comparator_review")
PAIR_KEYS = ["family_id", "condition", "model_key", "draw"]
REMEDY_OUTCOMES = (("y_unnecessary", "nonreq"), ("y_retained", "req"), ("y_adequacy", None),
                   ("y_refusal", None))


def _cp_upper(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided Clopper-Pearson upper bound for a binomial proportion."""
    from scipy.stats import beta
    return 1.0 if k >= n else float(beta.ppf(1 - alpha, k + 1, n - k))


def _pairs(m: pd.DataFrame, a: str, b: str) -> pd.DataFrame:
    """Rows of arm a joined to arm b on the same (family, condition, model, draw)."""
    left = m[m["arm"] == a].drop_duplicates(PAIR_KEYS).set_index(PAIR_KEYS)
    right = m[m["arm"] == b].drop_duplicates(PAIR_KEYS).set_index(PAIR_KEYS)
    p = left.join(right, how="inner", lsuffix="_a", rsuffix="_b").reset_index()
    return p.rename(columns={"category_a": "category"})


def _paired(p: pd.DataFrame, y: str, cond: str | None) -> tuple[dict, dict, pd.DataFrame]:
    """Mean paired difference (a - b) over matched sets with both labels present, a
    family-cluster bootstrap interval, and exact discordance counts."""
    from scipy.stats import binomtest
    sel = p if cond is None else p[(p["condition"] != "required") if cond == "nonreq"
                                   else (p["condition"] == "required")]
    sel = sel[sel[f"{y}_a"].notna() & sel[f"{y}_b"].notna()].reset_index(drop=True)
    info = {"n_pairs": int(len(sel)), "n_families": int(sel["family_id"].nunique()) if len(sel) else 0}
    if not len(sel):
        return {"estimate": float("nan"), "ci_low": float("nan"), "ci_high": float("nan"),
                "se": float("nan"), "p_boot": float("nan"), **info}, info, sel
    d = (sel[f"{y}_a"] - sel[f"{y}_b"]).to_numpy(dtype=float)
    bs = Bootstrap(sel)
    out = bs.run(lambda w: float((w * d).sum() / w.sum()) if w.sum() > 0 else float("nan"))
    out.update(info)
    if y != "y_adequacy":                       # binary outcome: discordant pairs and exact McNemar
        n_ab = int(((sel[f"{y}_a"] == 1) & (sel[f"{y}_b"] == 0)).sum())
        n_ba = int(((sel[f"{y}_a"] == 0) & (sel[f"{y}_b"] == 1)).sum())
        out.update({"n_a1_b0": n_ab, "n_a0_b1": n_ba,
                    "p_mcnemar_exact": float(binomtest(min(n_ab, n_ba), n_ab + n_ba, 0.5).pvalue)
                    if n_ab + n_ba else 1.0})
    out["_bs"] = bs
    out["_d"] = d
    return out, info, sel


def matched_remedies(d: pd.DataFrame, E: dict, res: dict) -> None:
    """Remedy contrasts on human-labelled matched sets only (each remedy output paired with
    the B0 output of the same family, condition, model and draw)."""
    m = d[d["sample"].isin(REMEDY_SAMPLES)].copy()
    if m.empty or m["arm"].nunique() < 2:
        return
    m["draw"] = pd.to_numeric(m["draw"], errors="coerce").astype("Int64")
    arms = sorted(a for a in m["arm"].unique() if a != "B0")
    margin = C.GATE_TARGETS["noninferiority_margin"]
    ni, design = {}, {}

    def store(tag: str, est: dict) -> None:
        E[tag] = {k: v for k, v in est.items() if not k.startswith("_")}

    for a in arms:
        p = _pairs(m, a, "B0")
        design[a] = {"matched_sets": int(len(p)), "families": int(p["family_id"].nunique())}
        for y, cond in REMEDY_OUTCOMES:
            est, info, sel = _paired(p, y, cond)
            tag = "H_remedy_RGR_vs_B0" if (a == "RGR" and y == "y_unnecessary") else f"remedy_{a}_vs_B0__{y}"
            store(tag, est)
            design[a][y] = info
            if y == "y_retained":
                # Conservative exact bound: credit observed gains, bound losses by Clopper-Pearson.
                n = info["n_pairs"]
                unresolved_a = int(p[(p["condition"] == "required") &
                                     (p["retained_raw_a"] == "no_majority")].shape[0]) if "retained_raw_a" in p else 0
                na_a = int(p[(p["condition"] == "required") & (p["retained_raw_a"] == "na")].shape[0]) \
                    if "retained_raw_a" in p else 0
                if n:
                    lower = est["n_a1_b0"] / n - _cp_upper(est["n_a0_b1"], n)
                    worst = (est["n_a1_b0"]) / (n + unresolved_a) - \
                        _cp_upper(est["n_a0_b1"] + unresolved_a, n + unresolved_a)
                else:
                    lower = worst = float("nan")
                ni[f"{a}__y_retained"] = {"lower_95_one_sided": lower, "margin": -margin,
                                          "noninferior": bool(lower > -margin) if n else None,
                                          "lower_worst_case_unresolved": worst, "n_pairs": n,
                                          "n_losses": est.get("n_a0_b1"), "n_gains": est.get("n_a1_b0"),
                                          "n_unresolved_arm": unresolved_a, "n_na_arm": na_a,
                                          "method": "exact Clopper-Pearson bound on paired losses"}
            elif y == "y_adequacy" and est.get("_bs") is not None:
                dd = est["_d"]
                lower = est["_bs"].one_sided_lower(
                    lambda w, dd=dd: float((w * dd).sum() / w.sum()) if w.sum() > 0 else float("nan"))
                ni[f"{a}__y_adequacy"] = {"lower_95_one_sided": lower, "margin": -margin,
                                          "noninferior": bool(lower > -margin), "n_pairs": info["n_pairs"],
                                          "method": "family-cluster bootstrap, mean adequacy score (0-2) / 2"}
        sel = p[(p["condition"] != "required") & p["y_unnecessary_a"].notna() & p["y_unnecessary_b"].notna()]
        E[f"level_{a}__y_unnecessary"] = {"estimate": float(sel["y_unnecessary_a"].mean()) if len(sel) else float("nan"),
                                         "n_pairs": int(len(sel))}
        E[f"level_B0_paired_with_{a}__y_unnecessary"] = {
            "estimate": float(sel["y_unnecessary_b"].mean()) if len(sel) else float("nan"), "n_pairs": int(len(sel))}
    if "RGR" in arms:
        for b in (x for x in arms if x != "RGR"):
            p = _pairs(m, "RGR", b)
            for y, cond in REMEDY_OUTCOMES[:3]:
                store(f"rgr_vs_{b}__{y}", _paired(p, y, cond)[0])
    if {"B1", "B2"} <= set(arms):
        store("intervention_B2_vs_B1", _paired(_pairs(m, "B2", "B1"), "y_unnecessary", "nonreq")[0])
    res["noninferiority_vs_B0"] = ni
    res["remedy_design"] = design


def analyse(df: pd.DataFrame, split: str = "locked") -> dict:
    res: dict = {"split": split, "estimates": {}}
    E = res["estimates"]
    d = df[df["split"] == split] if split != "all" else df
    human_design = "sample" in d.columns
    single = d[(d["experiment"] == "single") & (d["arm"] == "B0")]
    if human_design:            # prevalence only from the representative probability sample
        single = single[single["sample"] == "representative_single"]
    single = single.reset_index(drop=True)
    if len(single):
        bs = Bootstrap(single)
        for cond in C.CONDITIONS:
            E[f"prevalence_unnecessary_{cond}"] = bs.run(
                CellMean(single, "y_unnecessary", CELLS, single["condition"] == cond, "ipw"))
        E["incidental_minus_absent"] = _contrast(
            bs, CellMean(single, "y_unnecessary", CELLS, single["condition"] == "incidental", "ipw"),
            CellMean(single, "y_unnecessary", CELLS, single["condition"] == "absent", "ipw"))
        E["attribution_absent"] = bs.run(CellMean(single, "y_attribution", CELLS,
                                                  single["condition"] == "absent", "ipw"))
        E["retention_required"] = bs.run(CellMean(single, "y_retained", CELLS,
                                                  single["condition"] == "required", "ipw"))
    dia = df[df["experiment"] == "dialogue"]
    if human_design:
        dia = dia[dia["sample"] == "representative_dialogue"]
    dia = dia.reset_index(drop=True)
    if len(dia):
        bs = Bootstrap(dia)
        for t in range(1, C.DIALOGUE_SWITCH_TURNS + 1):
            name = "H_dialogue_first_switch" if t == 1 else f"dialogue_switch_{t}"
            E[name] = _contrast(
                bs,
                CellMean(dia, "y_unnecessary", CELLS, (dia["turn"] == t) & (dia["history_arm"] == "identity"), "ipw"),
                CellMean(dia, "y_unnecessary", CELLS, (dia["turn"] == t) & (dia["history_arm"] == "neutral"), "ipw"))
    if human_design:
        matched_remedies(d, E, res)
        return res
    # Automated labels (secondary only): every generated output, cell-mean contrasts.
    sub = d[d["intervention_subset"] & (((d["experiment"] == "single") & (d["arm"] == "B0")) |
                                        d["experiment"].isin(["intervention", "comparators"]))]
    sub = sub.reset_index(drop=True)
    if len(sub) and sub["arm"].nunique() > 1:
        bs = Bootstrap(sub)
        nonreq = sub["condition"] != "required"
        req = sub["condition"] == "required"
        arm = sub["arm"]
        arms = [a for a in sub["arm"].unique() if a != "B0"]
        outcomes = (("y_unnecessary", nonreq), ("y_retained", req), ("y_adequacy", None),
                    ("y_refusal", None), ("n_words", None))

        def cm(y, mask, a):
            m = (arm == a) if mask is None else (mask & (arm == a))
            return CellMean(sub, y, CELLS, m, "ipw")

        for a in sorted(arms):
            for y, mask in outcomes:
                tag = f"remedy_{a}_vs_B0__{y}"
                if a == "RGR" and y == "y_unnecessary":
                    tag = "H_remedy_RGR_vs_B0"
                E[tag] = _contrast(bs, cm(y, mask, a), cm(y, mask, "B0"))
            E[f"level_{a}__y_unnecessary"] = bs.run(cm("y_unnecessary", nonreq, a))
        E["level_B0__y_unnecessary"] = bs.run(cm("y_unnecessary", nonreq, "B0"))
        if "RGR" in arms:
            for b in sorted(x for x in arms if x != "RGR"):
                for y, mask in outcomes[:3]:
                    E[f"rgr_vs_{b}__{y}"] = _contrast(bs, cm(y, mask, "RGR"), cm(y, mask, b))
        if {"B1", "B2"} <= set(arms):
            E["intervention_B2_vs_B1"] = _contrast(bs, cm("y_unnecessary", nonreq, "B2"),
                                                   cm("y_unnecessary", nonreq, "B1"))
        margin = C.GATE_TARGETS["noninferiority_margin"]
        ni = {}
        for a in sorted(arms):
            for y, mask in (("y_retained", req), ("y_adequacy", None)):
                a_, b_ = cm(y, mask, a), cm(y, mask, "B0")
                lower = bs.one_sided_lower(lambda w, a_=a_, b_=b_: a_(w) - b_(w))
                ni[f"{a}__{y}"] = {"lower_95_one_sided": lower, "margin": -margin,
                                   "noninferior": lower > -margin}
        res["noninferiority_vs_B0"] = ni
    return res


def primary_holm(res: dict, audit_path: Path | None = None) -> dict:
    p = {k: v["p_boot"] for k, v in res["estimates"].items()
         if k in ("H_dialogue_first_switch", "H_remedy_RGR_vs_B0")}
    if audit_path and Path(audit_path).exists():
        with open(audit_path, encoding="utf-8") as fh:
            audit = json.load(fh)
        av = audit.get("added_value") or {}
        if av.get("p_boot") is not None:
            # Directional: only an F1 gain over the best baseline counts as support.
            p["H_evaluator"] = av["p_boot"] if (av.get("estimate") or 0) > 0 else 1.0
    return holm(p)


def to_table(res: dict) -> pd.DataFrame:
    rows = [{"estimand": k, **v} for k, v in res["estimates"].items()]
    return pd.DataFrame(rows)
