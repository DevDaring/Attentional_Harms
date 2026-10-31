"""
revision/reanalysis.py - Phase B/C/D/E re-analyses on EXISTING human labels (co-author review).

Every analysis here is a NEW REANALYSIS of data collected before 13 Oct 2026. None is
prespecified, and none adds new human labels. Results go to revision/results/.

    python revision/reanalysis.py          (after build_ledger.py)

Sections (review item in brackets):
  A  reproduction of the manuscript estimates from the ledger                  [audit]
  B  denominators per estimate: units, families, positives                     [R02, R04, R14]
  C  boundary-aware / family-level uncertainty for prevalence and remedies     [R03, R10]
  D  retention eligibility and 'na' bounds; exact zero-loss bounds             [R05, s8]
  E  dialogue estimate: bootstrap mass at zero explains p = 0.756              [R10, R14]
  F  evaluator confusion matrices, own vs common denominators, abstentions      [R09]
  G  lexical term counts vs human retention / unnecessary on identical pairs    [R15, D4]
  H  RGR stage audit: trigger coverage, span-level lexicon recall, restores    [R07, R08]
  I  pronoun-coding sensitivity of the incidental contrast                      [R06]
  J  reliability by field and batch, class prevalence                           [s7]
  K  chronology, model logs, truncation, split leakage                          [R13, s6]
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import beta, binomtest, fisher_exact, norm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import config_v2 as C  # noqa: E402
import lexicon_v2 as LX  # noqa: E402
from analysis.bootstrap import Bootstrap, CellMean  # noqa: E402
from remedies import identity_terms  # noqa: E402

REV = ROOT / "revision"
OUT = REV / "results"
OUT.mkdir(parents=True, exist_ok=True)
CELLS = ["model_key", "category"]
PAIR_KEYS = ["family_id", "condition", "model_key", "draw"]
ARMS = ["B1", "B2", "SI_NLI", "IF_MSC", "DAIQ_GUARD", "REPROMPT", "CAI_REVISE", "SYS2_IMPL", "RGR"]
ARM_NAME = {"B1": "Avoid-identity instruction (B1)", "B2": "Relevance instruction (B2)",
            "SI_NLI": "Fairness instruction", "IF_MSC": "Moral self-correction instruction",
            "DAIQ_GUARD": "DAIQ guardrail", "REPROMPT": "Self-debiasing reprompt",
            "CAI_REVISE": "Critique-and-revise prompting", "SYS2_IMPL": "Implication prompting",
            "RGR": "RGR", "RGR_RO": "RGR without restore"}
REPORT: dict = {}


def yn(s: pd.Series) -> pd.Series:
    return s.map({"yes": 1.0, "no": 0.0}).astype(float)


# ------------------------------------------------------------------ interval helpers
def wilson(k: float, n: float, z: float = 1.959964) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return (max(0.0, c - h), min(1.0, c + h))


def cp(k: int, n: int, alpha: float = 0.05) -> tuple[float, float]:
    lo = 0.0 if k == 0 else beta.ppf(alpha / 2, k, n - k + 1)
    hi = 1.0 if k == n else beta.ppf(1 - alpha / 2, k + 1, n - k)
    return float(lo), float(hi)


def cp_upper_one_sided(k: int, n: int, alpha: float = 0.05) -> float:
    return 1.0 if k >= n else float(beta.ppf(1 - alpha, k + 1, n - k))


def newcombe_paired(a: int, b: int, c: int, d: int) -> tuple[float, float, float]:
    """Newcombe (1998) method 10 hybrid score CI for p1 - p2 with paired binary data.
    a: both 1, b: arm 1 / base 0, c: arm 0 / base 1, d: both 0. Difference = (b - c)/n.
    Assumes independent pairs."""
    n = a + b + c + d
    if n == 0:
        return (float("nan"),) * 3
    p1, p2 = (a + b) / n, (a + c) / n
    l1, u1 = wilson(a + b, n)
    l2, u2 = wilson(a + c, n)
    den = math.sqrt((a + b) * (c + d) * (a + c) * (b + d))
    phi = (a * d - b * c) / den if den > 0 else 0.0
    dlt = p1 - p2
    lo = dlt - math.sqrt(max(0.0, (p1 - l1) ** 2 - 2 * phi * (p1 - l1) * (u2 - p2) + (u2 - p2) ** 2))
    hi = dlt + math.sqrt(max(0.0, (u1 - p1) ** 2 - 2 * phi * (u1 - p1) * (p2 - l2) + (p2 - l2) ** 2))
    return dlt, lo, hi


def newcombe_unpaired(k1: float, n1: float, k2: float, n2: float) -> tuple[float, float, float]:
    """Newcombe (1998) method 10 hybrid score CI for independent proportions p1 - p2.
    Fractional k, n are allowed (effective sample sizes)."""
    p1, p2 = k1 / n1, k2 / n2
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    d = p1 - p2
    return d, d - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2), d + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)


def kish(w: np.ndarray) -> float:
    w = np.asarray(w, float)
    return float(w.sum() ** 2 / (w ** 2).sum()) if len(w) else 0.0


def jsonable(o):
    if isinstance(o, (np.floating,)):
        return float(o)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    raise TypeError(type(o))


# ------------------------------------------------------------------ data
def load() -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    L = pd.read_csv(REV / "unit_ledger.csv", dtype=str)
    L["needs_adjudication"] = L["needs_adjudication"].astype(str).str.lower().eq("true")
    L["y_unnecessary"] = yn(L["final_unnecessary"])
    L["y_retained"] = yn(L["final_required_identity_retained"])
    L["y_attribution"] = yn(L["final_unsupported_personal_attribution"])
    L["y_adequacy"] = pd.to_numeric(L["final_task_adequacy"], errors="coerce") / 2.0
    L["ipw"] = 1.0 / pd.to_numeric(L["inclusion_prob"], errors="coerce")
    L["ipw"] = L["ipw"].replace([np.inf], np.nan).fillna(1.0)
    L["draw_i"] = pd.to_numeric(L["draw"], errors="coerce").astype("Int64")
    L["turn_i"] = pd.to_numeric(L["turn"], errors="coerce")
    T = {}
    with open(REV / "data" / "labelled_texts.jsonl", encoding="utf-8") as fh:
        for line in fh:
            r = json.loads(line)
            T[r["request_key"]] = r
    G = pd.read_csv(REV / "data" / "gen_index.csv.gz", dtype=str)
    return L, T, G


def request_of(rec: dict) -> str:
    msgs = rec.get("messages") or []
    users = [m["content"] for m in msgs if m.get("role") == "user"]
    return users[-1] if users else ""


# ------------------------------------------------------------------ A + B + C prevalence
def prevalence(L: pd.DataFrame) -> None:
    rep = L[L["sample"] == "representative_single"].reset_index(drop=True)
    bs = Bootstrap(rep)
    out = {"n_units": len(rep), "n_families": int(rep.family_id.nunique()),
           "n_models": int(rep.model_key.nunique()), "by_condition": {}}
    for cond in C.CONDITIONS:
        s = rep[rep.condition == cond]
        cm = CellMean(rep, "y_unnecessary", CELLS, rep["condition"] == cond, "ipw")
        b = bs.run(cm)
        k = int(s.y_unnecessary.sum())
        n = int(s.y_unnecessary.notna().sum())
        neff = kish(s.loc[s.y_unnecessary.notna(), "ipw"])
        p_w = b["estimate"]
        out["by_condition"][cond] = {
            "units": n, "families": int(s.family_id.nunique()), "positives": k,
            "cells_present": int(s.groupby(CELLS).ngroups), "kish_ess": neff,
            "estimate_weighted_cellmean": p_w, "bootstrap_ci": [b["ci_low"], b["ci_high"]],
            "unweighted": k / n if n else float("nan"), "wilson_unweighted": list(wilson(k, n)),
            "clopper_pearson_unweighted": list(cp(k, n)),
            "wilson_weighted_ess": list(wilson(p_w * neff, neff)),
            "positives_by_model": s.groupby("model_key").y_unnecessary.sum().astype(int).to_dict(),
            "units_by_model": s.groupby("model_key").size().to_dict(),
            "positives_by_category": s.groupby("category").y_unnecessary.sum().astype(int).to_dict(),
        }
    inc = CellMean(rep, "y_unnecessary", CELLS, rep["condition"] == "incidental", "ipw")
    ab = CellMean(rep, "y_unnecessary", CELLS, rep["condition"] == "absent", "ipw")
    b = bs.run(lambda w: inc(w) - ab(w))
    bi, ba = out["by_condition"]["incidental"], out["by_condition"]["absent"]
    nd = newcombe_unpaired(bi["estimate_weighted_cellmean"] * bi["kish_ess"], bi["kish_ess"],
                           ba["estimate_weighted_cellmean"] * ba["kish_ess"], ba["kish_ess"])
    nu = newcombe_unpaired(bi["positives"], bi["units"], ba["positives"], ba["units"])
    f = fisher_exact([[bi["positives"], bi["units"] - bi["positives"]],
                      [ba["positives"], ba["units"] - ba["positives"]]], alternative="two-sided")
    # Do absent and incidental units come from the same families? (paired claim in methods)
    fam_a = set(rep[rep.condition == "absent"].family_id)
    fam_i = set(rep[rep.condition == "incidental"].family_id)
    same_cell = rep[rep.condition.isin(["absent", "incidental"])].groupby(
        ["family_id", "model_key"]).condition.nunique()
    out["incidental_minus_absent"] = {
        "estimate_weighted": b["estimate"], "bootstrap_ci": [b["ci_low"], b["ci_high"]],
        "p_boot": b["p_boot"], "newcombe_on_kish_ess": list(nd), "newcombe_unweighted": list(nu),
        "fisher_exact_unweighted_p": float(f.pvalue),
        "families_absent": len(fam_a), "families_incidental": len(fam_i),
        "families_in_both_conditions": len(fam_a & fam_i),
        "family_model_cells_with_both_conditions": int((same_cell == 2).sum()),
        "note": "Representative units were drawn independently per model x category x condition "
                "cell; the contrast is a between-unit comparison of conditions over the same "
                "family bank, not a within-family paired contrast.",
    }
    # Effective sample size needed for +-3 points at 13% (independent orientation only)
    out["planning_note"] = {"n_for_halfwidth_3pts_at_13pct_independent":
                            math.ceil((1.959964 ** 2) * 0.13 * 0.87 / 0.03 ** 2)}
    attr = CellMean(rep, "y_attribution", CELLS, rep["condition"] == "absent", "ipw")
    ret = CellMean(rep, "y_retained", CELLS, rep["condition"] == "required", "ipw")
    out["attribution_absent"] = bs.run(attr)
    out["retention_required"] = bs.run(ret)
    req = rep[rep.condition == "required"]
    out["retention_required"].update({
        "eligible_units": int(req.y_retained.notna().sum()), "losses": int((req.y_retained == 0).sum()),
        "na_or_unresolved": int(req.y_retained.isna().sum()),
        "wilson_unweighted_retained": list(wilson(int((req.y_retained == 1).sum()), int(req.y_retained.notna().sum())))})
    REPORT["prevalence"] = out


# ------------------------------------------------------------------ remedies
def pairs(m: pd.DataFrame, a: str, b: str = "B0") -> pd.DataFrame:
    left = m[m["arm"] == a].drop_duplicates(PAIR_KEYS[:-1] + ["draw_i"]).set_index(PAIR_KEYS[:-1] + ["draw_i"])
    right = m[m["arm"] == b].drop_duplicates(PAIR_KEYS[:-1] + ["draw_i"]).set_index(PAIR_KEYS[:-1] + ["draw_i"])
    p = left.join(right, how="inner", lsuffix="_a", rsuffix="_b").reset_index()
    return p.rename(columns={"category_a": "category"})


def family_sign_test(sel: pd.DataFrame, ya: str, yb: str) -> dict:
    fam = sel.assign(d=sel[ya] - sel[yb]).groupby("family_id")["d"].sum()
    pos, neg = int((fam > 0).sum()), int((fam < 0).sum())
    p = float(binomtest(min(pos, neg), pos + neg, 0.5).pvalue) if pos + neg else 1.0
    return {"families_net_arm_worse_or_higher": pos, "families_net_arm_lower": neg,
            "families_tied": int((fam == 0).sum()), "exact_sign_test_p": p}


def remedies(L: pd.DataFrame) -> None:
    m = L[L["sample"].isin(["intervention", "comparator_review"]) & (L.split == "locked")].copy()
    rows, ret_rows = [], []
    for a in ARMS:
        p = pairs(m, a)
        nr = p[(p.condition != "required") & p.y_unnecessary_a.notna() & p.y_unnecessary_b.notna()]
        aa = int(((nr.y_unnecessary_a == 1) & (nr.y_unnecessary_b == 1)).sum())
        bb = int(((nr.y_unnecessary_a == 1) & (nr.y_unnecessary_b == 0)).sum())
        cc = int(((nr.y_unnecessary_a == 0) & (nr.y_unnecessary_b == 1)).sum())
        dd = int(((nr.y_unnecessary_a == 0) & (nr.y_unnecessary_b == 0)).sum())
        dlt, lo, hi = newcombe_paired(aa, bb, cc, dd)
        bsb = Bootstrap(nr.reset_index(drop=True))
        dv = (nr.y_unnecessary_a - nr.y_unnecessary_b).to_numpy(float)
        bres = bsb.run(lambda w, dv=dv: float((w * dv).sum() / w.sum()) if w.sum() > 0 else float("nan"))
        rows.append({
            "arm": a, "method": ARM_NAME[a], "matched_sets": len(p), "families": int(p.family_id.nunique()),
            "nonreq_pairs": len(nr), "nonreq_families": int(nr.family_id.nunique()),
            "B0_positives": int(nr.y_unnecessary_b.sum()), "arm_positives": int(nr.y_unnecessary_a.sum()),
            "arm_added": bb, "arm_removed": cc, "diff_pp": 100 * dlt,
            "bootstrap_ci_pp": [100 * bres["ci_low"], 100 * bres["ci_high"]],
            "newcombe_paired_ci_pp": [100 * lo, 100 * hi],
            "mcnemar_exact_p": float(binomtest(min(bb, cc), bb + cc, 0.5).pvalue) if bb + cc else 1.0,
            **family_sign_test(nr, "y_unnecessary_a", "y_unnecessary_b"),
        })
        # retention on required condition
        rq = p[p.condition == "required"]
        raw_a = rq["final_required_identity_retained_a"].fillna("missing")
        raw_b = rq["final_required_identity_retained_b"].fillna("missing")
        elig = rq[rq.y_retained_a.notna() & rq.y_retained_b.notna()]
        loss = int(((elig.y_retained_a == 0) & (elig.y_retained_b == 1)).sum())
        gain = int(((elig.y_retained_a == 1) & (elig.y_retained_b == 0)).sum())
        both0 = int(((elig.y_retained_a == 0) & (elig.y_retained_b == 0)).sum())
        n = len(elig)
        # 'na' or 'no_majority' on the arm while B0 retained: bounds
        b0_yes = rq[rq.y_retained_b == 1]
        arm_unres = b0_yes[b0_yes.y_retained_a.isna()]
        # any-rater rule: arm loses if ANY rater says 'no' while B0 majority yes
        anyno = b0_yes.apply(lambda r: any(r.get(f"R{i}_required_identity_retained_a") == "no" for i in (1, 2, 3)), axis=1)
        ret_rows.append({
            "arm": a, "method": ARM_NAME[a], "required_sets": len(rq), "required_families": int(rq.family_id.nunique()),
            "arm_label_counts": raw_a.value_counts().to_dict(), "B0_label_counts": raw_b.value_counts().to_dict(),
            "eligible_pairs_original_rule": n, "eligible_families": int(elig.family_id.nunique()),
            "losses": loss, "gains": gain, "both_not_retained": both0,
            "net_diff_pp": 100 * (gain - loss) / n if n else float("nan"),
            "loss_rate_cp95": list(cp(loss, n)) if n else None,
            "loss_rate_one_sided_upper95": cp_upper_one_sided(loss, n) if n else None,
            "arm_na_or_unresolved_where_B0_retained": int(len(arm_unres)),
            "losses_if_all_arm_na_counted_as_loss": loss + int(len(arm_unres)),
            "denominator_if_counted": n + int(len(arm_unres)),
            "B0_retained_pairs": int(len(b0_yes)),
            "losses_any_rater_no_rule": int(anyno.sum()),
        })
    REPORT["remedies_nonreq"] = rows
    REPORT["remedies_retention"] = ret_rows
    REPORT["zero_loss_bounds"] = {str(n): 1 - 0.05 ** (1 / n) for n in (9, 10, 11, 12, 29, 44, 59, 60)}
    REPORT["n_for_zero_loss_bound_below_5pct"] = int(math.ceil(math.log(0.05) / math.log(0.95)))
    pd.DataFrame(rows).to_csv(OUT / "remedies_unnecessary.csv", index=False)
    pd.DataFrame(ret_rows).to_csv(OUT / "remedies_retention.csv", index=False)


# ------------------------------------------------------------------ dialogue
def dialogue(L: pd.DataFrame) -> None:
    dia = L[L["sample"] == "representative_dialogue"].reset_index(drop=True)
    bs = Bootstrap(dia)
    a = CellMean(dia, "y_unnecessary", CELLS, (dia.turn_i == 1) & (dia.history_arm == "identity"), "ipw")
    b = CellMean(dia, "y_unnecessary", CELLS, (dia.turn_i == 1) & (dia.history_arm == "neutral"), "ipw")
    est = a(np.ones(len(dia))) - b(np.ones(len(dia)))
    boots = np.array([a(bs.counts[i][bs.inv]) - b(bs.counts[i][bs.inv]) for i in range(bs.reps)])
    fin = boots[np.isfinite(boots)]
    REPORT["dialogue"] = {
        "units": len(dia), "families": int(dia.family_id.nunique()),
        "turns_labelled": dia.turn.value_counts().to_dict(),
        "by_arm": dia.groupby("history_arm").agg(units=("unit_id", "size"), positives=("y_unnecessary", "sum"),
                                                families=("family_id", "nunique")).to_dict("index"),
        "estimate": est, "share_boot_exactly_zero": float((fin == 0).mean()),
        "share_boot_below_zero": float((fin < 0).mean()), "share_boot_above_zero": float((fin > 0).mean()),
        "ci": [float(np.percentile(fin, 2.5)), float(np.percentile(fin, 97.5))],
        "p_boot_reproduced": float(min(1.0, 2 * min((fin <= 0).mean(), (fin >= 0).mean()))),
        "explanation": "p_boot = 2 x min(P*(<=0), P*(>=0)). With a single positive unit, most "
                       "replicates are exactly 0, so P*(>=0) includes the zero mass and the p-value "
                       "is large although the upper percentile is 0. The interval and p-value use "
                       "different tail conventions; neither indicates an error, but the interval "
                       "end at 0.0 is a boundary artefact of resampling one event.",
    }


def dialogue_bank_defect(L: pd.DataFrame) -> None:
    """12 dialogue families carry reviewer edit text ('Replace neutral_opening with: ...') inside the
    neutral opening that was sent to the models. Re-estimate the first-switch contrast without them."""
    d = pd.read_csv(ROOT / "data" / "dialogue_bank.csv", dtype=str)
    bad = sorted(d[d.neutral_opening.str.contains("Replace neutral_opening", na=False)].family_id)
    dia = L[(L["sample"] == "representative_dialogue") & ~L.family_id.isin(bad)].reset_index(drop=True)
    bs = Bootstrap(dia)
    a = CellMean(dia, "y_unnecessary", CELLS, (dia.turn_i == 1) & (dia.history_arm == "identity"), "ipw")
    b = CellMean(dia, "y_unnecessary", CELLS, (dia.turn_i == 1) & (dia.history_arm == "neutral"), "ipw")
    res = bs.run(lambda w: a(w) - b(w))
    REPORT["dialogue_bank_defect"] = {
        "defective_families": bad, "n_defective": len(bad),
        "generated_records_affected": None, "labelled_units_excluded": int((L["sample"] == "representative_dialogue").sum() - len(dia)),
        "units_remaining": len(dia), "families_remaining": int(dia.family_id.nunique()),
        "positives_remaining": int(dia.y_unnecessary.sum()), "estimate_excluding": res}


# ------------------------------------------------------------------ evaluator
def evaluator() -> None:
    u = pd.read_csv(ROOT / "manifests" / "evaluator_audit.units.csv", dtype=str)
    rep = u[u["sample"].isin(["representative_single", "representative_dialogue"])].copy()
    rep["y"] = pd.to_numeric(rep["y"], errors="coerce")
    methods = ["ensemble", "deepseek_v32", "mistral_large3", "nova_pro", "qwen3_32b",
               "density_group_label", "any_group_label", "density_non_pronoun", "old_gss"]
    preds = {mth: pd.to_numeric(rep[f"pred_{mth}"], errors="coerce") for mth in methods}
    common = np.ones(len(rep), bool)
    for mth in ["ensemble", "deepseek_v32", "mistral_large3", "nova_pro", "qwen3_32b"]:
        common &= preds[mth].notna().to_numpy()
    out = {"units": len(rep), "human_positives": int(rep.y.sum()), "common_eligible_units": int(common.sum()),
           "common_eligible_positives": int(rep.y[common].sum()), "methods": {}}
    for mth, pr in preds.items():
        res = {}
        for tag, mask in (("own", pr.notna().to_numpy()), ("common", common & pr.notna().to_numpy())):
            yt, yp = rep.y.to_numpy()[mask], pr.to_numpy()[mask]
            tp = int(((yt == 1) & (yp == 1)).sum()); fp = int(((yt == 0) & (yp == 1)).sum())
            fn = int(((yt == 1) & (yp == 0)).sum()); tn = int(((yt == 0) & (yp == 0)).sum())
            res[tag] = {"n": int(mask.sum()), "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                        "precision": tp / (tp + fp) if tp + fp else None,
                        "recall": tp / (tp + fn) if tp + fn else None,
                        "recall_wilson": list(wilson(tp, tp + fn)) if tp + fn else None}
        res["abstained_or_missing"] = int(pr.isna().sum())
        res["positives_among_abstained"] = int(rep.y[pr.isna()].sum())
        # recall if abstentions are scored as misses (conservative)
        tp_all = int(((rep.y == 1) & (pr == 1)).sum())
        res["recall_abstain_as_miss"] = tp_all / int(rep.y.sum()) if rep.y.sum() else None
        req = (rep.condition == "required") & (rep.y == 0) & pr.notna()
        res["required_condition_negatives_judged"] = int(req.sum())
        res["required_condition_false_positives"] = int((pr[req] == 1).sum())
        out["methods"][mth] = res
    REPORT["evaluator"] = out


def nova_pro_examples(L: pd.DataFrame, T: dict, k: int = 6) -> None:
    """Pull raw Nova Pro judge outputs on representative required-condition units the humans
    labelled not-unnecessary, to audit whether the false positives are rubric or parser errors."""
    u = pd.read_csv(ROOT / "manifests" / "evaluator_audit.units.csv", dtype=str)
    sel = u[(u["sample"] == "representative_single") & (u.condition == "required") &
            (u.y == "0.0") & (u.pred_nova_pro == "1.0")]
    targets = set(sel.request_key)
    found = []
    with open(ROOT / "outputs" / "judgments.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if '"nova_pro"' not in line:
                continue
            r = json.loads(line)
            co = r.get("coords") or {}
            tgt = co.get("target") or r.get("target")
            if r.get("model_key") == "nova_pro" and tgt in targets:
                found.append({"target": tgt, "text": (r.get("text") or "")[:1500],
                              "request": request_of(T.get(tgt, {}))[:400],
                              "response": (T.get(tgt, {}).get("text") or "")[:600]})
                if len(found) >= k:
                    break
    REPORT["nova_pro_examples"] = {"n_required_fp_units": len(targets), "examples": found}


# ------------------------------------------------------------------ lexical vs human (D4)
def lexical_vs_human(L: pd.DataFrame, T: dict) -> None:
    m = L[L["sample"].isin(["intervention", "comparator_review"]) & (L.split == "locked")].copy()

    def nterms(key: str) -> int:
        return len([s for s in LX.find_spans((T.get(key) or {}).get("text") or "")
                    if s["resolved"] and s["layer"] != "pronoun"])

    m["lex_terms"] = m["request_key"].map(nterms)
    rows, per_arm = [], []
    for a in ARMS:
        p = pairs(m, a)
        p["lex_change"] = p.lex_terms_a - p.lex_terms_b
        for _, r in p.iterrows():
            rows.append({"arm": a, "family_id": r.family_id, "condition": r.condition, "model_key": r.model_key,
                         "lex_b0": r.lex_terms_b, "lex_arm": r.lex_terms_a, "lex_change": r.lex_change,
                         "unnecessary_b0": r.y_unnecessary_b, "unnecessary_arm": r.y_unnecessary_a,
                         "retained_b0": r.y_retained_b, "retained_arm": r.y_retained_a,
                         "retained_raw_arm": r.final_required_identity_retained_a,
                         "adequacy_b0": r.y_adequacy_b, "adequacy_arm": r.y_adequacy_a})
        rq = p[p.condition == "required"]
        lost = rq[(rq.y_retained_b == 1) & (rq.y_retained_a == 0)]
        per_arm.append({
            "arm": a, "required_pairs": len(rq),
            "required_pairs_lex_decreased": int((rq.lex_change < 0).sum()),
            "required_mean_lex_change": float(rq.lex_change.mean()),
            "retention_losses": len(lost),
            "losses_with_lex_decrease": int((lost.lex_change < 0).sum()),
            "nonreq_pairs": int((p.condition != "required").sum()),
            "nonreq_mean_lex_change": float(p[p.condition != "required"].lex_change.mean()),
        })
    pr = pd.DataFrame(rows)
    pr.to_csv(OUT / "lexical_vs_human_pairs.csv", index=False)
    rq = pr[(pr.condition == "required") & (pr.retained_b0 == 1) & pr.retained_arm.notna()]
    dec = rq[rq.lex_change < 0]
    nondec = rq[rq.lex_change >= 0]
    REPORT["lexical_vs_human"] = {
        "per_arm": per_arm,
        "required_pairs_B0_retained_arm_labelled": len(rq),
        "lex_decreased": {"pairs": len(dec), "losses": int((dec.retained_arm == 0).sum()),
                          "loss_rate_wilson": list(wilson(int((dec.retained_arm == 0).sum()), len(dec))) if len(dec) else None},
        "lex_not_decreased": {"pairs": len(nondec), "losses": int((nondec.retained_arm == 0).sum()),
                              "loss_rate_wilson": list(wilson(int((nondec.retained_arm == 0).sum()), len(nondec))) if len(nondec) else None},
        "all_losses_had_lex_decrease": bool((rq[rq.retained_arm == 0].lex_change < 0).all()),
        "note": "Lexical count = resolved non-pronoun identity spans (lexicon v2.1). A term-count "
                "evaluation scores a decrease as an improvement.",
    }


# ------------------------------------------------------------------ RGR audit (R07, R08)
def rgr_audit(L: pd.DataFrame, T: dict, G: pd.DataFrame) -> None:
    out = {}
    # R07 response-level trigger coverage across all labelled positives
    pos = L[L.y_unnecessary == 1]
    trig = pos["request_key"].map(lambda k: len(identity_terms((T.get(k) or {}).get("text") or "")) > 0)
    out["response_level_trigger_coverage"] = {"positives": len(pos), "with_any_lexicon_term": int(trig.sum())}
    # span-level: human-listed spans (each rater) vs lexicon spans in the same response
    tot = hit = 0
    miss_examples = []
    for _, r in pos.iterrows():
        text = ((T.get(r.request_key) or {}).get("text") or "")
        lex = {s["surface"].lower() for s in LX.find_spans(text) if s["resolved"]}
        lex_canon = {s["canonical"].lower() for s in LX.find_spans(text) if s["resolved"]}
        spans = set()
        for i in (1, 2, 3):
            v = r.get(f"R{i}_identity_spans")
            if isinstance(v, str) and v.strip() and v.strip().lower() not in ("nan", "none", "na"):
                spans |= {s.strip().lower() for s in re.split(r"[;,|\n]", v) if s.strip()}
        for s in spans:
            tot += 1
            ok = any(x in s or s in x for x in lex | lex_canon)
            hit += ok
            if not ok and len(miss_examples) < 25:
                miss_examples.append({"unit_id": r.unit_id, "human_span": s})
    out["span_level_lexicon_recall_on_positive_responses"] = {
        "human_listed_spans": tot, "spans_overlapping_a_lexicon_span": hit,
        "recall": hit / tot if tot else None, "recall_wilson": list(wilson(hit, tot)) if tot else None,
        "missed_examples": miss_examples,
        "caveat": "Raters listed identity spans per response, not which span was unnecessary; this is "
                  "recall over all human-listed spans in positive responses, not offending-span recall."}
    # R08 trace audit on all stored RGR runs
    g = G[(G.role == "final") & G.arm.isin(["RGR", "RGR_RO"]) & (G.status == "ok")].copy()
    g["restore_n"] = g.rgr_restore.fillna("[]").map(lambda s: len(json.loads(s)) if s else 0)
    g["restore_supp_n"] = g.rgr_restore_suppressed.fillna("[]").map(lambda s: len(json.loads(s)) if s else 0)
    g["remove_n"] = g.rgr_remove.fillna("[]").map(lambda s: len(json.loads(s)) if s else 0)
    g["restore_requested"] = (g.restore_n + g.restore_supp_n) > 0
    g["executed_edit"] = g.rgr_gate.eq("revised")
    g["restore_executed"] = g.executed_edit & (g.restore_n > 0)
    tab = g.groupby(["arm", "split", "condition"]).agg(
        answers=("request_key", "size"),
        skipped_no_identity=("rgr_gate", lambda s: int((s == "skipped_no_identity").sum())),
        gate_unparsed=("rgr_gate", lambda s: int((s == "unparsed_keep_draft").sum())),
        no_change=("rgr_gate", lambda s: int((s == "no_change").sum())),
        removal_requested=("remove_n", lambda s: int((s > 0).sum())),
        restore_requested=("restore_requested", "sum"),
        revised=("executed_edit", "sum"), restore_executed=("restore_executed", "sum"),
        guard_rejected=("rgr_gate", lambda s: int((s == "retention_guard_keep_draft").sum())),
        revision_failed=("rgr_gate", lambda s: int((s == "revision_failed_keep_draft").sum())),
    ).reset_index()
    tab.to_csv(OUT / "rgr_stage_counts.csv", index=False)
    out["stage_counts"] = tab.to_dict("records")
    # Restore content: list requested restore details for incidental answers (locked)
    inc = g[(g.arm == "RGR") & (g.split == "locked") & (g.condition == "incidental") & g.restore_requested]
    out["incidental_restore_examples"] = inc.rgr_restore.head(30).tolist()
    # Labelled RGR outputs: requested / executed / harmful
    lab = L[L.arm.isin(["RGR", "RGR_RO"])].merge(
        g[["request_key", "restore_requested", "executed_edit", "restore_executed", "rgr_gate"]].rename(
            columns={"rgr_gate": "gate2"}), on="request_key", how="left")
    b0 = L[L.arm == "B0"][["family_id", "condition", "model_key", "draw_i", "sample", "y_unnecessary"]]
    lab["pair_sample"] = lab["sample"].map({"intervention": "intervention", "rgr2_review": "rgr2_review"})
    lab = lab.merge(b0.rename(columns={"y_unnecessary": "b0_unnecessary", "sample": "pair_sample"}),
                    on=["family_id", "condition", "model_key", "draw_i", "pair_sample"], how="left")
    rows = []
    for (arm, split), s in lab.groupby(["arm", "split"]):
        ex = s[s.executed_edit == True]  # noqa: E712
        rows.append({"arm": arm, "split": split, "labelled": len(s), "edited": len(ex),
                     "edited_with_restore": int((ex.restore_executed == True).sum()),  # noqa: E712
                     "edited_final_unnecessary": int((ex.y_unnecessary == 1).sum()),
                     "edited_with_restore_final_unnecessary": int(((ex.restore_executed == True) & (ex.y_unnecessary == 1)).sum()),  # noqa: E712
                     "unnecessary_added_vs_B0": int(((s.y_unnecessary == 1) & (s.b0_unnecessary == 0)).sum()),
                     "added_where_restore_executed": int(((s.y_unnecessary == 1) & (s.b0_unnecessary == 0) & (s.restore_executed == True)).sum()),  # noqa: E712
                     "removed_vs_B0": int(((s.y_unnecessary == 0) & (s.b0_unnecessary == 1)).sum())})
    out["labelled_edit_outcomes"] = rows
    # Gate max_tokens 300: truncation of gate output
    gs = G[(G.role == "method_step") & G.system_id.fillna("").str.endswith(":gate")]
    out["gate_calls"] = {"n": len(gs), "stop_max_tokens": int((gs.stop_reason == "max_tokens").sum()),
                         "max_tokens_setting": gs.max_tokens.value_counts().to_dict()}
    out["revise_prompt_includes_history"] = False
    out["intervention_families_have_history"] = False
    REPORT["rgr_audit"] = out


# ------------------------------------------------------------------ pronoun sensitivity (R06)
def pronoun_sensitivity(L: pd.DataFrame) -> None:
    rep = L[L["sample"] == "representative_single"].reset_index(drop=True)
    pos = rep[rep.y_unnecessary == 1]
    pron_only = pos[(pos.maj_group_label_present == "no") & (pos.maj_role_or_context_term_present == "no") &
                    (pos.maj_person_gendered_pronoun == "yes")]
    rep["y_no_pron"] = rep.y_unnecessary
    rep.loc[pron_only.index, "y_no_pron"] = 0.0
    rep["y_union_attr"] = ((rep.y_unnecessary == 1) | (rep.y_attribution == 1)).astype(float)
    bs = Bootstrap(rep)
    res = {"positives": len(pos), "pronoun_only_positives": len(pron_only),
           "positives_by_least_supported": pos.maj_least_supported.value_counts().to_dict(),
           "attribution_yes_units": int((rep.y_attribution == 1).sum()),
           "attribution_yes_and_unnecessary": int(((rep.y_attribution == 1) & (rep.y_unnecessary == 1)).sum())}
    for tag, y in (("exclude_pronoun_only", "y_no_pron"), ("unnecessary_or_unsupported_attribution", "y_union_attr")):
        i = CellMean(rep, y, CELLS, rep.condition == "incidental", "ipw")
        a = CellMean(rep, y, CELLS, rep.condition == "absent", "ipw")
        res[tag] = bs.run(lambda w, i=i, a=a: i(w) - a(w))
    # cross-tab of support vs relevance on all labelled responses (independence of axes)
    res["support_by_relevance_all_units"] = pd.crosstab(L.maj_least_supported, L.maj_least_relevant).to_dict()
    REPORT["pronoun_sensitivity"] = res


# ------------------------------------------------------------------ reliability (J)
def reliability() -> None:
    rows = []
    for b in ["pilot", "main_1", "main_2", "main_3", "rgr2", "comparators"]:
        r = json.loads((ROOT / "annotation" / "reports" / b / "reliability.json").read_text(encoding="utf-8"))
        for f, v in r.items():
            if isinstance(v, dict) and "alpha" in v:
                rows.append({"batch": b, "field": f, "units": v.get("n_units"), "alpha": v.get("alpha"),
                             "level": v.get("alpha_level"), "raw_agreement": v.get("raw_agreement"),
                             "ac1": v.get("gwet_ac1"), "prevalence": json.dumps(v.get("prevalence"))})
    df = pd.DataFrame(rows)
    df.to_csv(OUT / "reliability_by_field_batch.csv", index=False)
    REPORT["reliability_primary"] = df[df.field == "unnecessary"].to_dict("records")
    REPORT["reliability_adequacy"] = df[df.field == "task_adequacy"].to_dict("records")
    REPORT["reliability_retained"] = df[df.field == "required_identity_retained"].to_dict("records")


def adjudication(L: pd.DataFrame) -> None:
    by = L.groupby("batch").agg(units=("unit_id", "size"), needs_adjudication=("needs_adjudication", "sum")).reset_index()
    comp = L[L.batch == "comparators"]
    REPORT["adjudication"] = {
        "by_batch": by.to_dict("records"),
        "comparator_no_majority_primary": int((comp.maj_unnecessary == "no_majority").sum()),
        "comparator_no_majority_retained": int((comp.maj_required_identity_retained == "no_majority").sum()),
        "comparator_no_majority_by_arm_retained": comp[comp.maj_required_identity_retained == "no_majority"].arm.value_counts().to_dict(),
        "note": "Comparator batch: majority vote only; 'no_majority' cells are missing and drop out "
                "of the paired denominators.",
        "positives_unanimous_vs_majority": L[L.y_unnecessary == 1].n_raters_unnecessary_yes.value_counts().to_dict(),
    }


# ------------------------------------------------------------------ chronology / logs / leakage (K)
def chronology(G: pd.DataFrame, L: pd.DataFrame) -> None:
    gates = json.loads((ROOT / "manifests" / "gates.json").read_text(encoding="utf-8"))
    frozen = gates["protocol_frozen"]
    cur = {}
    for name, path in (("protocol", "protocol.md"), ("prompt_bank", "data/prompt_bank.csv"),
                       ("dialogue_bank", "data/dialogue_bank.csv"), ("config", "config_v2.py"),
                       ("remedies", "remedies.py"), ("judges", "judges.py"), ("lexicon", "lexicon_v2.py")):
        p = ROOT / path
        cur[name] = hashlib.sha256(p.read_bytes()).hexdigest() if p.exists() else None
    match = {k: (cur.get(k) == v) for k, v in frozen["hashes"].items() if k in cur}
    g = G[G.role.isin(["generation", "final", "method_step"])].copy()
    g["ts"] = pd.to_datetime(g.ts, errors="coerce", utc=True)
    t = g.groupby(["experiment", "split"]).ts.agg(["min", "max"]).reset_index()
    t["min"] = t["min"].astype(str); t["max"] = t["max"].astype(str)
    models = g[g.role != "final"].groupby("model_key").agg(
        model_ids=("model_id", lambda s: sorted(set(s.dropna()))), regions=("region", lambda s: sorted(set(s.dropna()))),
        first=("ts", "min"), last=("ts", "max"), calls=("request_key", "size"),
        had_reasoning=("had_reasoning", lambda s: int((s == "True").sum()))).reset_index()
    models["first"] = models["first"].astype(str); models["last"] = models["last"].astype(str)
    gen = G[(G.role == "generation") & (G.status == "ok")]
    trunc = gen.groupby(["experiment", "split"]).apply(
        lambda s: pd.Series({"outputs": len(s), "stop_max_tokens": int((s.stop_reason == "max_tokens").sum()),
                             "max_tokens_settings": json.dumps(s.max_tokens.value_counts().to_dict())})).reset_index()
    locked_before_freeze = g[(g.split == "locked") & (g.ts < pd.Timestamp(frozen["set_at"]))]
    # family split consistency / leakage
    fs = G.dropna(subset=["family_id"]).groupby("family_id").split.nunique()
    bank = pd.read_csv(ROOT / "data" / "prompt_bank.csv", dtype=str)
    REPORT["chronology"] = {
        "freeze_set_at_utc": frozen["set_at"], "freeze_note": frozen["note"],
        "frozen_hash_matches_current_file": match,
        "ts_by_experiment_split": t.to_dict("records"),
        "locked_records_before_freeze": int(len(locked_before_freeze)),
        "models": models.to_dict("records"),
        "truncation": trunc.to_dict("records"),
        "families_with_more_than_one_split": int((fs > 1).sum()),
        "bank_columns": bank.columns.tolist()[:40],
        "registration": "No external registry record exists in the repository; the freeze is an "
                        "internal timestamped hash record in manifests/gates.json.",
    }
    # near-duplicate prompts across dev / locked (token Jaccard on incidental prompts)
    txtcol = "prompt"
    bank = bank[bank["condition"] == "incidental"]
    if "split" in bank.columns:
        def toks(s):
            return set(re.findall(r"[a-z]+", str(s).lower()))
        dev = bank[bank.split == "dev"][["family_id", txtcol]].values
        lock = bank[bank.split == "locked"][["family_id", txtcol]].values
        best = []
        for fid, t1 in lock:
            a = toks(t1)
            m = max(((len(a & toks(t2)) / max(1, len(a | toks(t2))), f2) for f2, t2 in dev), default=(0, None))
            best.append((m[0], fid, m[1]))
        best.sort(reverse=True)
        REPORT["chronology"]["max_dev_locked_jaccard"] = best[:8]
        REPORT["chronology"]["jaccard_column"] = txtcol


def main() -> None:
    L, T, G = load()
    prevalence(L)
    remedies(L)
    dialogue(L)
    dialogue_bank_defect(L)
    evaluator()
    nova_pro_examples(L, T)
    lexical_vs_human(L, T)
    rgr_audit(L, T, G)
    pronoun_sensitivity(L)
    reliability()
    adjudication(L)
    chronology(G, L)
    with open(OUT / "reanalysis.json", "w", encoding="utf-8") as fh:
        json.dump(REPORT, fh, indent=1, default=jsonable)
    print("wrote", OUT / "reanalysis.json")


if __name__ == "__main__":
    main()
