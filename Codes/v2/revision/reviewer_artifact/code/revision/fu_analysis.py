"""
revision/fu_analysis.py - frozen follow-up analyses S1, S3 and S6 (prospective_extension_plan.md,
frozen 24 Oct 2026). Run after revision/ingest_followup.py.

    python revision/fu_analysis.py

S1  matched replication of the incidental effect on 160 absent/incidental B0 pairs (H-R1):
    mean paired difference, family-cluster bootstrap stratified by category (2,000 reps,
    seed 20261008), one-sided decision on the bootstrap lower bound; exact McNemar and the
    Newcombe paired score interval as supplementary. Because primary-label alpha < 0.60, the
    all-raters and any-rater rules are co-primary sensitivity analyses (stopping rules).
S3  follow-up arms GEN_REVISE, REGEN and CI_REASON against the stored B0 labels of the 44
    matched sets: exploratory and descriptive, same tables as the nine original arms.
S6  span labels on 92 responses: offending-span recall of the lexicon; for RGR outputs,
    whether a restored detail is an offending span.

Anything not named in the plan is marked "not_prespecified" in the output.
Results: revision/results/followup.json and CSVs next to it.
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import binomtest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "revision"))
import lexicon_v2 as LX  # noqa: E402
from analysis.bootstrap import Bootstrap  # noqa: E402
from reanalysis import (ARM_NAME, cp, cp_upper_one_sided, family_sign_test, jsonable,  # noqa: E402
                        newcombe_paired, wilson)
from remedies import identity_terms  # noqa: E402

REV = ROOT / "revision"
FU = REV / "annotation_followup"
OUT = REV / "results"
SEED = 20261008
FU_ARMS = ["GEN_REVISE", "REGEN", "CI_REASON"]
ARM_NAME = {**ARM_NAME, "GEN_REVISE": "Generic revise-for-quality", "REGEN": "No-op regeneration",
            "CI_REASON": "Contextual-integrity reasoning (CI-CoT)"}
RULES = ["majority", "all_raters", "any_rater"]
REPORT: dict = {}


def load_fu1() -> pd.DataFrame:
    key = pd.read_csv(FU / "keys" / "FU1_key.csv", dtype=str)
    raw = pd.read_csv(FU / "reports" / "FU1" / "raw_labels_long.csv", dtype=str)
    raw.columns = [c.lstrip("﻿") for c in raw.columns]
    maj = pd.read_csv(FU / "reports" / "FU1" / "majority_labels.csv", dtype=str)
    maj.columns = [c.lstrip("﻿") for c in maj.columns]
    wide = raw.pivot_table(index="unit_id", columns="rater",
                           values=["unnecessary", "required_identity_retained", "task_adequacy",
                                   "least_relevant"], aggfunc="first")
    wide.columns = [f"{r}_{f}" for f, r in wide.columns]
    d = key.merge(wide.reset_index(), on="unit_id", how="left").merge(maj, on="unit_id", how="left")
    votes = [d[f"R{i}_unnecessary"] for i in (1, 2, 3)]
    n_yes = sum((v == "yes").astype(int) for v in votes)
    n_lab = sum(v.isin(["yes", "no"]).astype(int) for v in votes)
    d["y_majority"] = d.maj_unnecessary.map({"yes": 1.0, "no": 0.0})
    d["y_all_raters"] = np.where(n_lab == 3, (n_yes == 3).astype(float), np.nan)
    d["y_any_rater"] = np.where(n_lab >= 1, (n_yes >= 1).astype(float), np.nan)
    for i in (1, 2, 3):
        d[f"y_R{i}"] = d[f"R{i}_unnecessary"].map({"yes": 1.0, "no": 0.0})
    d["y_retained"] = d.maj_required_identity_retained.map({"yes": 1.0, "no": 0.0})
    d["y_adequacy"] = pd.to_numeric(d.maj_task_adequacy, errors="coerce") / 2.0
    d["n_yes"] = n_yes
    return d


# ------------------------------------------------------------------ S1
def paired_table(p: pd.DataFrame, ya: str, yb: str) -> dict:
    """ya = incidental (or arm), yb = absent (or B0). Rows with either side missing drop."""
    q = p[p[ya].notna() & p[yb].notna()]
    a = int(((q[ya] == 1) & (q[yb] == 1)).sum())
    b = int(((q[ya] == 1) & (q[yb] == 0)).sum())
    c = int(((q[ya] == 0) & (q[yb] == 1)).sum())
    dd = int(((q[ya] == 0) & (q[yb] == 0)).sum())
    dlt, lo, hi = newcombe_paired(a, b, c, dd)
    return {"pairs": len(q), "families": int(q.family_id.nunique()), "both_yes": a,
            "first_only": b, "second_only": c, "both_no": dd,
            "rate_first": (a + b) / len(q) if len(q) else np.nan,
            "rate_second": (a + c) / len(q) if len(q) else np.nan,
            "wilson_first": list(wilson(a + b, len(q))), "wilson_second": list(wilson(a + c, len(q))),
            "diff": dlt, "newcombe_paired_95": [lo, hi],
            "mcnemar_exact_two_sided_p": float(binomtest(min(b, c), b + c, 0.5).pvalue) if b + c else 1.0,
            "mcnemar_exact_one_sided_p": float(binomtest(b, b + c, 0.5, alternative="greater").pvalue) if b + c else 1.0}


def s1(d: pd.DataFrame) -> None:
    m = d[d["sample"] == "fu_matched_replication"]
    inc = m[m.condition == "incidental"].set_index("pair_id")
    ab = m[m.condition == "absent"].set_index("pair_id")
    cols = [f"y_{r}" for r in RULES] + ["y_R1", "y_R2", "y_R3", "n_yes", "model_key", "unit_id"]
    p = inc[["family_id", "category"] + cols].join(ab[cols], lsuffix="_inc", rsuffix="_abs", how="inner")
    p = p.reset_index()
    out = {"pairs_in_key": int(m.pair_id.nunique()), "pairs_joined": len(p),
           "families": int(p.family_id.nunique()), "categories": p.category.value_counts().to_dict(),
           "models": p.model_key_inc.value_counts().to_dict(),
           "inclusion_prob_values": sorted(m.inclusion_prob.astype(float).round(4).unique().tolist()),
           "rules": {}}
    bs = Bootstrap(p, family_col="family_id", strata_col="category", seed=SEED)
    for rule in RULES + ["R1", "R2", "R3"]:
        ya, yb = f"y_{rule}_inc", f"y_{rule}_abs"
        tab = paired_table(p, ya, yb)
        ok = (p[ya].notna() & p[yb].notna()).to_numpy()
        dv = np.where(ok, (p[ya] - p[yb]).fillna(0).to_numpy(float), 0.0)
        okf = ok.astype(float)

        def stat(w, dv=dv, okf=okf):
            den = (w * okf).sum()
            return float((w * dv).sum() / den) if den > 0 else float("nan")
        b = bs.run(stat)
        lo1 = bs.one_sided_lower(stat, 0.95)
        tab.update({"bootstrap_estimate": b["estimate"], "bootstrap_95": [b["ci_low"], b["ci_high"]],
                    "bootstrap_one_sided_lower_95": lo1, "bootstrap_reps": b["reps_used"],
                    "bootstrap_families": b["n_families"],
                    "H_R1_supported_one_sided_0.05": bool(lo1 > 0),
                    "status": "primary" if rule == "majority" else
                    ("co_primary_sensitivity (alpha < 0.60)" if rule in RULES else "not_prespecified")})
        out["rules"][rule] = tab
    # where the incidental positives sit (descriptive)
    pos = p[p.y_majority_inc == 1]
    out["incidental_positives_by_model"] = pos.model_key_inc.value_counts().to_dict()
    out["incidental_positives_by_category"] = pos.category.value_counts().to_dict()
    out["incidental_positives_families"] = int(pos.family_id.nunique())
    out["absent_positives_by_model"] = p[p.y_majority_abs == 1].model_key_abs.value_counts().to_dict()
    out["incidental_votes_distribution"] = p.n_yes_inc.value_counts().sort_index().to_dict()
    out["absent_votes_distribution"] = p.n_yes_abs.value_counts().sort_index().to_dict()
    # comparison with the original representative estimate (independent samples, 13 Oct)
    rep = json.loads((OUT / "reanalysis.json").read_text(encoding="utf-8"))["prevalence"]["by_condition"]
    out["original_representative"] = {c: {"positives": rep[c]["positives"], "units": rep[c]["units"]}
                                      for c in ("absent", "incidental")}
    p.to_csv(OUT / "followup_s1_pairs.csv", index=False)
    REPORT["S1"] = out


# ------------------------------------------------------------------ S3
def s3(d: pd.DataFrame) -> None:
    L = pd.read_csv(REV / "unit_ledger.csv", dtype=str)
    L = L[(L["sample"] == "intervention") & (L.split == "locked") & (L.arm == "B0")].copy()
    L["y_b0"] = L.final_unnecessary.map({"yes": 1.0, "no": 0.0})
    L["y_ret_b0"] = L.final_required_identity_retained.map({"yes": 1.0, "no": 0.0})
    L["y_adeq_b0"] = pd.to_numeric(L.final_task_adequacy, errors="coerce") / 2.0
    k = ["family_id", "condition", "model_key", "draw"]
    L["draw"] = pd.to_numeric(L.draw, errors="coerce").astype("Int64").astype(str)
    b0 = L.drop_duplicates(k)[k + ["y_b0", "y_ret_b0", "y_adeq_b0", "final_required_identity_retained"]]
    f = d[d["sample"] == "fu_followup_arms"].copy()
    f["draw"] = pd.to_numeric(f.draw, errors="coerce").astype("Int64").astype(str)
    rows, ret_rows = [], []
    prior = pd.read_csv(OUT / "remedies_unnecessary.csv")
    for a in FU_ARMS:
        p = f[f.arm == a].merge(b0, on=k, how="left")
        nr = p[p.condition != "required"]
        tab = paired_table(nr, "y_majority", "y_b0")
        q = nr[nr.y_majority.notna() & nr.y_b0.notna()].reset_index(drop=True)
        dv = (q.y_majority - q.y_b0).to_numpy(float)
        bres = Bootstrap(q, seed=SEED).run(
            lambda w, dv=dv: float((w * dv).sum() / w.sum()) if w.sum() > 0 else float("nan"))
        row = {"arm": a, "method": ARM_NAME[a], "matched_sets": len(p), "b0_matched": int(p.y_b0.notna().sum()),
               "families": int(p.family_id.nunique()), "nonreq_pairs": tab["pairs"],
               "nonreq_families": tab["families"], "B0_positives": tab["both_yes"] + tab["second_only"],
               "arm_positives": tab["both_yes"] + tab["first_only"], "arm_added": tab["first_only"],
               "arm_removed": tab["second_only"], "diff_pp": 100 * tab["diff"],
               "bootstrap_ci_pp": [100 * bres["ci_low"], 100 * bres["ci_high"]],
               "newcombe_paired_ci_pp": [100 * x for x in tab["newcombe_paired_95"]],
               "mcnemar_exact_p": tab["mcnemar_exact_two_sided_p"],
               **family_sign_test(q, "y_majority", "y_b0"),
               "arm_positives_any_rater": int(nr.y_any_rater.sum()),
               "arm_positives_all_raters": int(nr.y_all_raters.sum()),
               "arm_adequacy_mean": float(p.y_adequacy.mean()), "b0_adequacy_mean": float(p.y_adeq_b0.mean()),
               "arm_adequacy_below_2": int((p.y_adequacy < 1).sum()),
               "ci_cleaned": int(p.ci_cleaned.fillna("").astype(str).str.lower().isin(["true", "1", "yes"]).sum()),
               "status": "exploratory_descriptive"}
        rows.append(row)
        rq = p[p.condition == "required"]
        elig = rq[rq.y_retained.notna() & rq.y_ret_b0.notna()]
        loss = int(((elig.y_retained == 0) & (elig.y_ret_b0 == 1)).sum())
        gain = int(((elig.y_retained == 1) & (elig.y_ret_b0 == 0)).sum())
        b0_yes = rq[rq.y_ret_b0 == 1]
        unres = b0_yes[b0_yes.y_retained.isna()]
        anyno = b0_yes.apply(lambda r: any(r.get(f"R{i}_required_identity_retained") == "no" for i in (1, 2, 3)), axis=1)
        n = len(elig)
        ret_rows.append({"arm": a, "method": ARM_NAME[a], "required_sets": len(rq),
                         "required_families": int(rq.family_id.nunique()),
                         "arm_label_counts": rq.maj_required_identity_retained.fillna("missing").value_counts().to_dict(),
                         "eligible_pairs_original_rule": n, "eligible_families": int(elig.family_id.nunique()),
                         "losses": loss, "gains": gain,
                         "both_not_retained": int(((elig.y_retained == 0) & (elig.y_ret_b0 == 0)).sum()),
                         "loss_rate_cp95": list(cp(loss, n)) if n else None,
                         "loss_rate_one_sided_upper95": cp_upper_one_sided(loss, n) if n else None,
                         "arm_na_or_unresolved_where_B0_retained": len(unres),
                         "losses_if_all_arm_na_counted_as_loss": loss + len(unres),
                         "B0_retained_pairs": len(b0_yes), "losses_any_rater_no_rule": int(anyno.sum())})
    REPORT["S3"] = {"nonreq": rows, "retention": ret_rows,
                    "original_arms_for_reference": prior[["arm", "nonreq_pairs", "B0_positives", "arm_positives",
                                                          "diff_pp"]].to_dict("records"),
                    "caveat": "Arm labels come from the FU1 batch (majority vote, alpha 0.31); B0 labels come "
                              "from the original batches (adjudicated). Differences mix arm effects with any "
                              "shift between label batches."}
    pd.DataFrame(rows).to_csv(OUT / "followup_s3_unnecessary.csv", index=False)
    pd.DataFrame(ret_rows).to_csv(OUT / "followup_s3_retention.csv", index=False)


# ------------------------------------------------------------------ S6
def _norm(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").replace("’", "'").replace("‘", "'")).strip().lower()


def _locate(span: str, text: str) -> tuple[int, int] | None:
    t, s = _norm(text), _norm(span).strip(" .,;:!?\"'")
    if not s:
        return None
    i = t.find(s)
    return (i, i + len(s)) if i >= 0 else None


def s6() -> None:
    key = pd.read_csv(FU / "keys" / "FU2_key.csv", dtype=str)
    long = pd.read_csv(FU / "reports" / "FU2" / "raw_spans_long.csv", dtype=str)
    long.columns = [c.lstrip("﻿") for c in long.columns]
    texts = {}
    from annotation.sheets_io import read_sheet
    for pth in sorted((FU / "dist" / "Annotated").glob("FU2_spans_R1__*.xlsx")):
        sh = read_sheet(pth)
        texts = dict(zip(sh.unit_id.astype(str).str.strip(), sh.response.fillna("").astype(str)))
    G = pd.read_csv(REV / "data" / "gen_index.csv.gz", dtype=str)
    G = G[G.role == "final"].drop_duplicates("request_key", keep="last").set_index("request_key")
    span_rows, unit_rows = [], []
    for _, u in key.iterrows():
        text = texts.get(u.unit_id, "")
        nt = _norm(text)
        lex_all = [s for s in LX.find_spans(nt) if s["resolved"]]
        trig = {_norm(x) for x in identity_terms(text)}
        per_rater = {}
        for _, r in long[long.unit_id == u.unit_id].iterrows():
            spans = json.loads(r.spans)
            locs = []
            for sp in spans:
                loc = _locate(sp, text)
                ns = _norm(sp)
                if loc:
                    hit_all = any(s["start"] < loc[1] and s["end"] > loc[0] for s in lex_all)
                    hit_trig = any(s["start"] < loc[1] and s["end"] > loc[0] and _norm(s["surface"]) in trig
                                   for s in lex_all)
                else:                                   # paraphrased span: lexicon run on the span itself
                    in_span = [s for s in LX.find_spans(ns) if s["resolved"]]
                    hit_all = bool(in_span)
                    hit_trig = any(_norm(s["surface"]) in trig or _norm(s["surface"]) in
                                   {_norm(x) for x in identity_terms(sp)} for s in in_span)
                span_rows.append({"unit_id": u.unit_id, "arm": u.arm, "rater": r.rater, "span": sp,
                                  "located_in_response": loc is not None, "lexicon_any_overlap": hit_all,
                                  "rgr_trigger_overlap": hit_trig,
                                  "in_span_terms": "; ".join(s["surface"] for s in LX.find_spans(ns) if s["resolved"])})
                locs.append(loc)
            per_rater[r.rater] = locs
        # consensus: a located span overlapping a span from at least one other rater
        raters_with = [k for k, v in per_rater.items() if v]
        agree = 0
        for a in raters_with:
            for b in raters_with:
                if a < b and any(x and y and x[0] < y[1] and y[0] < x[1] for x in per_rater[a] for y in per_rater[b]):
                    agree += 1
        g = G.loc[u.request_key] if u.request_key in G.index else None
        restore = json.loads(g.rgr_restore) if g is not None and isinstance(g.rgr_restore, str) and g.rgr_restore else []
        restore_hit, restore_raters = None, 0
        if restore:
            rs = [_norm(x) for x in restore]
            for _, r in long[long.unit_id == u.unit_id].iterrows():
                restore_raters += any(any(x in _norm(sp) or _norm(sp) in x for x in rs) for sp in json.loads(r.spans))
            restore_hit = restore_raters > 0
        unit_rows.append({"unit_id": u.unit_id, "arm": u.arm, "condition": u.condition, "sample": u["sample"],
                          "raters_with_span": len(raters_with), "rater_pairs_overlapping": agree,
                          "lexicon_spans_in_response": len(lex_all), "rgr_trigger_terms": len(trig),
                          "rgr_gate": None if g is None else g.rgr_gate,
                          "rgr_restore": json.dumps(restore, ensure_ascii=False) if restore else "",
                          "restore_matches_offending_span": restore_hit,
                          "raters_whose_span_contains_restore": restore_raters})
    S = pd.DataFrame(span_rows)
    U = pd.DataFrame(unit_rows)
    S.to_csv(OUT / "followup_s6_spans.csv", index=False)
    U.to_csv(OUT / "followup_s6_units.csv", index=False)

    def rec(x: pd.DataFrame, col: str) -> dict:
        k, n = int(x[col].sum()), len(x)
        return {"spans": n, "hits": k, "recall": k / n if n else None, "wilson_95": list(wilson(k, n)) if n else None}
    by_rater = {r: {"lexicon_any": rec(g, "lexicon_any_overlap"), "rgr_trigger": rec(g, "rgr_trigger_overlap"),
                    "units_with_span": int(U.unit_id.isin(g.unit_id).sum())} for r, g in S.groupby("rater")}
    # unit-level: units where >= 2 raters gave a span, recall over the union of their located spans
    cons = U[U.raters_with_span >= 2].unit_id
    Sc = S[S.unit_id.isin(cons)]
    rgr = U[U.arm.isin(["RGR", "RGR_RO"])]
    REPORT["S6"] = {
        "units": len(U), "spans_total": len(S), "spans_located_verbatim": int(S.located_in_response.sum()),
        "units_by_raters_with_span": U.raters_with_span.value_counts().sort_index().to_dict(),
        "units_with_overlapping_spans_two_or_more_raters": int((U.rater_pairs_overlapping >= 1).sum()),
        "offending_span_recall_all_spans": {"lexicon_any": rec(S, "lexicon_any_overlap"),
                                            "rgr_trigger": rec(S, "rgr_trigger_overlap")},
        "offending_span_recall_units_with_2plus_raters": {"units": int(len(cons)),
                                                          "lexicon_any": rec(Sc, "lexicon_any_overlap"),
                                                          "rgr_trigger": rec(Sc, "rgr_trigger_overlap")},
        "by_rater": by_rater,
        "missed_spans_examples": S[~S.rgr_trigger_overlap][["unit_id", "rater", "span"]].head(30).to_dict("records"),
        "rgr": {"units": len(rgr), "by_gate": rgr.rgr_gate.fillna("missing").value_counts().to_dict(),
                "with_restore": int((rgr.rgr_restore != "").sum()),
                "restore_matches_offending_span": int((rgr.restore_matches_offending_span == True).sum()),  # noqa: E712
                "restore_no_match": int((rgr.restore_matches_offending_span == False).sum()),  # noqa: E712
                "restore_match_raters": rgr.loc[rgr.restore_matches_offending_span == True,  # noqa: E712
                                                "raters_whose_span_contains_restore"].value_counts().sort_index().to_dict(),
                "rows": rgr[["unit_id", "arm", "condition", "raters_with_span", "rgr_gate", "rgr_restore",
                             "restore_matches_offending_span", "raters_whose_span_contains_restore"]].to_dict("records")},
        "note": "Units are the 92 responses whose original final label was 'unnecessary'. A blank row "
                "means the rater found no unnecessary span. Overlap is character overlap after "
                "whitespace and case normalisation; spans not found verbatim are checked by running "
                "the lexicon on the span text."}


def cost_fu() -> None:
    """Calls, tokens and latency per final answer from the stored follow-up records (same rules as
    method_cost_v2.py: GEN_REVISE reuses the shared B0 draft, so the draft call is counted)."""
    K = ["family_id", "condition", "model_key", "draw"]
    steps = []
    with open(ROOT / "outputs" / "generations.jsonl", encoding="utf-8") as fh:
        for line in fh:
            if not any(a in line for a in FU_ARMS):
                continue
            r = json.loads(line)
            co = r.get("coords") or {}
            if co.get("arm") in FU_ARMS and r.get("role") == "method_step" and r.get("status") == "ok":
                steps.append({"arm": co["arm"], "family_id": co.get("family_id"), "condition": co.get("condition"),
                              "model_key": r.get("model_key"), "draw": str(co.get("draw")),
                              "tok": (r.get("input_tokens") or 0) + (r.get("output_tokens") or 0),
                              "lat": r.get("latency_ms") or 0})
    S = pd.DataFrame(steps)
    G = pd.read_csv(REV / "data" / "gen_index.csv.gz", dtype=str)
    dr = G[(G.role == "generation") & (G.experiment == "single") & (G.split == "locked") & (G.status == "ok")].copy()
    dr["draw"] = pd.to_numeric(dr.draw, errors="coerce").astype("Int64").astype(str)
    for c in ("input_tokens", "output_tokens", "latency_ms"):
        dr[c] = pd.to_numeric(dr[c], errors="coerce").fillna(0)
    dr = dr.groupby(K)[["input_tokens", "output_tokens", "latency_ms"]].first()
    rows = []
    for a, s in S.groupby("arm"):
        m = s.groupby(K).agg(n=("tok", "size"), tok=("tok", "sum"), lat=("lat", "sum"))
        if a == "GEN_REVISE":
            m = m.join(dr, how="left")
            m["n"] += 1
            m["tok"] += m.input_tokens + m.output_tokens
            m["lat"] += m.latency_ms
        rows.append({"arm": a, "answers": len(m), "total_calls_mean": float(m.n.mean()),
                     "tokens_mean": float(m.tok.mean()), "latency_s_median": float(m.lat.median() / 1000),
                     "draft_missing": int(m.get("input_tokens", pd.Series(dtype=float)).isna().sum())})
    REPORT["S3_cost"] = rows
    pd.DataFrame(rows).to_csv(OUT / "followup_cost.csv", index=False)


def main() -> None:
    d = load_fu1()
    s1(d)
    s3(d)
    s6()
    cost_fu()
    REPORT["reliability"] = json.loads((FU / "reports" / "FU1" / "ingest_summary.json").read_text(encoding="utf-8"))
    (OUT / "followup.json").write_text(json.dumps(REPORT, indent=2, default=jsonable), encoding="utf-8")
    r = REPORT["S1"]["rules"]
    for k in ("majority", "all_raters", "any_rater", "R1", "R2", "R3"):
        x = r[k]
        print(f"S1 {k:10s} pairs={x['pairs']} inc={x['rate_first']:.3f} abs={x['rate_second']:.3f} "
              f"diff={100*x['diff']:+.1f} boot95=[{100*x['bootstrap_95'][0]:+.1f},{100*x['bootstrap_95'][1]:+.1f}] "
              f"lo1={100*x['bootstrap_one_sided_lower_95']:+.1f} b/c={x['first_only']}/{x['second_only']} "
              f"mcn1={x['mcnemar_exact_one_sided_p']:.4f} H={x['H_R1_supported_one_sided_0.05']}")
    for x in REPORT["S3"]["nonreq"]:
        print("S3", x["arm"], x["nonreq_pairs"], x["B0_positives"], x["arm_positives"], x["arm_added"],
              x["arm_removed"], round(x["diff_pp"], 1), [round(v, 1) for v in x["newcombe_paired_ci_pp"]])
    for x in REPORT["S3"]["retention"]:
        print("S3ret", x["arm"], x["eligible_pairs_original_rule"], x["losses"], x["gains"],
              x["arm_na_or_unresolved_where_B0_retained"], x["arm_label_counts"])
    s = REPORT["S6"]
    print("S6", json.dumps({k: s[k] for k in ("units", "spans_total", "spans_located_verbatim",
                                              "units_by_raters_with_span", "offending_span_recall_all_spans",
                                              "offending_span_recall_units_with_2plus_raters")}, default=jsonable))
    print("S6 rgr", json.dumps({k: v for k, v in s["rgr"].items() if k != "rows"}, default=jsonable))


if __name__ == "__main__":
    main()
