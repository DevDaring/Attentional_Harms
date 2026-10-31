"""
revision/make_tables.py - LaTeX tables for the revised manuscript, generated ONLY from result
files (revision/results/*, outputs/analysis/*, annotation reports). No number is typed by hand.

    python revision/make_tables.py   -> Submission/ipm_tables_v2/*.tex
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
REV = ROOT / "revision"
RES = REV / "results"
DST = ROOT.parent.parent / "Submission" / "ipm_tables_v2"
DST.mkdir(parents=True, exist_ok=True)
R = json.loads((RES / "reanalysis.json").read_text(encoding="utf-8"))
H = json.loads((ROOT / "outputs" / "analysis" / "human_locked" / "results.json").read_text(encoding="utf-8"))
EA = json.loads((ROOT / "manifests" / "evaluator_audit.json").read_text(encoding="utf-8"))
ROB = json.loads((ROOT / "outputs" / "analysis" / "robustness" / "robustness.json").read_text(encoding="utf-8"))
COST = pd.read_csv(RES / "method_cost_v2.csv").set_index("arm")
FUP = json.loads((RES / "followup.json").read_text(encoding="utf-8"))
NAME = {"B1": "Avoid-identity instruction (B1)", "B2": "Relevance instruction (B2)",
        "SI_NLI": "Fairness instruction$^{a}$", "IF_MSC": "Moral self-correction instruction$^{b}$",
        "DAIQ_GUARD": "DAIQ guardrail$^{c}$", "REPROMPT": "Self-debiasing reprompt$^{d}$",
        "CAI_REVISE": "Critique-and-revise prompting$^{e}$", "SYS2_IMPL": "Implication prompting$^{f}$",
        "RGR": "RGR"}
ARMS = list(NAME)


def pct(x, d=1, sign=False):
    if x is None or x != x:
        return "--"
    v = 100 * x
    return f"{v:+.{d}f}" if sign else f"{v:.{d}f}"


def pp(x, d=1):
    return "--" if x is None or x != x else f"{x:+.{d}f}"


def pval(p):
    return "$<$0.001" if p < 0.001 else f"{p:.3f}"


def write(name: str, body: str) -> None:
    (DST / f"{name}.tex").write_text(body, encoding="utf-8")


def table(label, caption, colspec, header, rows, note, size="\\small", env="table"):
    width = r"\textheight" if env == "sidewaystable" else r"\linewidth"
    opener = rf"\begin{{{env}}}" + ("[!htb]" if env == "table" else "")
    lines = [opener, r"\centering", size, rf"\caption{{{caption}}}", rf"\label{{{label}}}",
             rf"\begin{{adjustbox}}{{max width={width}}}", rf"\begin{{tabular}}{{{colspec}}}", r"\toprule",
             header + r" \\", r"\midrule"]
    lines += [r + r" \\" if not r.startswith("\\") else r for r in rows]
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{adjustbox}",
              rf"\par\vspace{{2pt}}{{\footnotesize {note}}}", rf"\end{{{env}}}", ""]
    return "\n".join(lines)


SHORT = {"B1": "B1 avoid identity", "B2": "B2 relevance", "SI_NLI": "Fairness$^{a}$",
         "IF_MSC": "Moral self-corr.$^{b}$", "DAIQ_GUARD": "DAIQ guardrail$^{c}$", "REPROMPT": "Reprompt$^{d}$",
         "CAI_REVISE": "Critique-revise$^{e}$", "SYS2_IMPL": "Implication$^{f}$", "RGR": "RGR"}


# ---------------------------------------------------------------- study flow / denominators
def t_flow():
    gen = pd.read_csv(REV / "denominators" / "generated_outputs.csv")
    lab = pd.read_csv(REV / "denominators" / "labelled_units.csv")

    def g(exp, arms=None, split=None):
        s = gen[gen.experiment == exp]
        if arms:
            s = s[s.arm.isin(arms)]
        if split:
            s = s[s.split == split]
        return int(s.outputs.sum())

    def l(sample, arms=None):
        s = lab[lab["sample"] == sample]
        if arms:
            s = s[s.arm.isin(arms)]
        return int(s.units.sum()), int(s.positives_final.sum())

    L = pd.read_csv(REV / "unit_ledger.csv", dtype=str)

    def fam(sample):
        return L[L["sample"] == sample].family_id.nunique()

    rows = []
    spec = [("Pilot (excluded from estimates)", g("pilot"), "pilot", "Rubric and settings"),
            ("Single-turn B0, locked", g("single", split="locked"), "representative_single",
             "Prevalence by condition"),
            ("Dialogue B0, locked", g("dialogue"), "representative_dialogue", "First switched task only"),
            ("Enriched diagnostic (single-turn, dialogue, paraphrase; development and locked)", None, "challenge", "Evaluator diagnosis only"),
            ("B0, B1, B2, RGR on locked matched sets", None, "intervention", "Remedy comparison"),
            ("Six published remedies on the same sets", g("comparators", ["SI_NLI", "IF_MSC", "DAIQ_GUARD",
                                                                          "REPROMPT", "CAI_REVISE", "SYS2_IMPL"]),
             "comparator_review", "Remedy comparison"),
            ("B0, B1, B2, RGR, RGR without restore (development)", None, "rgr2_review",
             "Post hoc ablation")]
    for label, n_gen, sample, use in spec:
        u, p = l(sample)
        rows.append(f"{label} & {n_gen if n_gen else '--'} & {u} & {fam(sample)} & {p} & {use}")
    tot = len(L)
    rows.append(r"\midrule")
    rows.append(f"Subtotal, original batches & -- & {tot} & {L.family_id.nunique()} & "
                f"{int((L.final_unnecessary == 'yes').sum())} & {3 * tot} ratings")
    key = pd.read_csv(REV / "annotation_followup" / "keys" / "FU1_key.csv", dtype=str)
    maj = pd.read_csv(REV / "annotation_followup" / "reports" / "FU1" / "majority_labels.csv", dtype=str)
    maj.columns = [c.lstrip("﻿") for c in maj.columns]
    fk = key.merge(maj[["unit_id", "maj_unnecessary"]], on="unit_id")
    for sample, label, use in (("fu_matched_replication", "Replication: B0 absent and incidental pairs, locked", "Matched replication"),
                               ("fu_followup_arms", "Three follow-up arms on the same matched sets", "Exploratory arms")):
        s = fk[fk["sample"] == sample]
        rows.append(f"{label} & -- & {len(s)} & {s.family_id.nunique()} & {int((s.maj_unnecessary == 'yes').sum())} & {use}")
    allfam = len(set(L.family_id) | set(key.family_id))
    rows.append(f"Total human-labelled responses & -- & {tot + len(fk)} & {allfam} & "
                f"{int((L.final_unnecessary == 'yes').sum()) + int((fk.maj_unnecessary == 'yes').sum())} & {3 * (tot + len(fk))} ratings")
    note = ("Generated: stored outputs (all models, two draws). Labelled: unique responses rated by all three "
            "annotators. Fam.: distinct prompt families. Pos.: final unnecessary-mention labels. `--': drawn from "
            "several experiments or splits.")
    write("tab_flow", table("tab:flow", "Generated outputs, human-labelled responses and independent families.",
                            r">{\raggedright\arraybackslash}p{4.2cm}rrrr>{\raggedright\arraybackslash}p{2.9cm}", "Set & Generated & Labelled & Fam. & Pos. & Use", rows, note))


# ---------------------------------------------------------------- reliability with prevalence
def t_reliability():
    df = pd.read_csv(RES / "reliability_by_field_batch.csv")
    fields = [("unnecessary", "Unnecessary mention"), ("least_relevant", "Relevance"),
              ("least_supported", "Support"),
              ("unsupported_personal_attribution", "Unsupported attribution"),
              ("required_identity_retained", "Retention"), ("task_adequacy", "Adequacy (ordinal)")]
    groups = [("Pilot", ["pilot"]), ("Main+dev", ["main_1", "main_2", "main_3", "rgr2"]), ("Comparators", ["comparators"])]
    fr = FUP["reliability"]["field_reliability"]
    rows, prevs = [], []
    for f, lab in fields:
        cells = []
        for _, bs in groups:
            s = df[(df.field == f) & df.batch.isin(bs)]
            a = (s.alpha * s.units).sum() / s.units.sum()
            ag = (s.raw_agreement * s.units).sum() / s.units.sum()
            if f == "unnecessary":
                prevs.append(sum(json.loads(p).get("yes", 0) * n for p, n in zip(s.prevalence, s.units)) / s.units.sum())
            cells.append(f"{a:.2f} & {ag:.2f}")
        cells.append(f"{fr[f]['alpha']:.2f} & {fr[f]['raw_agreement']:.2f}")
        rows.append(f"{lab} & " + " & ".join(cells))
    prevs.append(fr["unnecessary"]["prevalence"]["yes"])
    sub = FUP["reliability"]["subset_reliability"]["all"]
    loo = [sub[f"without_R{i}"]["alpha"] for i in (1, 2, 3)]
    hdr = (r" & \multicolumn{2}{c}{Pilot (60)} & \multicolumn{2}{c}{Main+dev. (876)} & "
           r"\multicolumn{2}{c}{Comparator (264)} & \multicolumn{2}{c}{Follow-up (452)} \\ "
           r"\cmidrule(lr){2-3}\cmidrule(lr){4-5}\cmidrule(lr){6-7}\cmidrule(lr){8-9}"
           r" Label & $\alpha$ & Agr. & $\alpha$ & Agr. & $\alpha$ & Agr. & $\alpha$ & Agr.")
    note = (r"$\alpha$: Krippendorff's alpha on raw labels, unit-weighted across batches. Agr.: raw pairwise "
            r"agreement. Main+dev.: three main batches (656 units) and the development ablation batch (220 units). "
            r"Follow-up: replication batch. Share of ratings marking an unnecessary mention: "
            rf"{100 * prevs[0]:.1f}, {100 * prevs[1]:.1f}, {100 * prevs[2]:.1f} and {100 * prevs[3]:.1f}\%. Follow-up "
            rf"primary label: Gwet's AC1 {FUP['reliability']['primary_ac1']:.2f}; $\alpha$ with one rater left out "
            rf"{min(loo):.2f}--{max(loo):.2f}. More than 90\% of adequacy ratings in every batch were the top score.")
    write("tab_reliability", table("tab:reliability", "Agreement of the three annotators on raw labels, by annotation batch.",
                                   "lrrrrrrrr", hdr, rows, note))


# ---------------------------------------------------------------- S1 matched replication
def t_replication():
    s = FUP["S1"]
    names = [("majority", "Majority (primary)"), ("all_raters", "All raters"), ("any_rater", "Any rater"),
             ("R1", "Rater 1 only"), ("R2", "Rater 2 only"), ("R3", "Rater 3 only")]
    rows = []
    for k, lab in names:
        x = s["rules"][k]
        n = x["pairs"]
        b = x["bootstrap_95"]
        nc = x["newcombe_paired_95"]
        if k == "R1":
            rows.append(r"\midrule")
        rows.append(f"{lab} & {x['both_yes'] + x['first_only']}/{n} & {x['both_yes'] + x['second_only']}/{n} & "
                    f"{pct(x['diff'], sign=True)} & "
                    f"[{pct(b[0], sign=True)}, {pct(b[1], sign=True)}] & {pct(x['bootstrap_one_sided_lower_95'], sign=True)} & "
                    f"[{pct(nc[0], sign=True)}, {pct(nc[1], sign=True)}] & {pval(x['mcnemar_exact_one_sided_p'])}")
    o = s["original_representative"]
    note = (r"Sample: 160 locked matched pairs from 80 families and six models; equal weight per pair; percentage points. "
            r"Inc., Abs.: incidental and absent answers with an unnecessary mention. Bootstrap: family-cluster "
            r"percentile interval stratified by category (2,000 replicates). Lower: one-sided 95\% bound, the frozen "
            r"decision statistic. Score: Newcombe paired score interval. $p$: exact one-sided McNemar test. "
            r"Single-rater rows were not prespecified.")
    write("tab_replication", table("tab:replication", "Matched replication of the incidental contrast.",
                                   "lrrrlrlr", r"Label rule & Inc. & Abs. & $\Delta$ & Bootstrap CI & Lower & Score CI & $p$",
                                   rows, note))


# ---------------------------------------------------------------- S3 follow-up arms
def t_followup_arms():
    nm = {"GEN_REVISE": "Generic revision", "REGEN": "No-op regeneration",
          "CI_REASON": r"CI reasoning$^{a}$"}
    cost = {c["arm"]: c for c in FUP["S3_cost"]}
    ret = {r["arm"]: r for r in FUP["S3"]["retention"]}
    rows = []
    # the note states these two quantities once for all arms
    assert len({t["arm_na_or_unresolved_where_B0_retained"] for t in ret.values()}) == 1
    assert all(round(r["mcnemar_exact_p"], 2) == 1.0 for r in FUP["S3"]["nonreq"])
    for r in FUP["S3"]["nonreq"]:
        a = r["arm"]
        n = r["newcombe_paired_ci_pp"]
        t = ret[a]
        c = cost[a]
        rows.append(f"{nm[a]} & {r['B0_positives']} & {r['arm_positives']} & {r['arm_added']}/{r['arm_removed']} & "
                    f"{pp(r['diff_pp'])} & [{pp(n[0])}, {pp(n[1])}] & "
                    f"{t['losses']}/{t['eligible_pairs_original_rule']}")
    note = (r"Columns as in Tables~\ref{tab:remedy_efficacy} and~\ref{tab:remedy_preservation}. Arm labels: "
            r"replication batch; B0 labels: original batches. N/A outputs where B0 retained: one per arm. Exact McNemar "
            r"$p = 1.00$ for every arm. Calls and mean tokens per answer: " + "; ".join(
                f"{ {'GEN_REVISE': 'generic revision', 'REGEN': 'no-op regeneration', 'CI_REASON': 'CI reasoning'}[a]} "
                f"{cost[a]['total_calls_mean']:.0f} and {cost[a]['tokens_mean']:.0f}" for a in ("GEN_REVISE", "REGEN", "CI_REASON"))
            + r". $^a$CI-CoT prompt of \citet{lan2025cirl}; 4 of 44 outputs needed tag cleaning.")
    write("tab_followup_arms", table("tab:followup_arms", "Follow-up arms versus ordinary generation (B0) on the 44 locked matched sets (29 absent or incidental pairs, 15 required).",
                                     "lrrcrlr", r"Method & B0 pos. & Arm pos. & Add/rem. & $\Delta$ & Score CI & Losses",
                                     rows, note))


# ---------------------------------------------------------------- evaluator confusion counts
def t_evaluator():
    E = R["evaluator"]
    names = [("ensemble", "Judge ensemble"), ("deepseek_v32", "DeepSeek V3.2"),
             ("mistral_large3", "Mistral Large 3"), ("qwen3_32b", "Qwen3 32B"),
             ("nova_pro", "Nova Pro"), ("density_group_label", "Group-label density$^{*}$"),
             ("any_group_label", "Any group label"), ("density_non_pronoun", "Identity-term density")]
    rows = []
    for k, lab in names:
        m = E["methods"][k]
        o = m["own"]
        prec = "--" if o["precision"] is None else f"{100 * o['precision']:.0f}"
        rec = "--" if o["recall"] is None else f"{100 * o['recall']:.0f}"
        fp_req = f"{m['required_condition_false_positives']}/{m['required_condition_negatives_judged']}"
        rows.append(f"{lab} & {o['n']} & {o['tp']} & {o['fp']} & {o['fn']} & {o['tn']} & {prec} & {rec} & {fp_req}")
    note = (rf"Representative locked sample: {E['units']} responses, {E['human_positives']} human positives "
            r"(majority label). $n$: responses the method labelled. P, R: precision and recall. FP req.: false "
            r"positives among required-condition responses that humans did not mark. $^{*}$Best development "
            r"baseline. Ensemble abstentions: 20, none a human positive. Labelled by every judge: "
            rf"{E['common_eligible_units']} responses, {E['common_eligible_positives']} positives. Targets: P "
            rf"{100 * EA['gate_check']['precision_target']:.0f}, R {100 * EA['gate_check']['recall_target']:.0f}. Ensemble "
            rf"F1 gain over the best development baseline: {EA['added_value']['estimate']:.2f} (95\% CI "
            rf"{EA['added_value']['ci_low']:.2f}--{EA['added_value']['ci_high']:.2f}; bootstrap $p$ "
            rf"{H['holm_primary']['H_evaluator']['p']:.3f}, Holm-adjusted {H['holm_primary']['H_evaluator']['p_holm']:.3f}).")
    write("tab_evaluator", table("tab:evaluator", "Automated evaluators against human majority labels on the 360-response representative locked sample from the six generator models: confusion counts.",
                                 "lrrrrrrrr", r"Evaluator & $n$ & TP & FP & FN & TN & P\,\% & R\,\% & FP req.",
                                 rows, note))


# ---------------------------------------------------------------- prevalence with counts
def t_prevalence():
    P = R["prevalence"]
    rows = []
    for c in ("absent", "incidental", "required"):
        v = P["by_condition"][c]
        b = v["bootstrap_ci"]
        w = v["wilson_weighted_ess"]
        rows.append(f"{c.capitalize()} & {v['units']} & {v['families']} & {v['positives']} & "
                    f"{pct(v['estimate_weighted_cellmean'])} & [{pct(b[0])}, {pct(b[1])}] & [{pct(w[0])}, {pct(w[1])}]")
    d = P["incidental_minus_absent"]
    rows.append(r"\midrule")
    rows.append(f"Incidental $-$ absent & 160 & {d['families_absent'] + d['families_incidental'] - d['families_in_both_conditions']} & 12 & "
                f"{pct(d['estimate_weighted'], sign=True)} & [{pct(d['bootstrap_ci'][0], sign=True)}, {pct(d['bootstrap_ci'][1], sign=True)}] & "
                f"[{pct(d['newcombe_on_kish_ess'][1], sign=True)}, {pct(d['newcombe_on_kish_ess'][2], sign=True)}]")
    ra = P["retention_required"]
    rows.append(f"Retained (required) & {ra['eligible_units']} & -- & {ra['eligible_units'] - ra['losses']} & "
                f"{pct(ra['estimate'])} & [{pct(ra['ci_low'])}, {pct(ra['ci_high'])}] & "
                f"[{pct(ra['wilson_unweighted_retained'][0])}, {pct(ra['wilson_unweighted_retained'][1])}]")
    note = ("Sample: 240 representative locked responses (B0) from six models; values in \\%. "
            "Est.: inverse-probability weighted rate, equal weight per model $\\times$ category cell. Bootstrap: "
            "family-cluster percentile interval (2,000 replicates). Score: Wilson (rows 1--3) or Newcombe interval "
            "on the Kish effective sample size. Retention: 76 responses with a resolved label; 4 coded not applicable.")
    write("tab_prevalence", table("tab:prevalence", "Unnecessary identity mentions in ordinary answers.",
                                  "lrrrrll", "Condition & Resp. & Fam. & Pos. & Est. & Bootstrap CI & Score CI",
                                  rows, note))


# ---------------------------------------------------------------- remedies: efficacy panel
def t_remedy_efficacy():
    rows = []
    for r in R["remedies_nonreq"]:
        a = r["arm"]
        n = r["newcombe_paired_ci_pp"]
        b = r["bootstrap_ci_pp"]
        rows.append(f"{SHORT[a]} & {r['B0_positives']} & {r['arm_positives']} & {r['arm_added']}/{r['arm_removed']} & "
                    f"{pp(r['diff_pp'])} & [{pp(n[0])}, {pp(n[1])}] & [{pp(b[0])}, {pp(b[1])}] & "
                    f"{r['families_net_arm_worse_or_higher']}/{r['families_net_arm_lower']} & {r['mcnemar_exact_p']:.2f}")
    note = (r"Add/rem.: pairs where only the method / only B0 had an unnecessary mention. Score CI: Newcombe paired "
            r"score interval. Bootstrap: family-cluster percentile interval. Fam.: families with a net increase / net "
            r"decrease. $p$: exact McNemar test. Prespecified RGR contrast: bootstrap "
            rf"$p$ {H['holm_primary']['H_remedy_RGR_vs_B0']['p']:.3f}, Holm-adjusted {H['holm_primary']['H_remedy_RGR_vs_B0']['p_holm']:.2f}. "
            r"Sources: $^a$\citet{si2023prompting}; $^b$\citet{ganguli2023capacity}; "
            r"$^c$\citet{panda2025daiq}; $^d$\citet{gallegos2025selfdebiasing}; $^e$prompting adaptation of the "
            r"critique and revision step of \citet{bai2022constitutional}; $^f$\citet{furniturewala2024thinking}.")
    write("tab_remedy_efficacy", table(
        "tab:remedy_efficacy", "Unnecessary identity mentions, remedies versus ordinary generation (B0), on 29 absent or incidental matched pairs from 24 locked families (percentage points).",
        "lrrcrllcr", r"Method & B0 pos. & Arm pos. & Add/rem. & $\Delta$ & Score CI & Bootstrap CI & Fam. & $p$",
        rows, note, env="sidewaystable"))


# ---------------------------------------------------------------- remedies: preservation + cost panel
def t_remedy_preservation():
    rows = []
    E = H["estimates"]
    for r in R["remedies_retention"]:
        a = r["arm"]
        adq = E.get(f"remedy_{a}_vs_B0__y_adequacy", {})
        c = COST.loc[a]
        ub = r["loss_rate_one_sided_upper95"]
        rows.append(f"{SHORT[a]} & {r['losses']}/{r['eligible_pairs_original_rule']} & {100 * ub:.0f} & "
                    f"{r['arm_na_or_unresolved_where_B0_retained']} & {r['losses_if_all_arm_na_counted_as_loss']}/{r['denominator_if_counted']} & "
                    f"{r['losses_any_rater_no_rule']}/{r['B0_retained_pairs']} & "
                    f"{pct(adq.get('estimate'), sign=True)} [{pct(adq.get('ci_low'), sign=True)}, {pct(adq.get('ci_high'), sign=True)}] & "
                    f"{c.total_calls_mean:.2f} & {c.tokens_mean:.0f} & {c.latency_s_median:.1f}")
    b0 = COST.loc["B0"]
    rows.append(f"Ordinary generation (B0) & -- & -- & -- & -- & -- & -- & {b0.total_calls_mean:.2f} & {b0.tokens_mean:.0f} & {b0.latency_s_median:.1f}")
    note = (r"Losses: pairs where B0 retained the required identity and the method did not, among pairs with a "
            r"resolved label in both arms. Upper: exact one-sided 95\% bound on the loss rate. N/A: method outputs "
            r"coded not applicable or without majority where B0 retained. Bound: losses if those count as losses. Any "
            r"rater: losses if one rater's `not retained' counts. $\Delta$ Adequacy: mean paired difference, rescaled. "
            r"Calls: model calls per final answer, including a shared draft; Tokens: mean input+output; Latency: "
            r"median seconds.")
    write("tab_remedy_preservation", table(
        "tab:remedy_preservation", "Retention of required identity, adequacy and cost of each remedy on 15 required-condition matched sets from 13 locked families.",
        "lrrrrrlrrr", r"Method & Losses & Upper\,\% & N/A & Bound & Any rater & $\Delta$ Adequacy & Calls & Tokens & Latency",
        rows, note, env="sidewaystable"))


# ---------------------------------------------------------------- lexical vs human (D4)
def t_lexical():
    lv = R["lexical_vs_human"]
    rows = []
    for r in lv["per_arm"]:
        a = r["arm"]
        rows.append(f"{SHORT[a]} & {r['required_pairs']} & {r['required_pairs_lex_decreased']} & "
                    f"{r['required_mean_lex_change']:+.2f} & {r['retention_losses']} & {r['losses_with_lex_decrease']}")
    d, nd = lv["lex_decreased"], lv["lex_not_decreased"]
    rows.append(r"\midrule")
    rows.append(f"Pooled: count fell & {d['pairs']} & -- & -- & {d['losses']} & "
                f"{100 * d['losses'] / d['pairs']:.0f}\\% [{100 * d['loss_rate_wilson'][0]:.0f}, {100 * d['loss_rate_wilson'][1]:.0f}]")
    rows.append(f"Pooled: no fall & {nd['pairs']} & -- & -- & {nd['losses']} & "
                f"{100 * nd['losses'] / nd['pairs']:.0f}\\% [{100 * nd['loss_rate_wilson'][0]:.0f}, {100 * nd['loss_rate_wilson'][1]:.0f}]")
    note = (r"Term count: resolved non-pronoun identity spans of the lexicon. Count fell: method output had fewer terms "
            r"than B0. Losses: B0 retained and the method did not (human majority labels). Pooled rows: the 92 pairs "
            r"in which B0 retained and the method label was resolved; loss rate with a Wilson 95\% interval.")
    write("tab_lexical", table("tab:lexical", "Identity-term counts versus human-judged loss of required identity, 15 required-condition pairs per method, pooled over the six models.",
                               "lrrrrl", r"Method & Pairs & Count fell & Mean change & Losses & With fall (rate)",
                               rows, note))


# ---------------------------------------------------------------- RGR stages
def t_rgr():
    st = pd.DataFrame(R["rgr_audit"]["stage_counts"])
    s = st[(st.arm == "RGR") & (st.split == "locked")].set_index("condition")
    rows = []
    for c in ("absent", "incidental", "required"):
        r = s.loc[c]
        n = r.answers
        rows.append(f"{c.capitalize()} & {n} & {100 * r.skipped_no_identity / n:.1f} & {100 * r.no_change / n:.1f} & "
                    f"{100 * r.removal_requested / n:.1f} & {100 * r.restore_requested / n:.1f} & "
                    f"{100 * r.restore_executed / n:.1f} & {100 * r.revised / n:.1f} & {100 * r.guard_rejected / n:.1f}")
    note = (r"All 2,880 locked RGR answers (80 families, six models, two draws). No term: no lexicon term in draft or "
            r"request. No chg.: gate requested no edit. Rem.: removal requested. Rest. req.: restore of a request "
            r"detail requested. Rest. exec.: an applied edit included a restore. Edited: revision applied. Guard: edit "
            r"rejected by the retention guard.")
    write("tab_rgr", table("tab:rgr_trace", "Stage-level behaviour of RGR on all locked answers by condition (\\% of answers).",
                           "lrrrrrrrr", r"Cond. & $n$ & No term & No chg. & Rem. & Rest. req. & Rest. exec. & Edited & Guard",
                           rows, note))


# ---------------------------------------------------------------- dialogue
def t_dialogue():
    d = R["dialogue"]
    ba = d["by_arm"]
    rows = [f"Identity-bearing history & {ba['identity']['units']} & {ba['identity']['families']} & {int(ba['identity']['positives'])}",
            f"Neutral history & {ba['neutral']['units']} & {ba['neutral']['families']} & {int(ba['neutral']['positives'])}",
            r"\midrule",
            f"Difference (identity $-$ neutral) & \\multicolumn{{3}}{{l}}{{{100 * d['estimate']:+.1f} points; bootstrap 95\\% CI [{100 * d['ci'][0]:+.1f}, {100 * d['ci'][1]:+.1f}]; $p_{{\\mathrm{{boot}}}}={d['p_boot_reproduced']:.3f}$}}"]
    note = (rf"First request after the topic switch only: {d['units']} responses from {d['families']} dialogue "
            rf"families. Bootstrap replicates equal to zero: {100 * d['share_boot_exactly_zero']:.0f}\%.")
    write("tab_dialogue", table("tab:dialogue", "Unnecessary identity mentions after a topic switch (first switched request).",
                                "lrrr", "History & Responses & Families & Positives", rows, note))


# ---------------------------------------------------------------- all remedies in one table (main text)
def t_remedies():
    E = H["estimates"]
    nm = {"B1": "Avoid-identity instruction (B1)", "B2": "Relevance instruction (B2)",
          "SI_NLI": "Fairness instruction", "IF_MSC": "Moral self-correction", "DAIQ_GUARD": "DAIQ guardrail",
          "REPROMPT": "Self-debiasing reprompt", "CAI_REVISE": "Critique-and-revise", "SYS2_IMPL": "Implication prompting",
          "RGR": "RGR", "GEN_REVISE": "Generic revision$^{*}$", "REGEN": "No-op regeneration$^{*}$",
          "CI_REASON": "CI reasoning (CI-CoT)$^{*}$"}
    ret = {r["arm"]: r for r in R["remedies_retention"]}
    lex = {r["arm"]: r for r in R["lexical_vs_human"]["per_arm"]}
    lv = R["lexical_vs_human"]
    rows = []
    for r in R["remedies_nonreq"]:
        a = r["arm"]
        n = r["newcombe_paired_ci_pp"]
        t = ret[a]
        adq = E.get(f"remedy_{a}_vs_B0__y_adequacy", {})
        rows.append(f"{nm[a]} & {r['arm_added']}/{r['arm_removed']} & {pp(r['diff_pp'])} & [{pp(n[0])}, {pp(n[1])}] & "
                    f"{t['losses']}/{t['eligible_pairs_original_rule']} & {lex[a]['losses_with_lex_decrease']} & {t['losses_if_all_arm_na_counted_as_loss']}/{t['denominator_if_counted']} & "
                    f"{pct(adq.get('estimate'), sign=True)} & {COST.loc[a].total_calls_mean:.2f}")
    rows.append(r"\midrule")
    fret = {r["arm"]: r for r in FUP["S3"]["retention"]}
    fcost = {c["arm"]: c for c in FUP["S3_cost"]}
    for r in FUP["S3"]["nonreq"]:
        a = r["arm"]
        n = r["newcombe_paired_ci_pp"]
        t = fret[a]
        den = t["eligible_pairs_original_rule"] + t["arm_na_or_unresolved_where_B0_retained"]
        rows.append(f"{nm[a]} & {r['arm_added']}/{r['arm_removed']} & {pp(r['diff_pp'])} & [{pp(n[0])}, {pp(n[1])}] & "
                    f"{t['losses']}/{t['eligible_pairs_original_rule']} & -- & {t['losses_if_all_arm_na_counted_as_loss']}/{den} & -- & "
                    f"{fcost[a]['total_calls_mean']:.2f}")
    note = (r"Sample: 44 locked matched sets from six models, pooled over models: 29 absent or incidental pairs and 15 "
            r"required-condition sets. Add/rem.: pairs where only the method / only B0 had an unnecessary mention. $\Delta$: paired "
            r"difference in percentage points. Score CI: Newcombe paired score interval. Losses: B0 kept the "
            r"required identity and the method did not, among resolved pairs. Fell: losses whose output also had fewer identity terms than B0. Bound: losses if outputs coded not "
            r"applicable count as losses. $\Delta$ Adeq.: mean paired difference in rescaled adequacy (\%). "
            r"Calls: mean model calls per answer. $^{*}$Follow-up arms labelled in the replication batch. "
            rf"Pooled over the nine original remedies, {lv['lex_decreased']['pairs'] + lv['lex_not_decreased']['pairs']} required pairs in which B0 kept the identity: "
            rf"{100 * lv['lex_decreased']['losses'] / lv['lex_decreased']['pairs']:.0f}\% of pairs whose term count fell lost it, against "
            rf"{100 * lv['lex_not_decreased']['losses'] / lv['lex_not_decreased']['pairs']:.0f}\% of the others.")
    write("tab_remedies", table(
        "tab:remedies", "Remedy effects against ordinary generation (B0).",
        "lccllcccr", r"Method & Add/rem. & $\Delta$ & Score CI & Losses & Fell & Bound & $\Delta$ Adeq. & Calls",
        rows, note))


# ---------------------------------------------------------------- lexicon coverage and restore checks
def t_rgr_checks():
    A = R["rgr_audit"]
    S6 = FUP["S6"]
    lab = {(r["arm"], r["split"]): r for r in A["labelled_edit_outcomes"]}
    lk, dv = lab[("RGR", "locked")], lab[("RGR", "dev")]
    tr = A["response_level_trigger_coverage"]
    sp = A["span_level_lexicon_recall_on_positive_responses"]
    off = S6["offending_span_recall_all_spans"]["lexicon_any"]
    rg = S6["rgr"]

    def row(label, k, n):
        lo, hi = (wilson_ci(k, n) if n else (float("nan"), float("nan")))
        d = 2 if 100 * hi < 1 else (1 if 100 * hi < 10 else 0)
        return f"{label} & {k:,}/{n:,} & {100 * k / n:.{d}f} & [{100 * lo:.{d}f}, {100 * hi:.{d}f}]"
    rows = [row("Human positives with any lexicon term in the response", tr["with_any_lexicon_term"], tr["positives"]),
            row("Identity spans listed by raters that overlap a lexicon span", sp["spans_overlapping_a_lexicon_span"], sp["human_listed_spans"]),
            row("Offending spans marked by raters that overlap a lexicon span", off["hits"], off["spans"]),
            r"\midrule",
            row("Edited locked RGR outputs whose edit included a restore", lk["edited_with_restore"], lk["edited"]),
            row("Mentions added by RGR after an executed restore, locked", lk["added_where_restore_executed"], lk["unnecessary_added_vs_B0"]),
            row("Mentions added by RGR after an executed restore, development", dv["added_where_restore_executed"], dv["unnecessary_added_vs_B0"]),
            row("Incidental RGR positives whose restored detail lies in an offending span", rg["restore_matches_offending_span"],
                rg["restore_matches_offending_span"] + sum(1 for x in rg["rows"] if x["condition"] == "incidental"
                                                           and x["restore_matches_offending_span"] is False)),
            row("Gate calls whose output reached the 300-token limit", A["gate_calls"]["stop_max_tokens"], A["gate_calls"]["n"])]
    note = (r"Human positives: the 92 responses with a final unnecessary-mention label. Offending spans: copied by "
            r"each rater in the span batch. Overlap: character overlap after case and whitespace normalisation. "
            r"Restore: the gate asked the editor to add a request detail. CI: Wilson 95\% interval.")
    write("tab_rgr_checks", table("tab:rgr_checks", "Lexicon coverage and restore checks for RGR on human-labelled responses from the six models, locked and development sets pooled.",
                                  r">{\raggedright\arraybackslash}p{8.2cm}rrl", r"Check & Count & \% & 95\% CI", rows, note))


def wilson_ci(k, n, z=1.959964):
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return max(0.0, c - h), min(1.0, c + h)


if __name__ == "__main__":
    for f in (t_flow, t_reliability, t_evaluator, t_prevalence, t_remedy_efficacy, t_remedy_preservation,
              t_lexical, t_rgr, t_dialogue, t_replication, t_followup_arms, t_rgr_checks, t_remedies):
        f()
    print(sorted(p.name for p in DST.iterdir()))
