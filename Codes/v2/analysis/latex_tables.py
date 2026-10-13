"""
analysis/latex_tables.py - writes every results table of the IPM manuscript from
pipeline outputs into Submission/ipm_tables/. Nothing in these tables is typed
by hand. When an input does not exist yet, the table is written with '--'
cells and a red 'pending' note, so the manuscript always compiles and never
shows invented numbers.

Inputs (paid run, Codes/v2/outputs/ unless --out is given):
  annotation/reports/<batch>/reliability.json   inter-rater reliability
  manifests/evaluator_audit.json                evaluator vs baselines
  outputs/analysis/<labels>_locked/results.json estimands
  outputs/method_cost.csv                       remedy cost
"""
from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

import config_v2 as C

TABLE_DIR = C.CODES_DIR.parent / "Submission" / "ipm_tables"
MODEL_LABEL = {k: v["label"] for k, v in C.MODELS.items()}
METHOD_LABEL = {"B0": "No intervention (B0)", "B1": "Blanket suppression (B1)",
                "B2": "Relevance instruction (B2)", "SI_NLI": "Fairness instruction",
                "IF_MSC": "Moral self-correction", "DAIQ_GUARD": "DAIQ guardrail",
                "REPROMPT": "Self-debiasing reprompt", "CAI_REVISE": "Critique and revise",
                "SYS2_IMPL": "Implication prompting", "RGR": "RGR (proposed)"}
METHOD_CITE = {"SI_NLI": "\\citep{si2023prompting}", "IF_MSC": "\\citep{ganguli2023capacity}",
               "DAIQ_GUARD": "\\citep{panda2025daiq}", "REPROMPT": "\\citep{gallegos2025selfdebiasing}",
               "CAI_REVISE": "\\citep{bai2022constitutional}", "SYS2_IMPL": "\\citep{furniturewala2024thinking}"}
DASH = "--"


def _pct(est: dict | None, signed: bool = False) -> str:
    if not est or est.get("estimate") is None or est["estimate"] != est["estimate"]:
        return DASH
    f = "{:+.1f}" if signed else "{:.1f}"
    e, lo, hi = 100 * est["estimate"], 100 * est["ci_low"], 100 * est["ci_high"]
    return f"{f.format(e)} [{f.format(lo)}, {f.format(hi)}]"


def _wrap(caption: str, label: str, colspec: str, header: str, rows: list[str], note: str,
          pending: bool, size: str = "\\small") -> str:
    pend = "\\pend{Pending: generated from pipeline outputs.} " if pending else ""
    return "\n".join([
        "\\begin{table}[!htb]", "\\centering", size,
        f"\\caption{{{caption} {pend}}}", f"\\label{{{label}}}",
        "\\begin{adjustbox}{max width=\\linewidth}", f"\\begin{{tabular}}{{{colspec}}}", "\\toprule",
        header + " \\\\", "\\midrule", *[r + " \\\\" for r in rows], "\\bottomrule",
        "\\end{tabular}", "\\end{adjustbox}",
        *(["\\par\\vspace{2pt}\\footnotesize " + note] if note else []), "\\end{table}", ""])


def _load_json(path: Path) -> dict | None:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else None


def reliability_table(ann_root: Path) -> str:
    fields = [("unnecessary", "Unnecessary mention (primary)"), ("least_relevant", "Relevance (Q6)"),
              ("least_supported", "Support (Q5)"), ("unsupported_personal_attribution", "Unsupported attribution (Q7)"),
              ("task_adequacy", "Task adequacy (Q8)"), ("required_identity_retained", "Retention (Q9)")]
    reps = {}
    rep_dir = ann_root / "reports"
    for batch_dir in sorted(rep_dir.glob("*")) if rep_dir.exists() else []:
        r = _load_json(batch_dir / "reliability.json")
        if r:
            reps[batch_dir.name] = r
    main = {k: v for k, v in reps.items() if k not in ("pilot", "comparators")}
    comp = reps.get("comparators")

    def _pool(group, f):
        if isinstance(group, dict) and f in group:
            return group[f]
        vals = [g[f] for g in group.values() if f in g]
        n = sum(v["n_units"] for v in vals) or 1
        return {k: sum(v[k] * v["n_units"] for v in vals) / n for k in ("alpha", "raw_agreement", "gwet_ac1")}

    pilot_n = reps["pilot"]["unnecessary"]["n_units"] if "pilot" in reps else 0
    main_n = sum(g["unnecessary"]["n_units"] for g in main.values() if "unnecessary" in g) if main else 0
    comp_n = comp["unnecessary"]["n_units"] if comp else 0

    groups = [reps.get("pilot"), main]
    group_labels = [f"Pilot ({pilot_n} units)", f"Main batches ({main_n} units)"]
    if comp:
        groups.append(comp)
        group_labels.append(f"Comparators ({comp_n} units)")

    rows = []
    for f, label in fields:
        cells = [label]
        for group in groups:
            if not group:
                cells += [DASH, DASH, DASH]
                continue
            s = _pool(group, f)
            cells += [f"{s['alpha']:.2f}", f"{s['raw_agreement']:.2f}",
                      DASH if s["gwet_ac1"] != s["gwet_ac1"] else f"{s['gwet_ac1']:.2f}"]
        rows.append(" & ".join(cells))
    col_spec = "l" + "rrr" * len(groups)
    hdr_cols = []
    cmirules = []
    label_cols = ["Label"]
    for i, gl in enumerate(group_labels):
        start = 2 + i * 3
        hdr_cols.append(f"\\multicolumn{{3}}{{c}}{{{gl}}}")
        cmirules.append(f"\\cmidrule(lr){{{start}-{start + 2}}}")
        label_cols += ["$\\alpha$", "Agree.", "AC1"]
    hdr = " & " + " & ".join(hdr_cols) + " \\\\ " + "".join(cmirules) + " " + " & ".join(label_cols)
    return _wrap("Inter-rater reliability of the human labels before adjudication (three raters).",
                 "tab:reliability", col_spec, hdr,
                 rows, "$\\alpha$: Krippendorff's alpha (ordinal for adequacy, nominal otherwise); Agree.: raw "
                 "pairwise agreement; AC1: Gwet's AC1 (sensitivity check).", not reps)


def evaluator_table(audit_path: Path) -> str:
    audit = _load_json(audit_path)
    names = [("ensemble", "Judge ensemble (two other-family judges)"), ("nova_pro", "Nova Pro judge alone"),
             ("mistral_large3", "Mistral Large 3 judge alone"), ("deepseek_v32", "DeepSeek V3.2 judge alone"),
             ("qwen3_32b", "Qwen3 32B judge alone"), ("novel_group_label", "History-novel group label"),
             ("any_group_label", "Any group label"), ("density_non_pronoun", "Identity density (no pronouns)"),
             ("old_gss", "GSS (May 2026 score)"), ("n_words", "Response length")]
    rows = []
    rep = (audit or {}).get("subsets", {}).get("representative", {}).get("methods", {})
    for key, label in names:
        m = rep.get(key, {})
        cells = [label] + [_pct(m.get(x)) for x in ("precision", "recall", "f1", "fp_rate_required")]
        rows.append(" & ".join(cells))
    note = ("Held-out representative sample from locked families; human majority label as reference; 95\\% "
            "family-cluster bootstrap intervals in brackets. Lexical thresholds were tuned on development units only.")
    if audit and "best_baseline_on_dev" in audit:
        note += f" Best development baseline: {audit['best_baseline_on_dev'].replace('_', ' ')}."
    return _wrap("Agreement of automated evaluators with human labels for unnecessary identity mentions (\\%).",
                 "tab:evaluator", "lrrrr",
                 "Evaluator & Precision & Recall & F1 & FP on required", rows, note, audit is None)


def prevalence_table(results: dict | None) -> str:
    est = (results or {}).get("estimates", {})
    rows = [" & ".join([cond.capitalize(), _pct(est.get(f"prevalence_unnecessary_{cond}"))])
            for cond in C.CONDITIONS]
    rows.append(" & ".join(["Incidental $-$ absent", _pct(est.get("incidental_minus_absent"), True)]))
    rows.append(" & ".join(["Unsupported attribution (absent)", _pct(est.get("attribution_absent"))]))
    rows.append(" & ".join(["Retention (required)", _pct(est.get("retention_required"))]))
    return _wrap("Unnecessary identity mentions by task condition, single-turn B0 outputs (\\%).",
                 "tab:prevalence", "lr", "Quantity & Estimate [95\\% CI]", rows,
                 "Human-labelled representative sample of locked single-turn outputs; inverse-probability "
                 "weighted; equal weight per model and category cell; 95\\% family-cluster bootstrap intervals.",
                 results is None)


def dialogue_table(results: dict | None) -> str:
    est = (results or {}).get("estimates", {})
    rows = []
    for t, name in ((1, "H_dialogue_first_switch"), (2, "dialogue_switch_2"), (3, "dialogue_switch_3")):
        rows.append(" & ".join([f"Switched task {t}" + (" (primary)" if t == 1 else ""),
                                _pct(est.get(name), True)]))
    return _wrap("Carryover of identity from earlier turns: identity history minus neutral history (percentage points).",
                 "tab:dialogue", "lr", "Turn & Difference [95\\% CI]", rows,
                 "80 dialogue families; six models; two trajectories per arm; real replies fed back.",
                 results is None)


def remedy_table(results: dict | None, cost_csv: Path) -> str:
    est = (results or {}).get("estimates", {})
    ni = (results or {}).get("noninferiority_vs_B0", {})
    cost = pd.read_csv(cost_csv) if cost_csv.exists() else pd.DataFrame()
    arms = ["B1", "B2", "SI_NLI", "IF_MSC", "DAIQ_GUARD", "REPROMPT", "CAI_REVISE", "SYS2_IMPL", "RGR"]
    rows = []
    for a in arms:
        un = est.get("H_remedy_RGR_vs_B0") if a == "RGR" else est.get(f"remedy_{a}_vs_B0__y_unnecessary")
        ret = est.get(f"remedy_{a}_vs_B0__y_retained")
        adq = est.get(f"remedy_{a}_vs_B0__y_adequacy")
        ni_ok = ni.get(f"{a}__y_retained", {}).get("noninferior")
        c = cost[cost["arm"] == a] if len(cost) else pd.DataFrame()
        calls = f"{c['extra_calls_mean'].iloc[0]:.2f}" if len(c) else ("0" if a in ("B1", "B2") else DASH)
        label = METHOD_LABEL[a] + (" " + METHOD_CITE[a] if a in METHOD_CITE else "")
        disc = DASH if not un or un.get("n_a1_b0") is None else f"{un['n_a1_b0']}/{un['n_a0_b1']}"
        rows.append(" & ".join([label, _pct(un, True), disc, _pct(ret, True), _pct(adq, True),
                                DASH if ni_ok is None else ("yes" if ni_ok else "no"), calls]))
    design = (results or {}).get("remedy_design", {})
    any_arm = next(iter(design.values()), {})
    n_sets = any_arm.get("matched_sets", "--")
    n_fam = any_arm.get("families", "--")
    n_un = any_arm.get("y_unnecessary", {}).get("n_pairs", "--")
    n_ret = any_arm.get("y_retained", {}).get("n_pairs", "--")
    return _wrap(f"Inference-time remedies compared with ordinary generation (B0) on {n_sets} human-labelled "
                 f"matched sets from {n_fam} locked families.",
                 "tab:remedies", "lrcrrcr",
                 "Method & $\\Delta$ Unnecessary & Disc. & $\\Delta$ Retention & $\\Delta$ Adequacy & NI & Extra calls",
                 rows, "Each remedy output is paired with the B0 output of the same family, condition, model and "
                 "draw. $\\Delta$: mean paired difference, method minus B0, in percentage points, with 95\\% "
                 f"family-cluster bootstrap intervals. Unnecessary mentions: {n_un} pairs on absent and incidental "
                 f"conditions. Disc.: pairs where only the method / only B0 had an unnecessary mention. "
                 f"Retention: {n_ret} required-condition pairs with a resolved label in both arms. Adequacy: mean "
                 "score (0--2) rescaled to 0--100. NI: retention non-inferiority at a 5-point margin by an exact "
                 "one-sided 95\\% bound on paired losses; `no' means not established. Extra calls per answer "
                 "beyond the shared B0 draft. Rows marked `--' have no human labels yet.",
                 results is None, size="\\footnotesize")


CAT_LABEL = {"caste": "Caste", "gender": "Gender", "religion": "Religion", "race_ethnicity": "Race or ethnicity"}


def _p1(x: float) -> str:
    return DASH if x is None or x != x else f"{100 * x:.1f}"


def robustness_table(rob: dict | None) -> str:
    h = (rob or {}).get("headline", {})
    b = (rob or {}).get("breakdown", {})
    rows = []

    def row(label, e):
        rows.append(" & ".join([label, _p1(e.get("incidental")), _p1(e.get("absent")), _pct(e, True)]))

    def span(label, keys):
        es = [h[k] for k in keys if k in h]
        if not es:
            return
        lo = lambda f: min(e[f] for e in es)
        hi = lambda f: max(e[f] for e in es)
        rows.append(" & ".join([label, f"{_p1(lo('incidental'))}--{_p1(hi('incidental'))}",
                                f"{_p1(lo('absent'))}--{_p1(hi('absent'))}",
                                f"{100 * lo('estimate'):+.1f} to {100 * hi('estimate'):+.1f} "
                                f"(lowest bound {100 * lo('ci_low'):+.1f})"]))

    rows.append("\\multicolumn{4}{l}{\\textit{(a) Sensitivity of the pooled contrast}}")
    for k, lab in (("main", "Main analysis (majority, adjudicated)"), ("labels_lenient", "Any rater marks unnecessary"),
                   ("labels_strict", "All three raters mark unnecessary"), ("labels_R1", "Rater R1 alone"),
                   ("labels_R2", "Rater R2 alone"), ("labels_R3", "Rater R3 alone")):
        if k in h:
            row(lab, h[k])
    span("Leave one model out (6 fits)", [k for k in h if k.startswith("drop_model_")])
    span("Leave one category out (4 fits)", [k for k in h if k.startswith("drop_category_")])
    rows.append("\\midrule \\multicolumn{4}{l}{\\textit{(b) Within one category or one model}}")
    for k, lab in CAT_LABEL.items():
        if f"category={k}" in b:
            row(lab, b[f"category={k}"])
    for k, lab in MODEL_LABEL.items():
        if f"model_key={k}" in b:
            row(f"\\textit{{{lab}}}", b[f"model_key={k}"])
    note = ("Human-labelled representative sample of locked single-turn B0 outputs, inverse-probability weighted; "
            "95\\% family-cluster bootstrap intervals. Panel (b) has 20 incidental units per category and 13--14 per "
            "model, so its intervals are wide. These checks are post hoc and outside the Holm family.")
    return _wrap("Robustness of the incidental $-$ absent contrast in unnecessary identity mentions (\\%).",
                 "tab:robustness", "lrrl", "Analysis & Incidental & Absent & Difference [95\\% CI]", rows, note,
                 rob is None)


def ablation_table(rob: dict | None) -> str:
    a = (rob or {}).get("rgr_ablation_dev", {})
    rows = []
    for x, y, lab in (("B1", "B0", "Blanket suppression (B1) $-$ B0"), ("B2", "B0", "Relevance instruction (B2) $-$ B0"),
                      ("RGR", "B0", "RGR $-$ B0"), ("RGR_RO", "B0", "RGR without restore step $-$ B0"),
                      ("RGR", "RGR_RO", "RGR $-$ RGR without restore step")):
        u = a.get(f"{x}_vs_{y}__y_unnecessary", {})
        disc = f"{u.get('n_a1_b0', DASH)}/{u.get('n_a0_b1', DASH)}"
        rows.append(" & ".join([lab, _pct(u, True), disc, _pct(a.get(f"{x}_vs_{y}__y_retained"), True),
                                _pct(a.get(f"{x}_vs_{y}__y_adequacy"), True)]))
    n_u = a.get("RGR_vs_B0__y_unnecessary", {}).get("n_pairs", DASH)
    n_r = a.get("RGR_vs_B0__y_retained", {}).get("n_pairs", DASH)
    des = a.get("design", {})
    note = (f"Development split, {des.get('matched_sets', DASH)} human-labelled matched sets from "
            f"{des.get('families', DASH)} families; {n_u} absent or incidental pairs "
            f"for unnecessary mentions and {n_r} required-condition pairs for retention. Disc.: pairs where only the "
            "first / only the second arm had an unnecessary mention. The variant without the restore step was "
            "defined after the locked results and is exploratory.")
    return _wrap("Ablation of the RGR restore step and instruction baselines on the development split "
                 "(percentage points).", "tab:ablation", "lrcrr",
                 "Contrast & $\\Delta$ Unnecessary & Disc. & $\\Delta$ Retention & $\\Delta$ Adequacy", rows, note,
                 rob is None, size="\\footnotesize")


def rgr_trace_table(rob: dict | None) -> str:
    mech = (rob or {}).get("rgr_mechanism", {})
    rows = []
    for cnd in C.CONDITIONS:
        e = mech.get(f"RGR|locked|{cnd}")
        if not e:
            continue
        rows.append(" & ".join([cnd.capitalize(), str(e["n"]), _p1(e["skipped_no_identity"]), _p1(e["no_change"]),
                                _p1(e["revised"]), _p1(e["remove_any"]), _p1(e["restore_any"]),
                                _p1(e["guard_kept_draft"])]))
    note = ("All RGR answers of the 80 locked intervention families, six models, two draws. No term: neither "
            "draft nor request contained a lexicon term, so no gate call was made. Remove / Restore: the gate "
            "asked to delete a draft term / to add a request detail it judged required. Guard: the edit dropped a "
            "kept term and the draft was returned.")
    return _wrap("Stage-level behaviour of RGR by task condition (\\% of answers).", "tab:rgr_trace", "lrrrrrrr",
                 "Condition & $n$ & No term & No change & Revised & Remove & Restore & Guard", rows, note,
                 rob is None)


def write_all(out_dir: Path, ann_root: Path, manifest_dir: Path) -> list[str]:
    TABLE_DIR.mkdir(parents=True, exist_ok=True)
    # Human labels only.
    results = _load_json(out_dir / "analysis" / "human_locked" / "results.json")
    tables = {
        "tab_reliability.tex": reliability_table(ann_root),
        "tab_evaluator.tex": evaluator_table(manifest_dir / "evaluator_audit.json"),
        "tab_prevalence.tex": prevalence_table(results),
        "tab_dialogue.tex": dialogue_table(results),
        "tab_remedies.tex": remedy_table(results, out_dir / "method_cost.csv"),
    }
    rob = _load_json(out_dir / "analysis" / "robustness" / "robustness.json")
    tables.update({"tab_robustness.tex": robustness_table(rob), "tab_ablation.tex": ablation_table(rob),
                   "tab_rgr_trace.tex": rgr_trace_table(rob)})
    for name, tex in tables.items():
        (TABLE_DIR / name).write_text(tex, encoding="utf-8")
    return list(tables)
