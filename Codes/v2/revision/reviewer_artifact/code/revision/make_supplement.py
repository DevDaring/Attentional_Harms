"""
revision/make_supplement.py - Submission/IPM_Supplement.tex from repository files only.

S1 literature details (from literature_audit.md matrix), S2 worked-example selection rules and full
texts (results/worked_examples.json + select_examples.py rules), S3 remedy prompts verbatim (remedies.py),
S4 chronology and deviations (manifests/gates.json, deviations_post_freeze.md),
Tables S1-S5 (robustness, ablation, reliability per batch, per-judge bootstrap metrics, follow-up arms).
"""
from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import remedies as RM  # noqa: E402
import config_v2 as C  # noqa: E402

REV = ROOT / "revision"
SUB = ROOT.parent.parent / "Submission"


def ascii_text(s: str) -> str:
    rep = {"\u2022": "-", "\u2013": "-", "\u2014": "-", "\u2019": "'", "\u2018": "'", "\u201c": '"',
           "\u201d": '"', "\u2026": "...", "\u00a0": " ", "\ufffd": "'"}
    for a, b in rep.items():
        s = s.replace(a, b)
    return s.encode("ascii", "ignore").decode()


def tex_escape(s: str) -> str:
    s = ascii_text(s)
    s = s.replace("\\", r"\textbackslash{}")
    for ch in "&%$#_{}":
        s = s.replace(ch, "\\" + ch)
    s = s.replace("~", r"\textasciitilde{}").replace("^", r"\textasciicircum{}")
    s = s.replace("**", "")
    return s


def verb(s: str) -> str:
    return "\\begin{verbatim}\n" + "\n".join(
        line[i:i + 95] for line in ascii_text(s).split("\n") for i in range(0, max(1, len(line)), 95)) + \
        "\n\\end{verbatim}\n"


def old_table(path: Path, new_label: str) -> str:
    t = path.read_text(encoding="utf-8")
    return re.sub(r"\\label\{[^}]*\}", "\\\\label{" + new_label + "}", t)


ex = json.loads((REV / "results" / "worked_examples.json").read_text(encoding="utf-8"))


def rec_block(title: str, r: dict) -> str:
    lab = r["labels"]
    return (f"\\paragraph{{{tex_escape(title)}}} Unit {tex_escape(r['unit_id'])}, family {tex_escape(r['family_id'])}, "
            f"{tex_escape(r['condition'])} condition, model \\texttt{{{tex_escape(r['model'])}}}, arm {tex_escape(r['arm'])}, "
            f"draw {tex_escape(str(r['draw']))}. Final labels: unnecessary = {lab['unnecessary']}, retained = "
            f"{lab['required_identity_retained']}, adequacy = {lab['task_adequacy']}, least supported = "
            f"{lab['least_supported']}, least relevant = {lab['least_relevant']}; raters marking unnecessary: "
            f"{lab['raters_unnecessary_yes']} of 3.\n\n\\textit{{Request.}}" + verb(r["request"] or "") +
            "\\textit{Response.}" + verb(r["response"] or ""))


inc = None
L = __import__("pandas").read_csv(REV / "unit_ledger.csv", dtype=str)
T = {json.loads(l)["request_key"]: json.loads(l) for l in open(REV / "data" / "labelled_texts.jsonl", encoding="utf-8")}
r1 = L[(L["sample"] == "representative_single") & (L.condition == "incidental") &
       (L.final_unnecessary == "yes")].sort_values(["family_id", "unit_id"]).iloc[0]
e1 = {"unit_id": r1.unit_id, "family_id": r1.family_id, "condition": r1.condition, "model": r1.model_key,
      "arm": r1.arm, "draw": r1.draw, "request": T[r1.request_key]["messages"][-1]["content"],
      "response": T[r1.request_key]["text"],
      "labels": {"unnecessary": r1.final_unnecessary, "required_identity_retained": r1.final_required_identity_retained,
                 "task_adequacy": r1.final_task_adequacy, "least_supported": r1.maj_least_supported,
                 "least_relevant": r1.maj_least_relevant, "raters_unnecessary_yes": r1.n_raters_unnecessary_yes}}

prompts = [
    ("B0 system instruction (all arms)", C.SYSTEM_PROMPTS["B0"]),
    ("B1 avoid-identity instruction", C.SYSTEM_PROMPTS.get("B1", "")),
    ("B2 relevance instruction", C.SYSTEM_PROMPTS.get("B2", "")),
    ("Fairness instruction (Si et al.; ADAPTED: multiple-choice ending removed)", RM.SI_TEXT),
    ("Moral self-correction (Ganguli et al.)", RM.IF_TEXT),
    ("DAIQ guardrail (Panda et al.; ADAPTED: 'respond with Unknown' removed)", RM.DAIQ_TEXT),
    ("Self-debiasing reprompt (Gallegos et al.; ADAPTED: single-letter ending removed)", RM.REPROMPT_TEXT),
    ("Critique request (Bai et al.)", RM.CAI_CRITIQUE),
    ("Revision request (Bai et al.; ADAPTED: 'Return only the rewritten response.' added)", RM.CAI_REVISION),
    ("Implication step (Furniturewala et al.)", RM.SYS2_IMPLICATION),
    ("Implication rewrite (Furniturewala et al.)", RM.SYS2_FINAL),
    ("RGR gate system", RM.RGR_GATE_SYSTEM),
    ("RGR gate user template", RM.RGR_GATE_USER),
    ("RGR revision template", RM.RGR_REVISE_USER),
    ("Follow-up control GEN_REVISE (generated 24 Oct 2026; labelled in FU1)", RM.GEN_REVISE_TEXT),
    ("Follow-up comparator CI_REASON system text (Lan et al. 2025, Fig. 2, verbatim box contents; labelled in FU1)",
     RM.CI_REASON_SYSTEM),
]

gates = json.loads((ROOT / "manifests" / "gates.json").read_text(encoding="utf-8"))
dev_rows = []
for line in (ROOT / "deviations_post_freeze.md").read_text(encoding="utf-8").splitlines():
    if line.startswith("| 2026"):
        cells = [c.strip() for c in line.strip("|").split("|")]
        # The supplement is read by double-blind reviewers: drop internal review-file names and
        # people, and describe repository folders by function, not by drafting history.
        text = re.sub(r"Revision audit \(co-author review, `?[A-Za-z_]+\.md`?\)", "Re-analysis audit", cells[1])
        text = text.replace("in `revision/`", "in the analysis archive")
        dev_rows.append((cells[0], text))
R = json.loads((REV / "results" / "reanalysis.json").read_text(encoding="utf-8"))
chron = R["chronology"]

out = [r"""\documentclass[11pt]{article}
\usepackage[a4paper,margin=2.3cm]{geometry}
\usepackage[T1]{fontenc}
\usepackage{lmodern}
\usepackage{adjustbox}
\usepackage{booktabs}
\usepackage{array}
\usepackage{graphicx}
\usepackage{hyperref}
\usepackage{microtype}
\usepackage{multirow}
\usepackage{caption}
\usepackage{rotating}
\usepackage{algorithm}
\usepackage{algpseudocode}
\newcolumntype{L}[1]{>{\raggedright\arraybackslash}p{#1}}
\usepackage[authoryear,round]{natbib}
\renewcommand{\thetable}{S\arabic{table}}
\renewcommand{\thesection}{S\arabic{section}}
\title{Supplementary Material\\[4pt]\large IRIS: Evaluating Identity Relevance and Information Retention in Generated Documents}
\date{}
\begin{document}
\maketitle
"""]
# S1
out.append(r"""\section{Closest prior studies: details behind Table~1}
Each study was read in full on 24 October 2026 from the proceedings or arXiv version named in the repository file \texttt{revision/literature\_audit.md}, which gives section, table and page references for every cell.
\begin{itemize}
\item \textbf{ConfAIde} \citep{mireshghallah2024confaide}. Tier~4 asks for meeting summaries and action items that exclude a secret while including required public information; leakage and omission are scored by string match (Sec.~3.4, Table~4). A chain-of-thought suffix raised GPT-4's omission of required public information in summaries from 0.10 to 0.21 (Table~5). Human labels (five crowd workers) cover Tiers~1--3 only, with $\alpha$ between 0.07 and 0.34.
\item \textbf{PrivacyLens} \citep{shao2024privacylens}. Final actions of tool-using agents are scored for leakage of seeded information against privacy norms, and helpfulness is scored 0--3 by a model judge (Sec.~5.1). A privacy-enhancing prompt did not markedly change leakage (Table~2).
\item \textbf{CI-CoT and CI-RL} \citep{lan2025cirl}. Each task defines required and restricted information; Utility requires all required items and Integrity excludes restricted ones (Sec.~4). The prompt-only CI-CoT method lowered leakage with a modest drop in helpfulness on PrivacyLens (Table~3). Data are synthetic, without human labels for the authors' own set.
\item \textbf{CIMemories} \citep{mireshghallah2026cimemories}. Drafted messages are scored for violation (inappropriate memory attributes revealed) and completeness (necessary attributes revealed), relative to task and recipient (Defs.~3.1--3.2). Conservative prompts reduced violations at a cost in completeness (Fig.~4c). Labels come from a model with persona consensus, validated by three raters on 100 pairs.
\item \textbf{Data minimisation} \citep{zhou2026datamin} transforms the user's input, not the output. \textbf{PrivaCI-Bench} \citep{li2025privacibench} and \textbf{CASE-Bench} \citep{sun2025casebench} classify information flows or safety decisions and do not score generated content.
\item \textbf{RUTEd} \citep{lum2025bias}: binary gender only, coded from pronoun counts; no human labels. \textbf{DAIQ} \citep{panda2025daiq}: an explicit inference probe; the guardrail is one instruction inside that probe, adapted here to drafting. \textbf{DART} \citep{pan2026dart}: classification with short rationales; open-ended transfer judged by GPT-4; two researchers labelled 200 pairs for harm drift only; an inference-time explanation policy accompanies the training method.
\end{itemize}
None of these studies covers caste, and none labels identity that a model introduces without support in the input.
""")
# S2
out.append(r"""\section{Worked examples: selection rules and full texts}
Examples were extracted by \texttt{revision/select\_examples.py}. Rules, applied in family then unit order with the first match taken: (1) first incidental-condition positive in the representative sample; (2) first locked pair in which ordinary generation retained the required identity and a remedy did not, searched in the order implication prompting, critique-and-revise, reprompt, moral self-correction, B1; (3) first locked labelled RGR output with an executed restore and a human positive that the paired ordinary answer did not have. Non-ASCII symbols (emoji, typographic quotes) are shown as ASCII.
""")
out.append(rec_block("Example 1: incidental condition, ordinary answer", e1))
e4 = ex["E4_retention_loss"]
out.append(rec_block("Example 2a: required condition, ordinary answer", e4["b0"]))
out.append(rec_block(f"Example 2b: same matched set, {e4['arm']}", e4["remedy"]))
e5 = ex["E5_rgr_restore"]
out.append(rec_block("Example 3a: incidental condition, ordinary answer", e5["b0"]))
out.append(rec_block(f"Example 3b: same matched set, RGR (gate restore list {e5['restore']})", e5["rgr"]))
# S3
out.append("\\section{Remedy prompts}\nPrompts are copied from \\texttt{remedies.py} and \\texttt{config\\_v2.py}. Adaptations are marked. Curly-brace fields are filled at run time.\n")
for title, txt in prompts:
    if txt:
        out.append(f"\\paragraph{{{tex_escape(title)}}}" + verb(txt))
# S4
out.append("\\section{Chronology and deviations}\n")
out.append(f"The protocol freeze was recorded at {tex_escape(chron['freeze_set_at_utc'])} with SHA-256 hashes of the protocol, banks, configuration, prices, remedy code, judge code, lexicon and label schema. ")
match = chron["frozen_hash_matches_current_file"]
same = ", ".join(k for k, v in match.items() if v)
diff = ", ".join(k for k, v in match.items() if not v)
out.append(f"At revision the files {tex_escape(same)} still matched the frozen hashes; {tex_escape(diff)} did not. ")
out.append(f"No locked-split record predates the freeze ({chron['locked_records_before_freeze']} records). ")
out.append("No external registration exists. Post-freeze deviations, verbatim from the repository log (dates in IST):\n\\begin{itemize}\n")
for d, txt in dev_rows:
    out.append(f"\\item \\textbf{{{tex_escape(d)}}}: {tex_escape(re.sub(r'`', '', txt))}\n")
out.append("\\end{itemize}\n")
# Tables
out.append("\\section{Additional tables}\n")
out.append(old_table(SUB / "archive" / "v2026-10-13" / "ipm_tables" / "tab_robustness.tex", "tab:S1"))
out.append(old_table(SUB / "archive" / "v2026-10-13" / "ipm_tables" / "tab_ablation.tex", "tab:S2"))
out.append(old_table(SUB / "archive" / "v2026-10-13" / "ipm_tables" / "tab_evaluator.tex", "tab:S3").replace(
    "\\caption{Agreement of automated evaluators with human labels for unnecessary identity mentions (\\%). }",
    "\\caption{Evaluator metrics on each method's own labelled responses, with family-cluster bootstrap intervals (\\%). Denominators differ by judge (see the evaluator table in the main text). `GSS': share of identity words in a response, an identity-word density score included for comparison.}"))
# reliability per batch with AC1
import pandas as pd  # noqa: E402
rb = pd.read_csv(REV / "results" / "reliability_by_field_batch.csv")
rows = []
for f in ["unnecessary", "least_relevant", "least_supported", "required_identity_retained", "task_adequacy"]:
    for b in ["pilot", "main_1", "main_2", "main_3", "rgr2", "comparators"]:
        x = rb[(rb.field == f) & (rb.batch == b)].iloc[0]
        ac1 = "--" if pd.isna(x.ac1) else f"{x.ac1:.2f}"
        rows.append(f"{tex_escape(f)} & {tex_escape(b)} & {int(x.units)} & {x.alpha:.2f} & {x.raw_agreement:.2f} & {ac1} \\\\")
out.append("\\begin{table}[!htb]\\centering\\small\\caption{Reliability by field and batch before adjudication.}\\label{tab:S4}\n"
           "\\begin{tabular}{llrrrr}\\toprule Field & Batch & Units & $\\alpha$ & Agree. & AC1 \\\\ \\midrule\n" +
           "\n".join(rows) + "\n\\bottomrule\\end{tabular}\\end{table}\n")
out.append(r"""\paragraph{Follow-up arms.} On 24 October 2026, three exploratory arms were generated on the 44 locked matched sets under a separate exploratory gate that leaves the original freeze record unchanged: a generic revise-for-quality control on the shared draft, a no-op regeneration control, and the CI-CoT prompt of \citet{lan2025cirl}. They were labelled in the blinded follow-up batch FU1 (Section~S6).
""")
# S6 follow-up details
FUP = json.loads((REV / "results" / "followup.json").read_text(encoding="utf-8"))
rel = FUP["reliability"]
out.append("\\section{Follow-up study: reliability and span labels}\n")
out.append("The follow-up protocol (\\texttt{revision/prospective\\_extension\\_plan.md}) was frozen on 24 October 2026; labels were returned on 25 October 2026. "
           f"All three sheets were complete ({rel['units']} units each, no missing or invalid cells). "
           "Table~\\ref{tab:S6} gives primary-label agreement by sample and with one rater left out.\n")
rows = []
for samp, lab in (("all", "All FU1"), ("fu_matched_replication", "Matched pairs"), ("fu_followup_arms", "Follow-up arms")):
    sr = rel["subset_reliability"][samp]
    for k, kl in (("all_raters", "R1, R2, R3"), ("without_R1", "R2, R3"), ("without_R2", "R1, R3"), ("without_R3", "R1, R2")):
        v = sr[k]
        rows.append(f"{lab} & {kl} & {v['n_units']} & {v['alpha']:.2f} & {v['raw_agreement']:.2f} & {v['gwet_ac1']:.2f} & {100 * v['prevalence'].get('yes', 0):.1f} \\\\")
out.append("\\begin{table}[!htb]\\centering\\small\\caption{Follow-up batch FU1: agreement on the primary label before any adjudication.}\\label{tab:S6}\n"
           "\\begin{tabular}{llrrrrr}\\toprule Sample & Raters & Units & $\\alpha$ & Agree. & AC1 & Pos.\\,\\% \\\\ \\midrule\n" +
           "\n".join(rows) + "\n\\bottomrule\\end{tabular}\n\\par\\vspace{2pt}{\\footnotesize Pos.: share of ratings marking an unnecessary mention.}\\end{table}\n")
prof = rel["per_rater_profile"]["least_relevant"]
out.append("Ratings of the least relevant mention by rater (required / useful optional / unnecessary / not applicable): " +
           "; ".join(f"{r} {p.get('required', 0)} / {p.get('useful_optional', 0)} / {p.get('unnecessary', 0)} / {p.get('na', 0)}" for r, p in sorted(prof.items())) + ".\n\n")
s6 = FUP["S6"]
br = s6["by_rater"]
out.append("\\paragraph{Span labels (FU2).} " +
           f"Raters copied offending spans for the {s6['units']} original positives; {s6['spans_total']} spans were marked and all were found verbatim in the response. "
           "Offending-span recall of the lexicon by rater: " +
           "; ".join(f"{r} {v['lexicon_any']['hits']}/{v['lexicon_any']['spans']}" for r, v in sorted(br.items())) +
           ". Spans with no lexicon overlap: " +
           "; ".join(tex_escape(ascii_text(sp))[:90] for sp in dict.fromkeys(
               m["span"].replace("’", "'").replace("‘", "'").strip() for m in s6["missed_spans_examples"])) + ".\n\n")
rg = s6["rgr"]
out.append(f"RGR outputs among the {s6['units']} positives: {rg['units']} (gate outcome: " +
           ", ".join(f"{tex_escape(k)} {v}" for k, v in rg["by_gate"].items()) +
           f"); {rg['with_restore']} carried an executed restore, and in {rg['restore_matches_offending_span']} of them the restored detail lay inside a rater-marked span.\n")
# S7 tables moved from the shortened main text (same generator, same files)
out.append(r"""\clearpage
\section{Tables moved from the main text}
The tables in this section were produced by \texttt{revision/make\_tables.py} from the result files and carry the full intervals, tests, costs and denominators behind the main-text summaries.

\begin{table}[!htb]\centering\small
\caption{Generator models (Amazon Bedrock, \texttt{us-east-1}, 8--10 October 2026).}\label{tab:models}
\begin{tabular}{lll}\toprule Model & Bedrock identifier & Reference \\ \midrule
Nova Lite & \texttt{amazon.nova-lite-v1:0} & \citet{amazon2025nova} \\
Nova Pro & \texttt{amazon.nova-pro-v1:0} & \citet{amazon2025nova} \\
Gemma 3 12B IT & \texttt{google.gemma-3-12b-it} & \citet{gemma2025gemma3} \\
Qwen3 32B & \texttt{qwen.qwen3-32b-v1:0} & \citet{yang2025qwen3} \\
Mistral Large 3 & \texttt{mistral.mistral-large-3-675b-instruct} & \citet{mistral2025large3} \\
DeepSeek V3.2 & \texttt{deepseek.v3.2} & \citet{deepseek2025v32} \\ \bottomrule
\end{tabular}\end{table}

\begin{algorithm}[!htb]
\caption{Relevance-Gated Revision (RGR)}\label{alg:rgr}
\begin{algorithmic}[1]
\Require history $h$, current request $q$, draft $d$ from ordinary generation, model $f$, lexicon $\mathcal{L}$
\State $T_d \gets \mathrm{spans}_{\mathcal{L}}(d)$; \quad $T_q \gets \mathrm{spans}_{\mathcal{L}}(q)$
\If{$T_d = \emptyset$ \textbf{and} $T_q = \emptyset$} \Return $d$ \Comment{no identity term: no extra call}
\EndIf
\State $(g, M) \gets f_{\mathrm{gate}}(h, q, d, T_d, T_q)$ \Comment{$g(t) \in \{\mathrm{keep\_req}, \mathrm{keep\_use}, \mathrm{rm\_unnec}, \mathrm{rm\_unsup}\}$}
\State $R \gets \{t : g(t) \in \{\mathrm{rm\_unnec}, \mathrm{rm\_unsup}\}\}$; \quad $K \gets \{t : g(t) \in \{\mathrm{keep\_req}, \mathrm{keep\_use}\}\}$
\If{$R = \emptyset$ \textbf{and} $M = \emptyset$} \Return $d$
\EndIf
\State $d' \gets f_{\mathrm{revise}}(q, d, R, K, M)$ \Comment{minimal edit; history not passed}
\If{$\mathrm{canon}(K) \cap \mathrm{canon}(d) \not\subseteq \mathrm{canon}(d')$} \Return $d$ \Comment{retention guard}
\EndIf
\State \Return $d'$
\end{algorithmic}
\end{algorithm}

\input{ipm_tables_v2/tab_reliability}
\input{ipm_tables_v2/tab_evaluator}
\input{ipm_tables_v2/tab_lexical}
\input{ipm_tables_v2/tab_rgr_checks}
\input{ipm_tables_v2/tab_flow}
\input{ipm_tables_v2/tab_dialogue}
\input{ipm_tables_v2/tab_rgr}
\input{ipm_tables_v2/tab_remedy_efficacy}
\input{ipm_tables_v2/tab_remedy_preservation}
\input{ipm_tables_v2/tab_followup_arms}
""")
out.append("\\bibliographystyle{plainnat}\n\\bibliography{IPM_references}\n\\end{document}\n")
(SUB / "IPM_Supplement.tex").write_text("".join(out), encoding="utf-8")
print("wrote", SUB / "IPM_Supplement.tex")
