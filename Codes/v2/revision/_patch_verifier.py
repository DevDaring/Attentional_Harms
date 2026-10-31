"""Apply the independent numerical audit's corrections (24 Oct 2026)."""
from pathlib import Path

SUB = Path(__file__).resolve().parents[3] / "Submission"
REV = Path(__file__).resolve().parent


def patch(path: Path, pairs):
    s = path.read_text(encoding="utf-8")
    for a, b in pairs:
        assert a in s, (path.name, a[:70])
        s = s.replace(a, b)
    path.write_text(s, encoding="utf-8")


patch(SUB / "IPM_Attentiona_Harm.tex", [
    ("Each identity mention is labelled for conversational support and for relevance to the current request.",
     "Each response is labelled for the conversational support of its least supported identity mention and the relevance of its least relevant mention to the current request."),
    ("unnecessary mentions occurred in 13.2\\% of incidental-condition answers and 1.0\\% of absent-condition answers (11 versus 1 of 80 responses;",
     "unnecessary mentions occurred in an estimated 13.2\\% of incidental-condition answers and 1.0\\% of absent-condition answers (11 versus 1 of 80 responses; weighted"),
    ("No remedy showed a statistically reliable reduction, because ordinary answers contained only two unnecessary mentions in 29 pairs.",
     "No remedy showed a statistically reliable reduction; with two unnecessary mentions in 29 ordinary answers, no reduction above 6.9 points was possible in this sample."),
    ("Trace inspection suggests that the proposed relevance-gated revision failed through erroneous restoration of incidental details.",
     "Trace inspection suggests erroneous restoration of incidental details as one failure mechanism of the proposed relevance-gated revision."),
    ("None of them asks whether the request needs the identity in the answer,",
     "These measures do not ask whether the request needs the identity in the answer,"),
    ("Retention of required identity has not been measured for these remedies in document drafting.",
     "No study known to the authors has measured retention of required identity for these remedies in document drafting."),
    ("It labels every identity mention for its support in the conversation and its relevance to the current request, and it scores retention of required identity separately.",
     "It labels each response for the support of its least supported identity mention and the relevance of its least relevant mention, and it scores retention of required identity separately."),
    ("and trace analysis locates a failure mechanism of the proposed revision",
     "and trace analysis suggests one failure mechanism of the proposed revision"),
    ("The limit was raised from 384 tokens after development runs truncated up to 17\\% of answers;",
     "The limit was raised from 384 to 512 and then to 1,024 tokens, after pilot and development runs truncated up to 17\\% of one model's answers;"),
    ("Later changes to the configuration, the remedy code and the judge code are listed with dates in Supplementary Section~S4.",
     "Later changes to the configuration, the remedy code and the judge code are listed in Supplementary Section~S4; the date of one judge-code change could not be recovered."),
    ("Task adequacy had $\\alpha$ of 0.48 in the main batches",
     "Task adequacy had $\\alpha$ of 0.48 in the main and development batches"),
    ("($+10.3$ points; score CI $-1.8$ to $+25.2$; $p_{\\mathrm{Holm}} = 0.12$ for the prespecified contrast)",
     "($+10.3$ points; score CI $-1.8$ to $+25.2$; bootstrap $p = 0.061$ and $p_{\\mathrm{Holm}} = 0.12$ for the prespecified contrast)"),
    ("(97.6\\%)", "(weighted estimate 97.6\\%)"),
    ("Ordinary answers retained the required identity in 12 of the 15 required-condition sets.",
     "Ordinary answers retained the required identity in 12 of the 15 required-condition sets; the other 3 were labelled not applicable."),
    ("Eligibility varied by arm because majority labels of ``not applicable'' drop a pair.",
     "Eligibility varied by arm because a majority label of ``not applicable'', or no majority, drops a pair."),
    ("The original rule therefore understates rather than overstates these losses.",
     "The original rule may therefore understate these losses; counting every such output as a loss gives at most 9 of 12 and 5 of 12."),
    ("Missed spans were mainly religious-practice words, place names and community names absent from the lexicon.",
     "Missed spans included religious-practice words, kinship terms, honorifics, place names and community names absent from the lexicon."),
    ("Most restored details were caste or community names taken from the request, as in Example~3 of Table~\\ref{tab:examples}.",
     "In the 30 logged incidental restores, most restored details were caste or community names taken from the request, as in Example~3 of Table~\\ref{tab:examples}."),
    ("The development sets showed the same pattern in 3 of 3 cases.",
     "Across the 44 development sets, all 3 unnecessary mentions that RGR added also followed an executed restore; one of them was in the required condition."),
    ("In these benchmark outputs, identity in the request propagated into the document.",
     "In these benchmark outputs, identity supplied in the request often reappeared in the document: 8 of the 11 incidental-condition positives repeated an explicit request detail."),
    ("Rates differed across the six models, but each model contributed only 13--14 incidental responses, so no model ranking is drawn.",
     "Observed per-model rates ranged from 0.0 to 25.0\\%, but each model contributed only 13--14 incidental responses, so no model ranking is drawn."),
    ("A judge built for one construct does not transfer to another without validation.",
     "The tested judge ensemble did not transfer to this construct without validation."),
    ("and one judge confused support with relevance (Table~\\ref{tab:evaluator}).",
     "and one judge labelled identity supplied in the request as unsupported (Table~\\ref{tab:evaluator})."),
    ("Term counts and single LLM judges are therefore weak proxies for task relevance of identity.",
     "The tested term counts and single judges were therefore weak proxies for task relevance of identity."),
])

patch(REV / "make_tables.py", [
    ("(every restore request was executed)",
     "(every restore request was executed in the absent and incidental conditions; 98 of 102 in the required condition)"),
    ("Gate output reached its 300-token limit in \"\n            rf\"{R['rgr_audit']['gate_calls']['stop_max_tokens']} of {R['rgr_audit']['gate_calls']['n']} calls.\")",
     "Gate output reached its 300-token limit in \"\n            rf\"{R['rgr_audit']['gate_calls']['stop_max_tokens']} of {R['rgr_audit']['gate_calls']['n']} gate calls across all RGR and RGR-without-restore runs.\")"),
    ("(\"Single-turn B0, enriched diagnostic\", None, \"challenge\", \"Evaluator diagnosis only\"),",
     "(\"Enriched diagnostic (single-turn, dialogue, paraphrase; development and locked)\", None, \"challenge\", \"Evaluator diagnosis only\"),"),
    ("`--': the row is a sample drawn from outputs already counted above.",
     "`--': sampled from several experiments or splits; generated totals per experiment are in the supplementary denominator files."),
    ("only B0 had an unnecessary mention. Score CI:",
     "only B0 had an unnecessary mention. For RGR, the prespecified bootstrap $p$ was 0.061 (Holm-adjusted 0.12). Score CI:"),
])

patch(SUB / "IPM_Cover_Letter.tex", [
    ("Two adapted revision methods lost required identity in 6 of 9 and 3 of 10 eligible pairs while lowering identity-term counts, a loss that term-count evaluation scores as improvement.",
     "Two adapted revision methods lost required identity in 6 of 9 and 3 of 10 eligible pairs; 12 of the 13 retention losses coincided with a fall in identity-term count, which term-count evaluation scores as improvement."),
    ("All limitations are stated,", "The main limitations are stated,"),
    ("Deviations from the internally frozen protocol are dated and reported.",
     "Deviations from the internally frozen protocol are reported, with dates where they could be recovered."),
])
patch(SUB / "IPM_Highlights.tex", [
    ("an LLM judge failed validation.", "an LLM judge ensemble failed validation."),
    ("Adapted revision methods deleted required identity while term counts fell.",
     "Two revision methods deleted required identity while term counts fell."),
])
print("all patches applied")
