from pathlib import Path

p = Path(__file__).resolve().parents[3] / "README.md"
s = p.read_text(encoding="utf-8")
R = [
    ("- **Incidental identity in the request raised unnecessary mentions** from 1.0% to 13.2% (difference 12.2 percentage points, 95% CI 5.2-19.8), robust across rater and leave-one-out checks (+9.6 to +16.3 pp).",
     "- **Incidental identity in the request was associated with more unnecessary mentions**: 11 of 80 sampled answers versus 1 of 80 (13.2% vs 1.0%, difference 12.2 points, 95% CI 5.2-19.8); the difference stayed between +9.6 and +16.3 points under rater and leave-one-out checks."),
    ("- **None of nine inference-time remedies reduced unnecessary mentions** on human-labelled matched sets.",
     "- **No inference-time remedy showed a statistically reliable reduction** on 44 human-labelled matched sets (only 2 baseline positives in 29 pairs)."),
    ("- **Traces locate the RGR failure** in its relevance gate (17.8% misjudgement rate), not in term detection (98.4% coverage).",
     "- **Trace inspection suggests restoration errors in RGR**: the gate requested restoring an incidental detail in 17.8% of incidental answers, and every unnecessary mention RGR added in the labelled sets followed such a restore."),
    ("- **Six models** on Amazon Bedrock: Claude 3 Haiku, Claude 3.5 Sonnet, Llama 3.1 8B, Mistral Large 2, Nova Pro, Nova Lite.",
     "- **Six models** on Amazon Bedrock (us-east-1, 8-10 Oct 2026): Nova Lite, Nova Pro, Gemma 3 12B IT, Qwen3 32B, Mistral Large 3, DeepSeek V3.2."),
    ("- **Three human annotators**, 980 responses each, with adjudication. Krippendorff's alpha = 0.85 (pilot), 0.69-0.93 (main batches).",
     "- **Three human annotators**, 1,200 responses each (3,600 ratings), with adjudication. Krippendorff's alpha on the primary label = 0.85 (pilot), 0.85 pooled over main and development batches, 0.66 (comparators)."),
    ("""  title   = {When Identity Belongs in the Answer: Measuring Task-Relevant
             Identity Use and Evaluating Inference-Time Remedies for
             LLM-Generated Text},
  author  = {Deb, Koushik and Sanyal, Debarshi Kumar and Mukherjee, Imon},""",
     """  title   = {When Identity Belongs in Generated Documents: Evaluating
             Relevance and Information Retention},
  author  = {Deb, Koushik and Basu, Abhinaba},"""),
]
for a, b in R:
    assert a in s, a[:60]
    s = s.replace(a, b)
s = s.replace("# Attentional Harms: Measuring Task-Relevant Identity Use in LLM-Generated Text",
              "# When Identity Belongs in Generated Documents: Evaluating Relevance and Information Retention")
s += ("\n\n## Revision of 24 October 2026\n\nThe co-author review is executed in `Codes/v2/revision/` "
      "(audit, evidence ledger, re-analyses, follow-up annotation packs, extension plan). Start with "
      "`Codes/v2/revision/readiness_report.md`.\n")
p.write_text(s, encoding="utf-8")
print("ok")
