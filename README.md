# IRIS: Evaluating Identity Relevance and Information Retention in Generated Documents

Code, prompts, model outputs, de-identified human labels and analysis scripts for the manuscript
*IRIS: Evaluating Identity Relevance and Information Retention in Generated Documents*
(Koushik Deb, IIIT Kalyani; Abhinaba Basu, IIIT Allahabad).

Language-model assistants draft notices, e-mails and summaries that other people read. Such a
document sometimes needs a person's gender, caste, religion or ethnicity, and sometimes exposes it
without purpose. **IRIS** (Identity Relevance and Information-retention Scoring) evaluates both
directions: it labels whether each identity mention is supported by the conversation, whether the
current request needs it, and whether identity that the task requires is kept.

---

## Main results

All results use human labels from three annotators over six hosted models.

| Result | Value |
|---|---|
| Incidental identity detail, matched replication (160 pairs, 80 families) | 10/160 incidental vs 0/160 absent answers with an unnecessary mention; +6.2 points, bootstrap CI [+2.5, +10.0] |
| First representative sample (240 responses) | 13.2% incidental vs 1.0% absent; difference 12.2 points |
| Retention losses under remedies (92 required-condition pairs, nine remedies) | 13 losses; 12 coincided with a fall in identity-term count |
| Implication prompting / critique-and-revise | lost required identity in 6 of 9 and 3 of 10 eligible pairs |
| Automated judge ensemble (360 responses) | precision 0.55, recall 0.43, against targets 0.80 and 0.70 |

**Annotator agreement on the primary label** (`Codes/v2/revision/results/reliability_depth.json`):

| Batch | Responses | Krippendorff's alpha [95% CI] | Gwet's AC1 | Positive agreement | Negative agreement |
|---|---:|---|---:|---:|---:|
| Main + development | 876 | 0.85 [0.78, 0.89] | 0.98 | 0.87 | 0.99 |
| Comparator remedies | 264 | 0.66 [0.46, 0.80] | 0.96 | 0.68 | 0.98 |
| Matched replication | 452 | 0.31 [0.17, 0.45] | 0.93 | 0.35 | 0.96 |

---

## Design

- **Benchmark:** 240 prompt families of everyday drafting tasks, 60 each for gender, caste, religion and race or ethnicity, plus dialogue families. Each family has an *absent*, an *incidental* and a *required* identity variant. Families were split into 80 development and 160 locked families before any output was examined.
- **Models:** Nova Lite, Nova Pro, Gemma 3 12B IT, Qwen3 32B, Mistral Large 3 and DeepSeek V3.2 on Amazon Bedrock (October 2026); temperature 0.7, 1,024-token limit, two draws per condition.
- **Methods:** ordinary generation (B0) against twelve methods: two instruction baselines, six published inference-time remedies, relevance-gated revision (RGR), two controls and a contextual-integrity reasoning prompt.
- **Labels:** support, relevance, retention, adequacy and secondary harms, labelled blind to model, method and condition. 1,652 unique responses were labelled.
- **Statistics:** stratified family-cluster bootstrap (2,000 replicates, seed 20261008), Wilson and Newcombe score intervals, exact McNemar tests, Holm correction for the primary contrasts.

---

## Repository structure

```
Codes/v2/
├── run_all.py                 # single entry point (generate, sample, judge, analyze, tables, ...)
├── config_v2.py               # models, settings and paths
├── bank.py, lexicon_v2.py     # prompt bank and identity lexicon
├── remedies.py, judges.py     # remedies (incl. RGR) and automated judges
├── data/
│   ├── prompt_bank.csv        # 240 prompt families (720 entries)
│   └── dialogue_bank.csv      # dialogue families
├── outputs/
│   ├── generations.jsonl.gz   # every model request and response (gzip)
│   └── judgments.jsonl.gz     # every automated-judge call (gzip)
├── annotation/
│   ├── reports/               # de-identified raw labels, majority and adjudicated labels, reliability, per batch
│   └── model_panel/           # preliminary model-generated labels from development; not used in any reported result
├── analysis/                  # agreement, bootstrap, effects, robustness, tables
└── revision/
    ├── make_tables.py         # regenerates every manuscript table from the result files
    ├── reliability_depth.py   # agreement by batch: alpha intervals, AC1, specific agreement
    ├── fu_analysis.py         # matched replication, follow-up arms, span analysis
    ├── reanalysis.py          # re-analyses on the existing human labels
    ├── results/               # JSON and CSV results behind the manuscript
    ├── annotation_followup/   # follow-up annotation keys and de-identified labels
    └── reviewer_artifact/     # self-contained archive with checksums
```

The earlier salience pipeline (`Codes/main.py`, `Codes/results/`) is kept for reference only.

---

## Reproducing the results

```bash
git clone https://github.com/DevDaring/Attentional_Harms.git
cd Attentional_Harms
pip install -r Codes/v2/requirements_v2.txt

# model outputs
gunzip -k Codes/v2/outputs/generations.jsonl.gz Codes/v2/outputs/judgments.jsonl.gz

# tables and agreement statistics from the stored labels (no API calls needed)
cd Codes/v2
python revision/make_tables.py
python revision/reliability_depth.py
```

New generations need Amazon Bedrock access. Credentials go in `Codes/.env`, which is never
committed:

```bash
AWS_ACCESS_KEY=your_key
AWS_SECRET_KEY=your_secret
```

Then run `python run_all.py --help` in `Codes/v2` for the pipeline stages.

---

## Ethics

Prompts were written for this study and contain no personal data; examples use synthetic people.
Annotators were volunteers from the authors' institution, were informed about sensitive content
and could skip any item. Annotator identities are replaced by R1-R3. Model outputs may contain
stereotypes and are shared for evaluation only.

---

## Citation

```bibtex
@misc{deb2026iris,
  title  = {{IRIS}: Evaluating Identity Relevance and Information Retention in Generated Documents},
  author = {Deb, Koushik and Basu, Abhinaba},
  year   = {2026},
  url    = {https://github.com/DevDaring/Attentional_Harms}
}
```
