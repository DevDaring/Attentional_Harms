# Attentional Harms: Measuring Task-Relevant Identity Use in LLM-Generated Text

This project measures **attentional harms** — unnecessary identity mentions in LLM-generated responses — and evaluates inference-time remedies. Unlike lexical counting approaches, the framework judges each identity mention against the current request: a mention of caste in a scholarship notice is required, but the same detail in a rent receipt is unnecessary.

---

## Overview

The study defines task-relevant identity use for a response in its conversation history and measures it together with retention of required identity and task adequacy. It proposes **Relevance-Gated Revision (RGR)**, a remedy that edits only the identity terms the current request does not need.

### Key Findings

- **Incidental identity in the request raised unnecessary mentions** from 1.0% to 13.2% (difference 12.2 percentage points, 95% CI 5.2-19.8), robust across rater and leave-one-out checks (+9.6 to +16.3 pp).
- **None of nine inference-time remedies reduced unnecessary mentions** on human-labelled matched sets.
- **Two published rewriting methods lost required identity** in 3/10 and 6/9 eligible pairs — a failure that term-count evaluation would score as success.
- **Traces locate the RGR failure** in its relevance gate (17.8% misjudgement rate), not in term detection (98.4% coverage).
- **An LLM judge ensemble did not meet pre-registered agreement targets** (P=0.55, R=0.43 vs 0.80/0.70), so all reported results use human labels.

### Key Metrics

| Metric | Description |
|--------|-------------|
| **Unnecessary mention** | Binary: least relevant identity mention rated "unnecessary" by human annotators |
| **Retention** | Whether identity information required by the task survives the remedy |
| **Task adequacy** | 0-2 rating of response quality for the request |

---

## Design

- **240 prompt families** across four identity categories (gender, caste, religion, race/ethnicity) and **80 dialogue families** for multi-turn evaluation.
- **Three task conditions** per family: absent (no identity), incidental (identity present but irrelevant), required (identity essential to the task).
- **Six models** on Amazon Bedrock: Claude 3 Haiku, Claude 3.5 Sonnet, Llama 3.1 8B, Mistral Large 2, Nova Pro, Nova Lite.
- **Nine remedies** tested: two instruction baselines (B1, B2), RGR, and six published methods (SI_NLI, IF_MSC, DAIQ_GUARD, REPROMPT, CAI_REVISE, SYS2_IMPL).
- **Three human annotators**, 980 responses each, with adjudication. Krippendorff's alpha = 0.85 (pilot), 0.69-0.93 (main batches).

---

## Repository Structure

```
Codes/
├── v2/                              # Main experimental pipeline
│   ├── run_all.py                   # Single entry point
│   ├── config_v2.py                 # Model and experiment configuration
│   ├── generation.py                # Response generation (Bedrock)
│   ├── remedies.py                  # B1, B2, RGR, RGR_RO, comparators
│   ├── judges.py                    # LLM judge ensemble
│   ├── protocol.md                  # Pre-registered analysis plan (frozen)
│   ├── deviations_post_freeze.md    # All deviations logged
│   ├── data/
│   │   ├── prompt_bank.csv          # 240 prompt families
│   │   └── dialogue_bank.csv        # 80 dialogue families
│   ├── annotation/                  # Annotation pipeline and reports
│   │   ├── keys/                    # Annotation keys
│   │   ├── returned/                # Human-annotated workbooks
│   │   └── reports/                 # Reliability and agreement reports
│   ├── analysis/
│   │   ├── effects.py               # Primary analysis (matched pairs, bootstrap)
│   │   ├── robustness.py            # Sensitivity and ablation analyses
│   │   └── latex_tables.py          # Table generation
│   └── outputs/                     # Generated responses and analysis results
├── main.py                          # Earlier 7-model salience pipeline
├── figures.py                       # Figure generation
├── requirements.txt
└── results/                         # Earlier pipeline results
    ├── study*_*.csv
    ├── figures/
    └── statistics/

Annotator_Packs/                     # Annotation workbook builds and returned packs
```

---

## Quick Start

```bash
# Clone
git clone https://github.com/DevDaring/Attentional_Harms.git
cd Attentional_Harms

# Install dependencies
pip install -r Codes/requirements.txt

# Configure API keys
cp Codes/.env.example Codes/.env   # Edit with your Bedrock credentials

# Run the full pipeline
cd Codes/v2
python run_all.py generate
python run_all.py analyze --labels human
python run_all.py tables
```

---

## Earlier Salience Pipeline

The `Codes/` root also contains the earlier attentional salience pipeline (`main.py`, `Re_Run/main_rerun.py`) which measures GSS, FI, VAR and multi-turn drift across 7 models. Results are in `Codes/results/`.

---

## Configuration

API keys go in `Codes/.env` (never committed):

```bash
AWS_ACCESS_KEY=your_key
AWS_SECRET_KEY=your_secret
```

---

## Citation

```bibtex
@misc{deb2026attentional,
  title   = {When Identity Belongs in the Answer: Measuring Task-Relevant
             Identity Use and Evaluating Inference-Time Remedies for
             LLM-Generated Text},
  author  = {Deb, Koushik and Sanyal, Debarshi Kumar and Mukherjee, Imon},
  year    = {2026},
  url     = {https://github.com/DevDaring/Attentional_Harms}
}
```

---

*Repository: [github.com/DevDaring/Attentional_Harms](https://github.com/DevDaring/Attentional_Harms)*
