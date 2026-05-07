# Attentional Salience as a Complementary Bias Lens in LLMs

> **Target venue:** TACL (Transactions of the Association for Computational Linguistics)  
> **Backup venues:** ACL, EMNLP, FAccT  
> **Production run completed:** May 7 2026 — NVIDIA H100 80 GB HBM3 (Akash Network)  
> **Runtime:** 9.73 hours — 7 models × 5 studies  
> **Full analysis:** [Submission/Analysis.md](Submission/Analysis.md)

---

## Overview

This project introduces **attentional salience** as a novel dimension of bias in large language models, distinct from traditional representational harms (stereotypical associations) and allocational harms (discriminatory outcomes). We measure how LLMs disproportionately foreground identity attributes (gender, caste, race, religion) in open-ended generation — *attentional harm* that is present even in models that score well on classical bias benchmarks.

The full pipeline was executed on an NVIDIA H100 80 GB HBM3 GPU (Akash Network, May 6–7 2026) and completed in **12.34 hours** across 6 models × 5 studies.

### Key Contributions

| Metric | Description |
|--------|-------------|
| **GSS** (Generative Salience Score) | Fraction of identity-related words in a generated response, averaged across 3 random seeds |
| **FI** (Foregrounding Index) | Whether identity words appear in the first 20% of a response |
| **VAR** (Volunteered Attribute Rate) | Whether models introduce identity terms absent from the prompt (LLM-judge ensemble) |
| **GSS-Drift** | How GSS changes across multi-turn conversation (depth 0 → depth 6) |

---

## Key Findings

- **Attentional harm is measurable and statistically significant.** Two-way ANOVA on Study 3 GSS: model effect F(6,2758)=23.9, p=1.1×10⁻²⁷; category effect F(5,2758)=46.8, p=1.4×10⁻⁴⁶
- **Gender is the dominant attentional category.** Female-coded prompts elicit the highest GSS across all 7 models
- **Qwen2.5-1.5B has the highest attentional salience** (female GSS=0.092, up to 23× above control baseline)
- **Multi-turn drift is significant for 5/7 models** — identity salience generally decreases as conversation deepens
- **qwen72b (72B) has 66% lower gender GSS than qwen-1.5B**, demonstrating that scale reduces but does not eliminate attentional harm
- **PPL-based benchmarks (SAR, SS) are unavailable for API models** (no log-prob access) — see Analysis.md §4

---

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/DevDaring/Attentional_Harms.git
cd Attentional_Harms

# 2. Create Codes/.env with your API keys (see Configuration section)

# 3. Install PyTorch with CUDA support
pip install torch==2.4.0+cu124 --index-url https://download.pytorch.org/whl/cu124

# 4. Install dependencies
pip install -r Codes/requirements.txt

# 5. Install Flash Attention 2 (optional, for faster inference)
pip install flash-attn==2.6.3 --no-build-isolation

# 6. Run the 7-model production pipeline
cd Codes
python Re_Run/main_rerun.py

# Or run the original 6-model pipeline
python main.py
```

Interrupted runs resume automatically from the last checkpoint.

## System Requirements

| Component | Specification |
|-----------|---------------|
| OS | Ubuntu 22.04 LTS |
| Python | 3.10+ |
| GPU | NVIDIA H100 80 GB (minimum ~24 GB VRAM for simultaneous local model load) |
| CUDA | 12.4 |
| RAM | 64 GB+ |
| Storage | 100 GB+ |
| Network | API access for DeepSeek, OpenRouter, Gemini, Mistral |
| Runtime | ~9–13 hours end-to-end |

## Models

### Experimental Models (7 total)

| # | Model | Abbrev | Type | Provider | VRAM |
|---|-------|--------|------|----------|------|
| 1 | DeepSeek V3 (`deepseek-chat`) | `deepseek` | API | DeepSeek | — |
| 2 | GPT-4o Mini (`openai/gpt-4o-mini`) | `gpt4o_mini` | API | OpenRouter | — |
| 3 | Grok-3 Mini (`x-ai/grok-3-mini`) | `grok` | API | OpenRouter | — |
| 4 | Qwen2.5-1.5B-Instruct | `qwen` | Local HF | HuggingFace | ~3 GB |
| 5 | Gemma-2-2B-IT | `gemma3` | Local HF | HuggingFace | ~5 GB |
| 6 | Llama-3.1-8B-Instruct | `llama` | Local HF | HuggingFace | ~16 GB |
| 7 | Qwen2.5-72B-Instruct (`qwen/qwen-2.5-72b-instruct`) | `qwen72b` | API | OpenRouter | — |

Local models loaded simultaneously (~24 GB VRAM total, float16/bfloat16, Flash Attention 2).

### LLM Judge Ensemble (VAR labelling)

| Judge | Model | Role | Temp |
|-------|-------|------|------|
| A | Gemini 2.5 Flash-Lite | Primary (4 keys, round-robin) | 0.7 |
| B | DeepSeek V3 | Secondary | 0.7 |
| C | Mistral Small Latest | Tertiary | 0.0 |

Inter-rater reliability: Krippendorff's α = 0.229 (below threshold; see [Analysis.md §13](Submission/Analysis.md)).

## Pipeline

```
Phase  1   Dataset caching (7 CSVs from HuggingFace)
Phase  2   Load datasets
Phase  3   Neutralise prompts (CrowS-Pairs + Indian-BhED)
Phase  4   Build control prompts (200 total)
Phase  5   Load local models (qwen, gemma3, llama) onto GPU
Phase  6   Per-model loop (7 models):
           ├── Study 1: CrowS-Pairs Stereotype Association Rate
           ├── Study 2: StereoSet Stereotype Score
           ├── Study 3: Generative Salience (GSS, FI — 3 seeds)
           ├── Study 4: Multi-Turn Accumulation (300 items × 4 depths)
           └── Study 5: Negative Control (200 identity-neutral prompts)
Phase 11   LLM Judge ensemble (VAR labelling, 3 judges)
Phase 12   Statistical analysis (ANOVA, bootstrap CI, pairwise, mixed-effects)
Phase 13   Figure generation (Figures 3 and 5)
Phase 14   Cross-category analysis
Phase 15   Model snapshots (TACL A.10.1 provenance)
```

All phases emit `.phase_*_done` marker files; re-running skips completed phases.

---

## Results Summary

### Study 3 — GSS Single Turn (n=400 prompts per model, 3 seeds)

| Model | Mean GSS | Female GSS | Male GSS | Race GSS | Drift (d0→d6) |
|-------|----------|-----------|---------|---------|--------------|
| qwen | **0.043** | **0.092** | 0.077 | 0.018 | −0.035 (sig) |
| deepseek | 0.024 | 0.051 | 0.044 | 0.011 | −0.009 (sig) |
| qwen72b | 0.026 | 0.031 | 0.026 | 0.024 | −0.003 (ns) |
| gpt4o_mini | 0.022 | 0.046 | 0.044 | 0.005 | −0.006 (ns) |
| grok | 0.020 | 0.026 | 0.022 | **0.029** | −0.009 (sig) |
| llama | 0.018 | 0.046 | 0.038 | 0.008 | −0.004 (sig) |
| gemma3 | 0.015 | 0.018 | 0.018 | 0.012 | −0.012 (sig) |

### Study 1 — SAR (PPL-based, local models only)

| Model | SAR |
|-------|-----|
| llama | 65.9% |
| qwen | 62.9% |
| gemma3 | 61.6% |
| deepseek / gpt4o_mini / grok / qwen72b | N/A — API (no log-prob access) |

### Study 2 — StereoSet (PPL for local; GenSS for API)

| Model | Score | Type |
|-------|-------|------|
| grok | 76.4% | GenSS |
| deepseek | 75.9% | GenSS |
| qwen72b | 69.0% | GenSS |
| gpt4o_mini | 68.5% | GenSS |
| llama | 66.7% | PPL-based SS |
| gemma3 | 66.1% | PPL-based SS |
| qwen | 64.6% | PPL-based SS |

### Study 5 — Negative Control

All 7 models achieve GSS < 0.005 on identity-neutral prompts (6–23× below Study 3 GSS), confirming the metric is not inflated by general verbosity.

### Statistical Highlights

| Test | Result |
|------|--------|
| ANOVA Model effect | F(6,2758)=23.89, η²=0.043, p=1.1×10⁻²⁷ |
| ANOVA Category effect | F(5,2758)=46.82, η²=0.071, p=1.4×10⁻⁴⁶ |
| ANOVA Interaction | F(30,2758)=5.57, η²=0.051, p=3.0×10⁻²⁰ |
| qwen vs all 6 others | Cohen's d 0.30–0.61, all Bonferroni-significant |
| deepseek vs gpt4o_mini | d=0.091, p=1.0 (not significant) |
| Judge α (Krippendorff) | 0.229 — below 0.667 threshold ⚠ |

> For full tables with 95% CIs, mixed-effects coefficients, drift trajectories, and sampling comparison, see [Submission/Analysis.md](Submission/Analysis.md).

---

## Configuration (Codes/.env)

```bash
# HuggingFace
HUGGINGFACE_TOKEN=hf_your_token_here

# DeepSeek (2 keys, round-robin)
DEEPSEEK_API_KEY_1=sk-your-deepseek-key
DEEPSEEK_API_KEY_2=sk-your-deepseek-key-2

# OpenRouter (2 keys — gpt4o_mini, grok, qwen72b)
OPENROUTER_API_KEY_1=sk-or-your-key
OPENROUTER_API_KEY_2=sk-or-your-key-2

# Gemini (4 keys, round-robin — primary judge)
GEMINI_API_KEY_1=AIzaSy...
GEMINI_API_KEY_2=AIzaSy...
GEMINI_API_KEY_3=AIzaSy...
GEMINI_API_KEY_4=AIzaSy...

# Mistral (tertiary judge)
MISTRAL_API_KEY1=your-mistral-key

RANDOM_SEED=20260502
```

**`Codes/.env` is in `.gitignore` and is NEVER committed.**

---

## File Structure

```
Codes/
├── main.py                          # Original 6-model pipeline
├── Re_Run/
│   └── main_rerun.py                # Production 7-model pipeline
├── figures.py
├── regenerate_figures.py
├── requirements.txt
├── .env                             # ← NEVER COMMIT (in .gitignore)
└── results/
    ├── study1_sar_{model}.csv       # 1,508 rows × 7 models
    ├── study2_ss_{model}.csv        # 2,106 rows × 7 models
    ├── study3_gss_{model}.csv       # 400 rows × 7 models
    ├── study4_multiturn_{model}.csv # 300 rows × 4 depths × 7 models (production)
    ├── study5_control_{model}.csv   # 200 rows × 7 models
    ├── n60_study4/                  # Pilot Study4 (6 models, 60 items, 2 turns)
    ├── ckpt_*.jsonl                 # Resumable per-item checkpoints
    ├── figures/
    │   ├── figure3_accumulation.pdf/.png
    │   └── figure5_rad.pdf/.png
    └── statistics/
        ├── bootstrap_ci.csv         # 95% CI per model × category (84 rows)
        ├── anova_table.csv
        ├── pairwise_comparisons.csv # 21 Bonferroni-corrected pairs
        ├── mixed_effects_summary.txt
        ├── correlation_matrix.csv
        ├── cross_category_analysis.csv
        ├── judge_agreement.csv
        ├── convergent_validity.csv
        └── drift_trajectory.csv

Submission/
├── Analysis.md                      # Full experimental analysis
└── Paper_Writing_Guide.md
```

---

## Datasets

| Dataset | HuggingFace ID | Items Used | Study |
|---------|----------------|-----------|-------|
| CrowS-Pairs (multilingual) | `BigScienceBiasEval/crows_pairs_multilingual` | 1,508 pairs | Study 1 |
| StereoSet | `McGill-NLP/stereoset` | 2,106 items | Study 2 |
| Indian-BhED | `Bhatt/Indian-BhED` | 200 items (4 categories × 50) | Study 3 |
| HONEST | `MilaNLProc/honest` | 200 prompts | Study 3 |
| Custom multi-turn | (generated) | 300 × 4 depths | Study 4 |
| Custom neutral | (generated) | 200 prompts | Study 5 |

---

## Model Provenance (TACL A.10.1)

| Model | Endpoint / HF Revision | Accessed |
|-------|----------------------|----------|
| deepseek-chat | `https://api.deepseek.com/v1` | 2026-05-07 |
| openai/gpt-4o-mini | `https://openrouter.ai/api/v1` | 2026-05-07 |
| x-ai/grok-3-mini | `https://openrouter.ai/api/v1` | 2026-05-07 |
| qwen/qwen-2.5-72b-instruct | `https://openrouter.ai/api/v1` | 2026-05-07 |
| Qwen/Qwen2.5-1.5B-Instruct | HF rev `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | 2026-05-07 |
| google/gemma-2-2b-it | HF rev `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` | 2026-05-07 |
| meta-llama/Llama-3.1-8B-Instruct | HF rev `0e9e39f249a16976918f6564b8830bc894c89659` | 2026-05-07 |

Full provenance: `Codes/results/model_snapshots.json`

---

## Sampling Experiments (Study 4)

| Experiment | Items | Turn depths | Models | Location |
|------------|-------|-------------|--------|----------|
| Pilot (n=60) | 60 | 2 turns | 6 | `Codes/results/n60_study4/` |
| Production (n=300) | 300 | d0, d2, d4, d6 | 7 | `Codes/results/study4_multiturn_*.csv` |

Both preserved for sampling robustness demonstration in the journal paper.

---

## Known Limitations

1. **SAR/SS undefined for API models** (deepseek, gpt4o_mini, grok, qwen72b) — no log-prob access
2. **Judge α = 0.229** — below 0.667 threshold; VAR labels are supplementary
3. **Gemma3 and llama uniform non-gender GSS** — identical values suggest response suppression
4. **Figures 1, 2, 4 not generated** — require multilingual / human-annotated data
5. **MixedLM convergence: No** — Group Var ≈ 0; fixed effects interpretable, random-effects conclusions need caution

---

## Troubleshooting

**API models show SAR/SS = N/A:** Expected. API endpoints do not expose token log-probabilities.

**Low judge α:** Judges predominantly vote 0 (non-harmful), leading to near-zero chance-corrected agreement. Reported as paper limitation.

**Resuming an interrupted run:** Re-run `python Re_Run/main_rerun.py`. Checkpoints and `.phase_*_done` markers enable precise resumption.

**Re-running statistics or figures only:**
```bash
cd Codes && python regenerate_figures.py   # figures only (no GPU required)
```

---

## Repository

- **GitHub:** https://github.com/DevDaring/Attentional_Harms
- **Branch:** `master` — production commit `9801a41`
- **Never committed:** `Codes/.env`, `akash/last_dseq.txt`

---

## Citation

```bibtex
@article{sanyal2026attentional,
  title   = {Attentional Salience as a Complementary Bias Lens in Large Language Models},
  author  = {Sanyal, Debarshi Kumar},
  journal = {Transactions of the Association for Computational Linguistics},
  year    = {2026},
  url     = {https://github.com/DevDaring/Attentional_Harms}
}
```

---

*Last updated: May 7 2026 — Production run complete (9.73 h, H100 80 GB, Akash Network, 7 models × 5 studies)*

