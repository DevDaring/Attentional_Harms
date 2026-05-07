# Attentional Salience as a Complementary Bias Lens in LLMs

## Overview

This research project investigates **attentional salience** as a novel dimension of bias in large language models (LLMs), distinct from traditional representational harms (stereotypical associations) and allocational harms (discriminatory outcomes). We measure how LLMs disproportionately foreground identity attributes in open-ended generation — "attentionally harmful" behaviour present even in models that perform well on standard bias benchmarks.

The full pipeline was executed on an NVIDIA H100 80 GB HBM3 GPU (Akash Network, May 6–7 2026) and completed in **12.34 hours** across 6 models × 5 studies.

### Key Contributions
- **Generative Salience Score (GSS):** Fraction of identity-related words in a generated response (3-seed average)
- **Foregrounding Index (FI):** Whether identity appears in the first 20% of a response
- **Volunteered Attribute Rate (VAR):** Whether models introduce identity terms absent from the prompt (LLM-judge ensemble)
- **Accumulation Rate (AR / GSS drift):** How salience increases across multi-turn conversation
- **Cross-Category Analysis:** Salience variation across 6 identity categories (caste, female, gender, male, race, religion)

### Target Venue
- **Primary:** TACL (Transactions of the Association for Computational Linguistics)
- **Backup:** ACL, EMNLP, FAccT

## Quick Start

```bash
# 1. Clone the repository
git clone https://github.com/DevDaring/Attentional_Harms.git
cd Attentional_Harms

# 2. Create .env file with your API keys (see Configuration section)
# Edit Codes/.env with your keys

# 3. Install PyTorch with CUDA support
pip install torch==2.4.0+cu124 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

# 4. Install dependencies
pip install -r Codes/requirements.txt

# 5. Install Flash Attention 2 (optional, for faster inference)
pip install flash-attn==2.6.3 --no-build-isolation

# 6. Run the full pipeline
cd Codes
python main.py
```

Interrupted runs resume automatically from the last checkpoint.

## System Requirements

| Component | Specification |
|---|---|
| OS | Ubuntu 22.04 LTS |
| Python | 3.10+ |
| GPU | NVIDIA H100 80 GB (minimum 24 GB VRAM for local models) |
| CUDA | 12.4 |
| RAM | 64 GB+ |
| Storage | 100 GB+ |
| Runtime | ~12–13 hours end-to-end |

## Models

### Experimental Models (6 total)

| Model | Type | Provider | VRAM |
|---|---|---|---|
| DeepSeek V3 (`deepseek-chat`) | API | DeepSeek | — |
| GPT-4o Mini (`openai/gpt-4o-mini`) | API | OpenRouter | — |
| Grok-3 Mini (`x-ai/grok-3-mini`) | API | OpenRouter | — |
| Qwen2.5-1.5B-Instruct | Local HF | HuggingFace | ~3 GB |
| Gemma-2-2B-IT | Local HF | HuggingFace | ~5 GB |
| Llama-3.1-8B-Instruct | Local HF | HuggingFace | ~16 GB |

All three local models are loaded simultaneously (~24 GB total VRAM). Local models use float16/bfloat16 with Flash Attention 2.

### LLM Judges (for VAR / hurtful-completion labelling)
- **Primary:** Gemini 2.5 Flash (4 keys, temp=0.7)
- **Secondary:** DeepSeek V3 (temp=0.7)
- **Tertiary:** Mistral Small Latest (temp=0.0)

3-judge ensemble with Krippendorff's α inter-rater reliability.

## Pipeline

```
Phase  1  Dataset caching (7 CSVs, HuggingFace)
Phase  2  Load datasets
Phase  3  Neutralise prompts (CrowS-Pairs + Indian-BhED)
Phase  4  Build control prompts (200 total)
Phase  5  Load local models (qwen, gemma3, llama)
Phase  6  Per-model loop (6 models):
          ├── Study 1: CrowS-Pairs Stereotypical Association Rate
          ├── Study 2: StereoSet Stereotype Score
          ├── Study 3: Generative Salience (GSS, FI — 3 seeds)
          ├── Study 4: Multi-Turn Accumulation (60 items × 2 turns)
          └── Study 5: Negative Control (200 prompts)
Phase 11  LLM Judge ensemble (2,400 responses × 3 judges)
Phase 12  Statistical analysis
Phase 13  Figure generation (Figures 2, 3, 5)
Phase 14  Cross-category analysis
Phase 15  Model snapshots (TACL A.10.1 provenance)
```

All phases are checkpointed. Re-running `python main.py` resumes from the last completed item.

---

## Results (Full Run — May 6–7 2026)

### Study 3: Generative Salience Score (GSS)

Mean GSS across 400 prompts (200 HONEST + 200 Indian-BhED), 3 seeds per prompt.

| Model | GSS (mean) | FI (mean) |
|---|---|---|
| **Qwen2.5-1.5B** | **0.0514** | 0.5508 |
| DeepSeek V3 | 0.0310 | 0.4875 |
| GPT-4o Mini | 0.0264 | 0.3267 |
| Llama-3.1-8B | 0.0250 | 0.3325 |
| Grok-3 Mini | 0.0234 | 0.5817 |
| Gemma-2-2B | 0.0152 | 0.3075 |

> Qwen (smallest model) shows the highest salience. Grok has the highest foregrounding index despite moderate GSS, suggesting it places identity terms early in responses even when overall density is lower.

### Study 1: Stereotypical Association Rate (SAR — CrowS-Pairs)

SAR requires token-level log-probabilities, unavailable for API models (marked N/A).

| Model | SAR |
|---|---|
| Llama-3.1-8B | 0.659 |
| Qwen2.5-1.5B | 0.629 |
| Gemma-2-2B | 0.616 |
| DeepSeek V3 | N/A (API) |
| GPT-4o Mini | N/A (API) |
| Grok-3 Mini | N/A (API) |

> All local models show SAR > 0.5, indicating above-chance preference for stereotypical continuations. Llama exhibits the highest traditional bias.

### Study 2: StereoSet Stereotype Score (SS)

| Model | SS |
|---|---|
| Llama-3.1-8B | 0.667 |
| Gemma-2-2B | 0.661 |
| Qwen2.5-1.5B | 0.646 |
| DeepSeek V3 | N/A (API) |
| GPT-4o Mini | N/A (API) |
| Grok-3 Mini | N/A (API) |

### Study 4: GSS Drift (Multi-Turn Accumulation)

Drift = `gss_last − gss_turn1`. Positive values indicate salience increasing across conversation turns.

| Model | GSS Drift | Turn 1 GSS | Final GSS |
|---|---|---|---|
| **Qwen2.5-1.5B** | **+0.0515** | 0.0000 | 0.0515 |
| DeepSeek V3 | +0.0299 | 0.0000 | 0.0299 |
| Llama-3.1-8B | +0.0196 | 0.0000 | 0.0196 |
| GPT-4o Mini | +0.0124 | 0.0065 | 0.0189 |
| Gemma-2-2B | +0.0121 | 0.0000 | 0.0121 |
| Grok-3 Mini | +0.0094 | 0.0005 | 0.0100 |

> All 6 models show positive drift — salience consistently accumulates across turns, confirming the attentional harm accumulation hypothesis.

### Study 5: Negative Control GSS

Control prompts contain no identity-related content. Expected GSS ≈ 0.

| Model | Control GSS |
|---|---|
| DeepSeek V3 | 0.0042 |
| Gemma-2-2B | 0.0028 |
| GPT-4o Mini | 0.0026 |
| Qwen2.5-1.5B | 0.0019 |
| Llama-3.1-8B | 0.0020 |
| Grok-3 Mini | 0.0018 |

> Control GSS is 5–30× lower than Study 3 GSS for all models, validating that the metric captures identity-related salience specifically and is not a general generation artefact.

### Statistical Analysis

**Two-Way ANOVA** (Study 3 GSS ~ model × category):

| Factor | η² | F | p |
|---|---|---|---|
| Model | 0.047 | 27.12 | 9.3 × 10⁻²⁷ |
| Category | 0.085 | 49.23 | 1.3 × 10⁻⁴⁸ |
| Model × Category | 0.047 | 5.36 | 2.4 × 10⁻¹⁶ |

Category accounts for more GSS variance than model identity, suggesting that the nature of identity being probed (gender vs. race vs. caste) shapes output salience more than which model is used.

**Pairwise Model Comparisons** (Bonferroni-corrected, α = 0.05):

Significant pairs: deepseek–gemma3, deepseek–qwen, gemma3–gpt4o_mini, gemma3–grok, gemma3–llama, gemma3–qwen, gpt4o_mini–qwen, grok–qwen, llama–qwen  
Non-significant: deepseek–gpt4o_mini, deepseek–grok, deepseek–llama, gpt4o_mini–grok, gpt4o_mini–llama, grok–llama

**Mixed-Effects Regression** (GSS ~ model + category + model:category, random intercept per category):  
AIC = −7821.99, Log-likelihood = 3949.00

**LLM Judge Inter-Rater Agreement (Krippendorff's α):**

| Dimension | α | Status |
|---|---|---|
| Caste | 0.617 | ✅ Acceptable |
| Religion | 0.607 | ✅ Acceptable |
| Race | 0.616 | ✅ Acceptable |
| Gender | 0.551 | ⚠️ Below 0.6 |
| Female | 0.229 | ❌ Low |
| Male | 0.180 | ❌ Low |

Low agreement on gendered dimensions is noted as a limitation; gender-attribute ambiguity makes judge consensus harder.

### Figures Generated

| Figure | File | Description |
|---|---|---|
| Figure 2 | `results/figures/figure2_var_bars.pdf/.png` | Per-model VAR/GSS bar chart |
| Figure 3 | `results/figures/figure3_accumulation.pdf/.png` | Multi-turn GSS accumulation curves |
| Figure 5 | `results/figures/figure5_rad.pdf/.png` | Radar chart: SAR vs GSS vs drift per model |

Figures 1 and 4 require cross-lingual data and are not produced in the English-only run.

---

## Configuration

Create `Codes/.env`:

```bash
# HuggingFace
HF_KEY=hf_your_token_here

# DeepSeek
DEEPSEEK_API_KEY_1=sk-your-deepseek-key

# OpenRouter (for GPT-4o Mini and Grok)
OPENROUTER_API_KEY_1=sk-or-your-openrouter-key

# Gemini (LLM judge — 4 keys for rate-limit rotation)
GEMINI_API_KEY_1=AIzaSy...
GEMINI_API_KEY_2=AIzaSy...
GEMINI_API_KEY_3=AIzaSy...
GEMINI_API_KEY_4=AIzaSy...

# Mistral (tertiary judge)
MISTRAL_API_KEY=your-mistral-key

RANDOM_SEED=20260502
```

**Never commit `.env` to version control.** It is listed in `.gitignore`.

## Output Structure

```
Codes/
├── data/                          # 7 cached dataset CSVs
├── results/
│   ├── study1_sar_<model>.csv     # Study 1: CrowS-Pairs (1508 rows each)
│   ├── study2_ss_<model>.csv      # Study 2: StereoSet (2106 rows each)
│   ├── study3_gss_<model>.csv     # Study 3: Generative Salience (400 rows each)
│   ├── study4_multiturn_<model>.csv  # Study 4: Multi-Turn (60 rows each)
│   ├── study5_control_<model>.csv    # Study 5: Control (200 rows each)
│   ├── ckpt_*.jsonl               # Resumable checkpoints
│   ├── statistics/
│   │   ├── bootstrap_ci.csv       # 95% CI per model × category (72 rows)
│   │   ├── anova_table.csv        # Two-way ANOVA with η²
│   │   ├── pairwise_comparisons.csv  # 15 Bonferroni-corrected model pairs
│   │   ├── mixed_effects_summary.txt # MixedLM fit summary
│   │   ├── correlation_matrix.csv    # GSS seed cross-correlations
│   │   ├── cross_category_analysis.csv
│   │   └── judge_agreement.csv    # 2400-row judge labels + α
│   ├── figures/
│   │   ├── figure2_var_bars.pdf/.png
│   │   ├── figure3_accumulation.pdf/.png
│   │   └── figure5_rad.pdf/.png
│   ├── main_fullrun.log           # Full execution log
│   └── model_snapshots.json       # TACL A.10.1 provenance
└── model_snapshots.json
```

## Model Provenance (TACL A.10.1)

| Model | HF Revision / Endpoint | Accessed |
|---|---|---|
| deepseek-chat | `https://api.deepseek.com/v1/chat/completions` | 2026-05-07 |
| openai/gpt-4o-mini | `https://openrouter.ai/api/v1/chat/completions` | 2026-05-07 |
| x-ai/grok-3-mini | `https://openrouter.ai/api/v1/chat/completions` | 2026-05-07 |
| Qwen/Qwen2.5-1.5B-Instruct | `989aa7980e4cf806f80c7fef2b1adb7bc71aa306` | 2026-05-07 |
| google/gemma-2-2b-it | `299a8560bedf22ed1c72a8a11e7dce4a7f9f51f8` | 2026-05-07 |
| meta-llama/Llama-3.1-8B-Instruct | `0e9e39f249a16976918f6564b8830bc894c89659` | 2026-05-07 |

## Datasets

| Dataset | HF ID | Rows used | Purpose |
|---|---|---|---|
| CrowS-Pairs | `BigScienceBiasEval/crows_pairs_multilingual` | 1508 | Study 1 (SAR) |
| StereoSet | `McGill-NLP/stereoset` | 2106 | Study 2 (SS) |
| HONEST | `MilaNLProc/honest` | 200 | Study 3 generative prompts |
| Indian-BhED (caste) | `Bhatt/Indian-BhED` | 50 | Study 3 generative prompts |
| Indian-BhED (gender) | `Bhatt/Indian-BhED` | 50 | Study 3 generative prompts |
| Indian-BhED (religion) | `Bhatt/Indian-BhED` | 50 | Study 3 generative prompts |
| Indian-BhED (race) | `Bhatt/Indian-BhED` | 50 | Study 3 generative prompts |

## Troubleshooting

**API models show SAR/SS = −1:** Expected. CrowS-Pairs and StereoSet require token log-probabilities which are not exposed by API endpoints. Only local models (qwen, gemma3, llama) produce valid SAR/SS values.

**Mixed-effects regression convergence warnings:** Expected with 6 random-intercept groups. Model fit is reported but treated as exploratory (ANOVA is the primary inferential test).

**Low judge α for gender/female/male:** Gendered-attribute ambiguity makes these dimensions harder to classify consistently. Reported in paper limitations.

**Resuming an interrupted run:** Simply re-run `python main.py`. Checkpoints (`ckpt_*.jsonl`) are written after every item; completed items are skipped.

## Re-running individual phases

```bash
# Re-run statistics only (no GPU required)
cd Codes && python rerun_statistics.py

# Re-generate figures only (no GPU required)
cd Codes && python regenerate_figures.py
```

## Repository Information

- **GitHub:** https://github.com/DevDaring/Attentional_Harms.git
- **Branch:** master
- **Latest commit:** `9fb2485`
- **Excluded from repo:** `Codes/.env` (API keys), `Submission/`

## Citation

```bibtex
@article{attentional_harms_2026,
  title   = {Attentional Salience as a Complementary Bias Lens in Language Models},
  author  = {[Author]},
  journal = {Transactions of the Association for Computational Linguistics},
  year    = {2026},
  url     = {https://github.com/DevDaring/Attentional_Harms}
}
```

---

**Last Updated:** May 7, 2026 — Full pipeline execution complete (12.34 hours, H100 80 GB, Akash Network)
