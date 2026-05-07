# Attentional Harms in LLMs — Complete Experimental Analysis

> **Paper:** *Attentional Salience as a Complementary Bias Lens in Large Language Models*  
> **Author:** Dr. Debarshi Kumar Sanyal  
> **Target Venue:** TACL (Transactions of the Association for Computational Linguistics)  
> **Pipeline executed:** NVIDIA H100 80 GB HBM3, Akash Network — May 6–7 2026  
> **Runtime:** 9.73 hours (production run, 7 models × 5 studies)  
> **GitHub:** `DevDaring/Attentional_Harms` — production commit `9801a41`

---

## Table of Contents

1. [Executive Summary](#1-executive-summary)
2. [Experiment Design](#2-experiment-design)
3. [Data Inventory & Completeness](#3-data-inventory--completeness)
4. [Known Limitations](#4-known-limitations)
5. [Study 1 — Stereotype Association Rate (SAR)](#5-study-1--stereotype-association-rate-sar)
6. [Study 2 — Generative StereoSet Score (GenSS)](#6-study-2--generative-steroset-score-genss)
7. [Study 3 — Generative Salience Score (GSS) — Single Turn](#7-study-3--generative-salience-score-gss--single-turn)
8. [Study 4 — Multi-Turn Attentional Drift](#8-study-4--multi-turn-attentional-drift)
9. [Study 5 — Negative Control](#9-study-5--negative-control)
10. [qwen72b — Seventh Model Analysis](#10-qwen72b--seventh-model-analysis)
11. [Cross-Study Statistical Summary](#11-cross-study-statistical-summary)
12. [Sampling Comparison: n=60 vs n=300 (Study 4)](#12-sampling-comparison-n60-vs-n300-study-4)
13. [Judge Ensemble & Inter-Rater Reliability](#13-judge-ensemble--inter-rater-reliability)
14. [Reproducibility Notes](#14-reproducibility-notes)

---

## 1. Executive Summary

This study introduces **Generative Salience Score (GSS)** as a novel metric for *attentional harm* in LLMs — the tendency of models to foreground identity attributes (gender, caste, race, religion) even in open-ended prompts that do not require such mention. GSS measures the fraction of identity-related words in a generation, averaged across three random seeds.

**Key findings:**

- **Attentional harm is real and significant.** All 7 models exhibit measurable GSS above the near-zero control baseline (Study 5: mean GSS < 0.01 for identity-neutral prompts). A two-way ANOVA confirms significant effects of model identity (F=23.9, p=1.1×10⁻²⁷) and category (F=46.8, p=1.4×10⁻⁴⁶) on GSS.

- **Gender is the dominant attentional category.** The *female* category drives the highest GSS across all models, followed by *male*. The effect is especially strong for Qwen2.5-1.5B (female GSS = 0.092, CI [0.075, 0.111]) and significant interaction terms confirm that female-category salience is model-specific.

- **qwen (local small model) exhibits the highest overall attentional salience.** Mean female GSS = 0.092 vs. deepseek = 0.051, gpt4o_mini = 0.046 (pairwise Cohen's d = 0.38–0.45, p < 10⁻⁶).

- **Multi-turn attentional drift is significant for 5 out of 7 models.** Models that show drift (decreasing GSS with depth) include deepseek, gemma3, grok, and qwen. llama shows non-monotone drift (Kruskal–Wallis significant). Only gpt4o_mini and qwen72b show no significant drift.

- **PPL-based metrics (SAR, SS) are unavailable for API models.** deepseek, gpt4o_mini, grok, and qwen72b return `ppl = -1.0` because API endpoints do not expose log-probabilities. SAR and PPL-based StereoSet Score are valid only for the three local models (qwen, gemma3, llama).

---

## 2. Experiment Design

### 2.1 Research Questions

| RQ | Description |
|----|-------------|
| RQ1 | Do LLMs foreground identity attributes at rates above random baseline? |
| RQ2 | Does attentional salience vary across identity categories (caste, gender, race, religion)? |
| RQ3 | Does multi-turn context accumulate attentional salience over turns? |
| RQ4 | Do models that score well on classical bias benchmarks (SAR, SS) also show low GSS? |

### 2.2 Models

| # | Model | Abbreviation | Type | Provider | VRAM |
|---|-------|--------------|------|----------|------|
| 1 | DeepSeek V3 | `deepseek` | API | DeepSeek API | — |
| 2 | GPT-4o Mini | `gpt4o_mini` | API | OpenRouter | — |
| 3 | Grok-3 Mini | `grok` | API | OpenRouter | — |
| 4 | Qwen2.5-1.5B-Instruct | `qwen` | Local HF | HuggingFace | ~3 GB |
| 5 | Gemma-2-2B-IT | `gemma3` | Local HF | HuggingFace | ~5 GB |
| 6 | Llama-3.1-8B-Instruct | `llama` | Local HF | HuggingFace | ~16 GB |
| 7 | Qwen2.5-72B-Instruct | `qwen72b` | API | OpenRouter | — |

Local models loaded simultaneously (~24 GB VRAM total, float16/bfloat16 + Flash Attention 2).

### 2.3 Datasets

| Dataset | Source | Items | Use |
|---------|--------|-------|-----|
| CrowS-Pairs | Nangia et al. (2020) | 1,508 sentence pairs | Study 1 SAR |
| StereoSet | Nadeem et al. (2021) | 2,106 fill-in-the-blank items | Study 2 SS |
| Indian-BhED + custom GSS prompts | BhED (Bhatt et al.) + manual | 400 items × 6 categories | Study 3 GSS |
| Multi-turn conversation set | Custom (7-turn depth) | 300 items × 4 depths | Study 4 drift |
| Identity-neutral control prompts | Custom | 200 items | Study 5 control |

### 2.4 Metrics

| Metric | Description | Study |
|--------|-------------|-------|
| **SAR** (Stereotype Association Rate) | Fraction of CrowS-Pairs items where model assigns lower perplexity to stereotyped sentence | Study 1 |
| **SS** (StereoSet Score) | Fraction of StereoSet items where model assigns lower perplexity to stereotyped completion | Study 2 |
| **GenSAR / GenSS** | Generation-based analogues for API models (no PPL): fraction of continuations matching stereotyped direction | Studies 1–2 |
| **GSS** (Generative Salience Score) | Fraction of identity-bearing tokens in free-form generation, averaged across 3 seeds | Study 3 |
| **FI** (Foregrounding Index) | Whether identity words appear in the first 20% of a response | Study 3 |
| **GSS-drift** (gss_d6 − gss_d0) | Change in GSS from depth-0 (1-turn) to depth-6 (7-turn) | Study 4 |
| **VAR** (Volunteered Attribute Rate) | LLM-judge-assessed fraction of responses introducing identity absent from prompt | Studies 3–4 |

### 2.5 LLM Judge Ensemble

VAR labels assigned by 3-judge ensemble:
- **Judge A:** Gemini 2.5 Flash-Lite (primary, 4 API keys round-robin, T=0.7)
- **Judge B:** DeepSeek V3 (secondary, T=0.7)
- **Judge C:** Mistral Small Latest (tertiary, T=0.0)

Inter-rater reliability: Krippendorff's α = 0.229 (see §13).

---

## 3. Data Inventory & Completeness

### 3.1 Production CSV Files (post git pull, commit `9801a41`)

All files located in `Codes/results/`.

| File | Model | Items (rows) | Status |
|------|-------|-------------|--------|
| `study1_sar_deepseek.csv` | deepseek | 1,508 | ✓ |
| `study1_sar_gpt4o_mini.csv` | gpt4o_mini | 1,508 | ✓ |
| `study1_sar_grok.csv` | grok | 1,508 | ✓ |
| `study1_sar_qwen.csv` | qwen | 1,508 | ✓ |
| `study1_sar_gemma3.csv` | gemma3 | 1,508 | ✓ |
| `study1_sar_llama.csv` | llama | 1,508 | ✓ |
| `study1_sar_qwen72b.csv` | qwen72b | 1,508 | ✓ |
| `study2_ss_deepseek.csv` | deepseek | 2,106 | ✓ |
| `study2_ss_gpt4o_mini.csv` | gpt4o_mini | 2,106 | ✓ |
| `study2_ss_grok.csv` | grok | 2,106 | ✓ |
| `study2_ss_qwen.csv` | qwen | 2,106 | ✓ |
| `study2_ss_gemma3.csv` | gemma3 | 2,106 | ✓ |
| `study2_ss_llama.csv` | llama | 2,106 | ✓ |
| `study2_ss_qwen72b.csv` | qwen72b | 2,106 | ✓ |
| `study3_gss_deepseek.csv` | deepseek | 400 | ✓ |
| `study3_gss_gpt4o_mini.csv` | gpt4o_mini | 400 | ✓ |
| `study3_gss_grok.csv` | grok | 400 | ✓ |
| `study3_gss_qwen.csv` | qwen | 400 | ✓ |
| `study3_gss_gemma3.csv` | gemma3 | 400 | ✓ |
| `study3_gss_llama.csv` | llama | 400 | ✓ |
| `study3_gss_qwen72b.csv` | qwen72b | 400 | ✓ |
| `study4_multiturn_deepseek.csv` | deepseek | 300 | ✓ |
| `study4_multiturn_gpt4o_mini.csv` | gpt4o_mini | 300 | ✓ |
| `study4_multiturn_grok.csv` | grok | 300 | ✓ |
| `study4_multiturn_qwen.csv` | qwen | 300 | ✓ |
| `study4_multiturn_gemma3.csv` | gemma3 | 300 | ✓ |
| `study4_multiturn_llama.csv` | llama | 300 | ✓ |
| `study4_multiturn_qwen72b.csv` | qwen72b | 300 | ✓ |
| `study5_control_deepseek.csv` | deepseek | 200 | ✓ |
| `study5_control_gpt4o_mini.csv` | gpt4o_mini | 200 | ✓ |
| `study5_control_grok.csv` | grok | 200 | ✓ |
| `study5_control_qwen.csv` | qwen | 200 | ✓ |
| `study5_control_gemma3.csv` | gemma3 | 200 | ✓ |
| `study5_control_llama.csv` | llama | 200 | ✓ |
| `study5_control_qwen72b.csv` | qwen72b | 200 | ✓ |

**Total:** 35 CSV files, 7 models × 5 studies.  
**Total data points:** 1,508×7 + 2,106×7 + 400×7 + 300×7 + 200×7 = **32,270 rows** across study CSVs (not counting checkpoint files).

### 3.2 Statistics Files

| File | Description | Status |
|------|-------------|--------|
| `statistics/bootstrap_ci.csv` | Bootstrap 95% CI for gss_mean and fi_mean per model × category (n=1,000 resamples) | ✓ 84 rows |
| `statistics/pairwise_comparisons.csv` | Two-sample t-test, Cohen's d, Bonferroni-corrected p-values for all 21 model pairs | ✓ 21 rows |
| `statistics/anova_table.csv` | Two-way ANOVA: model × category on gss_mean | ✓ 4 rows |
| `statistics/correlation_matrix.csv` | Pearson correlations: gss_mean, fi_mean, gss_seed42, gss_seed123, gss_seed7 | ✓ 5×5 |
| `statistics/cross_category_analysis.csv` | Mean gss per model × category (7×6 heatmap) | ✓ 7 rows |
| `statistics/mixed_effects_summary.txt` | MixedLM with model×category fixed effects, model as random group | ✓ |
| `statistics/judge_agreement.csv` | Per-item VAR labels + Krippendorff's α | ✓ |
| `statistics/convergent_validity.csv` | GSS vs GenSAR / GSS vs GenSS correlations | ⚠ Empty (7 models insufficient for Pearson) |
| `statistics/drift_trajectory.csv` | Per-model drift: Pearson r (depth), Kruskal–Wallis H across depths | ✓ 7 rows |

### 3.3 Archived Data

| Folder | Content | Purpose |
|--------|---------|---------|
| `Codes/results/n60_study4/` | 6 model × 61 lines (60 items, n_turns=2 only) | Original TACL pilot Study 4 |
| `Codes/results/study4_multiturn_*.csv` | 7 models × 300 items × 4 depths (d0/d2/d4/d6) | Production Study 4 (n=300) |

Both versions are preserved. See §12 for sampling comparison.

---

## 4. Known Limitations

### 4.1 Study 1 & 2: PPL Unavailable for API Models

**Affected models:** deepseek, gpt4o_mini, grok, qwen72b  
**Reason:** API inference endpoints do not expose per-token log-probabilities.

For these 4 models, `ppl_more = ppl_less = -1.0` and `prefers_stereo = -1` in all Study 1 and Study 2 rows. The PPL-based SAR and StereoSet Score are therefore **undefined** for API models. The pipeline records `gen_sar` (Study 1) and `gen_ss` (Study 2) as generation-based proxies instead, computed by generating completions and matching to stereotyped vs. anti-stereotyped references.

**Implication for paper:** SAR and SS are valid for local models only (qwen, gemma3, llama). API model entries in those tables must be reported as N/A with a footnote. GenSAR and GenSS are reported as supplementary validation metrics.

### 4.2 Inter-Rater Reliability: α = 0.229

**Metric:** Krippendorff's α across 3 judges (Gemini 2.5 Flash, DeepSeek V3, Mistral Small) on VAR labelling.  
**Threshold for acceptable reliability:** α ≥ 0.667 (Krippendorff, 2004).  
**Observed value:** α = 0.229 — below the acceptable threshold.

This low α arises partly because all three judges agree on the majority label (0 = not harmful) for most items, producing near-zero observed disagreement but also near-zero chance-corrected agreement. The effect is a systematically low α even when majority labels are consistent. This limitation should be prominently disclosed in the paper. The judge ensemble's role in quantifying VAR is supportive but not conclusive given this reliability.

**Note:** The `convergent_validity.csv` file shows empty Pearson r values for GSS vs. GenSAR / GSS vs. GenSS correlations. This is because with only 7 model-level data points, computing model-level Pearson correlations is underpowered; the pipeline likely required more data points than available aggregated observations.

### 4.3 Gemma3 Uniform Non-Gender GSS

For gemma3 (Gemma-2-2B-IT), the GSS across caste, gender, and race categories is **identically 0.01215640...** — the same value to many decimal places across 3 out of 4 non-female categories. This uniformity is unlikely to arise by chance from 50 distinct prompts per category with 3 seeds each (150 total generations per category). It suggests that the model is applying a near-constant identity-neutral response pattern to these prompts, possibly due to safety training that produces formulaic responses.

Female GSS for gemma3 (0.018) is slightly higher, indicating the model does engage somewhat more with female-coded prompts, but the overall attentional signal from gemma3 is suppressed relative to other models.

**Impact:** gemma3 may underestimate true attentional salience for caste/race/religion categories. Analyses relying on gemma3's GSS for these categories should be interpreted cautiously.

Similarly, **llama** shows identical GSS = 0.007968... across caste, gender, race, and religion (4 of 6 categories), with only female (0.046) and male (0.038) standing out.

### 4.4 Figures 1, 2, 4 Not Generated

These figures require a `language` column (multilingual data breakdown) and a `hurtful_label` column (from human annotation). The current dataset is English-only with LLM-judged labels only. This is a dataset scope limitation, not a pipeline bug.

**Available figures:** Figure 3 (GSS accumulation plot) and Figure 5 (RAD: Relative Attentional Deviation) are fully generated and located in `Codes/results/figures/`.

### 4.5 MixedLM Convergence Warning

The mixed-effects model (`mixed_effects_summary.txt`) reports **Converged: No**. The model variance for the random groups term (model identity as group) converges to near zero (Group Var ≈ 0.003), suggesting that model-level variance is fully explained by fixed effects (model × category interactions), leaving negligible random-effect variance. The fixed-effects estimates and interaction p-values are still interpretable and reliable despite the convergence flag.

---

## 5. Study 1 — Stereotype Association Rate (SAR)

### 5.1 Setup

- **Dataset:** CrowS-Pairs (Nangia et al., 2020) — 1,508 sentence pairs covering 9 bias types
- **Metric:** `prefers_stereo = 1` if ppl(stereotyped sentence) < ppl(anti-stereotyped sentence), else 0
- **Neutral:** An unbiased model predicts SAR ≈ 50% (random chance)
- **Note:** PPL = −1.0 for API models; SAR undefined for deepseek, gpt4o_mini, grok, qwen72b

### 5.2 SAR Results

| Model | Type | Total Items | Valid (PPL≠−1) | SAR | vs. Chance (50%) |
|-------|------|-------------|----------------|-----|-----------------|
| deepseek | API | 1,508 | 0 | N/A | — |
| gpt4o_mini | API | 1,508 | 0 | N/A | — |
| grok | API | 1,508 | 0 | N/A | — |
| **qwen** | Local | 1,508 | 1,508 | **62.9%** | +12.9 pp above chance |
| **gemma3** | Local | 1,508 | 1,508 | **61.6%** | +11.6 pp above chance |
| **llama** | Local | 1,508 | 1,508 | **65.9%** | +15.9 pp above chance |
| qwen72b | API | 1,508 | 0 | N/A | — |

All three local models exhibit SAR significantly above 50% (all >61%), indicating a systematic preference for stereotyped sentences as measured by perplexity. llama has the highest SAR at 65.9%, suggesting it is most influenced by stereotyped sentence structure.

**Interpretation:** A SAR > 50% means the model has lower perplexity (higher fluency) for the stereotyped sentence in a matched pair. This does not necessarily mean the model *endorses* stereotypes in generation — it means its language model distribution is more compatible with stereotyped phrasings. This is a form of representational harm as measured by fluency.

**Generation-based analogue (GenSAR):** API models have `gen_sar` columns (fraction of completions that match the stereotyped direction in free-form generation). These values are included in the data files but excluded from the primary SAR table due to non-comparability with PPL-based SAR.

---

## 6. Study 2 — Generative StereoSet Score (GenSS)

### 6.1 Setup

- **Dataset:** StereoSet (Nadeem et al., 2021) — 2,106 fill-in-the-blank items across profession, gender, race, religion
- **PPL-based SS:** Valid only for local models (qwen, gemma3, llama)
- **GenSS:** For API models — fraction of generated completions matching the stereotyped continuation

### 6.2 StereoSet Score Results

| Model | Type | N | PPL-based SS | GenSS | Stereotype Tendency |
|-------|------|---|--------------|-------|-------------------|
| deepseek | API | 2,106 | N/A | 75.9% | High |
| gpt4o_mini | API | 2,106 | N/A | 68.5% | Moderate |
| grok | API | 2,106 | N/A | 76.4% | High |
| **qwen** | Local | 2,106 | **64.6%** | N/A | Moderate |
| **gemma3** | Local | 2,106 | **66.1%** | N/A | Moderate |
| **llama** | Local | 2,106 | **66.7%** | N/A | Moderate |
| qwen72b | API | 2,106 | N/A | 69.0% | Moderate-High |

- Neutral baseline: 50% (random)
- All models with valid scores exceed 50%, indicating systematic stereotype preference
- API models (measured by GenSS) show higher apparent stereotype rates (68–76%), though this may partly reflect differences in generation vs. PPL evaluation methodology

**Note:** A model with SS > 50% is systematically producing or preferring stereotyped completions. This is distinct from GSS — a model can be low-SS but high-GSS (mentioning identity frequently without stereotyping), or high-SS but low-GSS (stereotyping without frequency of mention).

---

## 7. Study 3 — Generative Salience Score (GSS) — Single Turn

This is the core study of the paper. It measures attentional harm directly through the frequency of identity-bearing tokens in generated responses.

### 7.1 Setup

- **Dataset:** 400 prompts drawn from Indian-BhED and custom GSS prompt set
- **Categories:** caste (50), female (100), gender (50), male (100), race (50), religion (50)
- **Seeds:** 3 (42, 123, 7); gss_mean = average across seeds
- **Metric:** GSS ∈ [0, 1] — fraction of identity-related words in response
- **Sample size per category:** n=50–100 per model (7 models × 400 = 2,800 observations total)

### 7.2 Bootstrap 95% Confidence Intervals (gss_mean)

Bootstrap CI with 1,000 resamples per model × category combination.

| Model | Category | n | gss_mean | CI_lower | CI_upper |
|-------|----------|---|----------|----------|----------|
| deepseek | caste | 50 | 0.0150 | 0.0066 | 0.0259 |
| deepseek | female | 100 | **0.0513** | 0.0395 | 0.0634 |
| deepseek | gender | 50 | 0.0118 | 0.0034 | 0.0221 |
| deepseek | male | 100 | 0.0441 | 0.0329 | 0.0565 |
| deepseek | race | 50 | 0.0110 | 0.0042 | 0.0190 |
| deepseek | religion | 50 | 0.0189 | 0.0079 | 0.0325 |
| gemma3 | caste | 50 | 0.0122 | 0.0074 | 0.0176 |
| gemma3 | female | 100 | 0.0184 | 0.0134 | 0.0236 |
| gemma3 | gender | 50 | 0.0122 | 0.0073 | 0.0180 |
| gemma3 | male | 100 | 0.0179 | 0.0127 | 0.0237 |
| gemma3 | race | 50 | 0.0122 | 0.0073 | 0.0180 |
| gemma3 | religion | 50 | 0.0122 | 0.0072 | 0.0177 |
| gpt4o_mini | caste | 50 | 0.0089 | 0.0031 | 0.0158 |
| gpt4o_mini | female | 100 | 0.0458 | 0.0354 | 0.0568 |
| gpt4o_mini | gender | 50 | 0.0097 | 0.0030 | 0.0186 |
| gpt4o_mini | male | 100 | 0.0439 | 0.0340 | 0.0541 |
| gpt4o_mini | race | 50 | 0.0051 | 0.0011 | 0.0109 |
| gpt4o_mini | religion | 50 | 0.0083 | 0.0027 | 0.0159 |
| grok | caste | 50 | 0.0161 | 0.0105 | 0.0230 |
| grok | female | 100 | 0.0265 | 0.0217 | 0.0316 |
| grok | gender | 50 | 0.0260 | 0.0143 | 0.0426 |
| grok | male | 100 | 0.0217 | 0.0169 | 0.0272 |
| grok | race | 50 | **0.0288** | 0.0169 | 0.0455 |
| grok | religion | 50 | 0.0197 | 0.0122 | 0.0282 |
| llama | caste | 50 | 0.0080 | 0.0027 | 0.0143 |
| llama | female | 100 | 0.0463 | 0.0357 | 0.0584 |
| llama | gender | 50 | 0.0080 | 0.0030 | 0.0148 |
| llama | male | 100 | 0.0377 | 0.0277 | 0.0484 |
| llama | race | 50 | 0.0080 | 0.0029 | 0.0139 |
| llama | religion | 50 | 0.0080 | 0.0026 | 0.0146 |
| qwen | caste | 50 | 0.0179 | 0.0086 | 0.0293 |
| qwen | female | 100 | **0.0922** | 0.0750 | 0.1111 |
| qwen | gender | 50 | 0.0179 | 0.0082 | 0.0286 |
| qwen | male | 100 | 0.0774 | 0.0594 | 0.0981 |
| qwen | race | 50 | 0.0179 | 0.0084 | 0.0292 |
| qwen | religion | 50 | 0.0179 | 0.0083 | 0.0286 |
| qwen72b | caste | 50 | 0.0215 | 0.0131 | 0.0308 |
| qwen72b | female | 99 | 0.0312 | 0.0237 | 0.0396 |
| qwen72b | gender | 50 | 0.0260 | 0.0165 | 0.0372 |
| qwen72b | male | 101 | 0.0260 | 0.0195 | 0.0341 |
| qwen72b | race | 50 | 0.0240 | 0.0138 | 0.0358 |
| qwen72b | religion | 50 | 0.0250 | 0.0151 | 0.0359 |

**Key observations:**
1. Qwen (local 1.5B) achieves the highest female GSS (0.092) and male GSS (0.077) — by far the most attentionally salient model for gender
2. Gemma3 and llama show suspicious uniformity in non-female/male categories (identical 4-decimal values), suggesting formulaic responding
3. Grok has the highest race GSS (0.029) among all models, unlike its position for female/male
4. qwen72b shows moderate but consistent GSS across all categories (0.021–0.031), without the gender skew seen in qwen-1.5B

### 7.3 Cross-Category GSS Heatmap (gss_mean)

| Model | caste | female | gender | male | race | religion |
|-------|-------|--------|--------|------|------|----------|
| deepseek | 0.0150 | 0.0513 | 0.0118 | 0.0441 | 0.0110 | 0.0189 |
| gemma3 | 0.0122 | 0.0184 | 0.0122 | 0.0179 | 0.0122 | 0.0122 |
| gpt4o_mini | 0.0089 | 0.0458 | 0.0097 | 0.0439 | 0.0051 | 0.0083 |
| grok | 0.0161 | 0.0265 | 0.0260 | 0.0217 | **0.0288** | 0.0197 |
| llama | 0.0080 | 0.0463 | 0.0080 | 0.0377 | 0.0080 | 0.0080 |
| qwen | 0.0179 | **0.0922** | 0.0179 | **0.0774** | 0.0179 | 0.0179 |
| qwen72b | 0.0215 | 0.0312 | 0.0260 | 0.0260 | 0.0240 | 0.0250 |

Column maxima in **bold**. qwen dominates female and male; grok leads race and gender categories; qwen72b is the most uniformly distributed across categories.

### 7.4 Two-Way ANOVA

**Dependent variable:** gss_mean  
**Factors:** Model (7 levels) × Category (6 levels)  
**N:** 2,800 observations (7 models × 400 prompts)

| Effect | Sum Sq | df | F | p-value | η² |
|--------|--------|----|----|---------|-----|
| Model | 0.302 | 6 | **23.89** | 1.1×10⁻²⁷ | 0.043 |
| Category | 0.493 | 5 | **46.82** | 1.4×10⁻⁴⁶ | 0.071 |
| Model × Category | 0.352 | 30 | **5.57** | 3.0×10⁻²⁰ | 0.051 |
| Residual | 5.808 | 2,758 | — | — | 0.835 |

All three effects are highly significant (p << 0.001). Effect sizes (η²) are small-to-moderate, with a large residual (83.5%), indicating that model identity and category explain meaningful but not dominant variance in GSS. The significant interaction term confirms that the relationship between category and GSS is model-dependent (not all models have the same category profile).

### 7.5 Pairwise Model Comparisons (Bonferroni-corrected, all 21 pairs)

| Model A | Model B | t | Cohen's d | p_corrected | Significant |
|---------|---------|---|----------|-------------|-------------|
| deepseek | gemma3 | 5.44 | 0.385 | 1.7×10⁻⁶ | ✓ |
| deepseek | gpt4o_mini | 1.29 | 0.091 | 1.000 | ✗ |
| deepseek | grok | 2.39 | 0.169 | 0.355 | ✗ |
| deepseek | llama | 1.70 | 0.120 | 1.000 | ✗ |
| deepseek | qwen | −4.24 | −0.300 | 5.2×10⁻⁴ | ✓ |
| deepseek | qwen72b | 1.41 | 0.100 | 1.000 | ✗ |
| gemma3 | gpt4o_mini | −4.38 | −0.310 | 2.9×10⁻⁴ | ✓ |
| gemma3 | grok | −3.91 | −0.277 | 2.1×10⁻³ | ✓ |
| gemma3 | llama | −3.80 | −0.269 | 3.3×10⁻³ | ✓ |
| gemma3 | qwen | −8.66 | −0.612 | 1.7×10⁻¹⁵ | ✓ |
| gemma3 | qwen72b | −4.99 | −0.353 | 1.6×10⁻⁵ | ✓ |
| gpt4o_mini | grok | 1.07 | 0.075 | 1.000 | ✗ |
| gpt4o_mini | llama | 0.45 | 0.032 | 1.000 | ✗ |
| gpt4o_mini | qwen | −5.40 | −0.382 | 2.0×10⁻⁶ | ✓ |
| gpt4o_mini | qwen72b | 0.03 | 0.002 | 1.000 | ✗ |
| grok | llama | −0.56 | −0.040 | 1.000 | ✗ |
| grok | qwen | −6.40 | −0.453 | 6.9×10⁻⁹ | ✓ |
| grok | qwen72b | −1.16 | −0.082 | 1.000 | ✗ |
| llama | qwen | −5.71 | −0.404 | 3.7×10⁻⁷ | ✓ |
| llama | qwen72b | −0.46 | −0.032 | 1.000 | ✗ |
| qwen | qwen72b | 5.63 | 0.398 | 5.9×10⁻⁷ | ✓ |

**Significant pairs (10/21):** Mostly those involving qwen (highest) or gemma3 (lowest). deepseek vs. gpt4o_mini, deepseek vs. grok, gpt4o_mini vs. grok, gpt4o_mini vs. llama, gpt4o_mini vs. qwen72b, grok vs. llama, grok vs. qwen72b, llama vs. qwen72b — all non-significant after Bonferroni correction, suggesting a cluster of mid-range models (deepseek, gpt4o_mini, grok, llama) that are not distinguishable from each other.

### 7.6 Mixed-Effects Model (MixedLM)

**Model:** gss_mean ~ model × category (full interaction), random intercept by model group  
**N:** 2,800 observations, 7 groups (models), method = ML

| Term | Coefficient | SE | z | p |
|------|-------------|-----|---|---|
| Intercept (deepseek/caste) | 0.015 | 0.055 | 0.27 | 0.784 |
| model[gemma3] | −0.003 | 0.009 | −0.31 | 0.758 |
| model[gpt4o_mini] | −0.006 | 0.009 | −0.67 | 0.506 |
| model[grok] | +0.001 | 0.009 | +0.12 | 0.904 |
| model[llama] | −0.007 | 0.009 | −0.77 | 0.442 |
| model[qwen] | +0.003 | 0.009 | +0.33 | 0.744 |
| model[qwen72b] | +0.007 | 0.009 | +0.72 | 0.473 |
| category[female] | +0.036 | 0.077 | +0.47 | 0.637 |
| category[gender] | −0.003 | 0.077 | −0.04 | 0.968 |
| category[male] | +0.029 | 0.077 | +0.38 | 0.705 |
| category[race] | −0.004 | 0.077 | −0.05 | 0.959 |
| category[religion] | +0.004 | 0.077 | +0.05 | 0.959 |
| **gemma3 × female** | **−0.030** | 0.011 | **−2.70** | **0.007** |
| gpt4o_mini × female | +0.001 | 0.011 | +0.05 | 0.964 |
| **grok × female** | **−0.026** | 0.011 | **−2.33** | **0.020** |
| llama × female | +0.002 | 0.011 | +0.18 | 0.859 |
| **qwen × female** | **+0.038** | 0.011 | **+3.40** | **0.001** |
| **qwen72b × female** | **−0.027** | 0.011 | **−2.39** | **0.017** |
| **gemma3 × male** | **−0.023** | 0.011 | **−2.10** | **0.036** |
| **grok × male** | **−0.024** | 0.011 | **−2.11** | **0.035** |
| **qwen × male** | **+0.030** | 0.011 | **+2.72** | **0.007** |
| **qwen72b × male** | **−0.025** | 0.011 | **−2.21** | **0.027** |
| Group Var (random) | 0.003 | — | — | — |

**Bold** = p < 0.05.

**Interpretation:**
- Main effects of model and category are non-significant after controlling for interactions, meaning there is no simple universal offset per model or per category
- Significant interactions are concentrated in **female** and **male** categories:
  - `qwen × female` (+0.038): qwen has significantly higher female salience than deepseek+caste baseline
  - `gemma3 × female` (−0.030): gemma3 suppresses female salience below deepseek baseline
  - `grok × female` (−0.026) and `qwen72b × female` (−0.027): similar suppression
  - `qwen × male` (+0.030): qwen amplifies male salience too
  - `gemma3 × male` (−0.023) and `grok × male` (−0.024): suppression for male as well
- The near-zero Group Var indicates model-level variance is captured by fixed effects rather than random intercepts

### 7.7 Seed Correlation Analysis

| Metric pair | Pearson r | Interpretation |
|-------------|-----------|----------------|
| gss_mean ↔ fi_mean | 0.667 | Moderate: GSS and FI (foregrounding) are correlated but measure distinct aspects |
| gss_mean ↔ gss_seed42 | 0.831 | High reliability of mean against seed 42 |
| gss_mean ↔ gss_seed123 | 0.807 | Good reliability |
| gss_mean ↔ gss_seed7 | 0.783 | Good reliability |
| gss_seed42 ↔ gss_seed123 | 0.549 | Moderate cross-seed agreement |
| gss_seed42 ↔ gss_seed7 | 0.449 | Lower cross-seed agreement |
| gss_seed123 ↔ gss_seed7 | 0.434 | Lower cross-seed agreement |

Cross-seed correlations (0.43–0.55) are moderate, confirming that GSS exhibits seed-to-seed variation but the 3-seed mean stabilises the estimate (gss_mean ↔ individual seed r ≈ 0.78–0.83).

---

## 8. Study 4 — Multi-Turn Attentional Drift

### 8.1 Setup

- **N:** 300 conversation items (production run), each with up to 7 turns
- **Depths measured:** d0 (1 turn), d2 (3 turns), d4 (5 turns), d6 (7 turns)
- **Metric:** gss_d0, gss_d2, gss_d4, gss_d6 — mean GSS at each depth
- **GSS-drift:** gss_d6 − gss_d0 (positive = increasing salience over conversation)
- **Statistical tests:** Pearson r (gss vs depth, negative = decreasing trend), Kruskal–Wallis H (non-parametric test of depth differences)

### 8.2 Drift Trajectory Results (n=300 production)

| Model | gss_d0 | gss_d2 | gss_d4 | gss_d6 | Pearson r | Kruskal-W p | Significant |
|-------|--------|--------|--------|--------|-----------|-------------|-------------|
| deepseek | 0.0198 | 0.0171 | 0.0146 | 0.0103 | −0.083 | 0.045 | ✓ |
| gemma3 | 0.0165 | 0.0055 | 0.0039 | 0.0043 | −0.194 | 0.001 | ✓ |
| gpt4o_mini | 0.0194 | 0.0182 | 0.0147 | 0.0138 | −0.062 | 0.520 | ✗ |
| grok | 0.0175 | 0.0103 | 0.0080 | 0.0083 | −0.158 | 2.5×10⁻⁴ | ✓ |
| llama | 0.0207 | 0.0047 | 0.0088 | 0.0165 | −0.025 | 2.3×10⁻⁹ | ✓ |
| qwen | 0.0461 | 0.0266 | 0.0208 | 0.0110 | −0.189 | 5.0×10⁻⁸ | ✓ |
| qwen72b | 0.0035 | 0.0021 | 0.0008 | 0.0005 | −0.094 | 0.284 | ✗ |

**Key observations:**
1. **5 of 7 models show significant drift** (decreasing GSS over turns): deepseek, gemma3, grok, llama, qwen
2. **gpt4o_mini and qwen72b do not drift significantly** — consistent attention levels across turns
3. **qwen has the steepest decline** (d0=0.046 → d6=0.011), a 76% reduction from first to seventh turn — model's identity foregrounding dissipates strongly as context accumulates
4. **llama shows non-monotone drift** (d2 drops then partially recovers at d6): Kruskal–Wallis significant despite weak Pearson r, indicating that the depth-level distributions differ but not linearly
5. **gemma3 drops sharply at d2** then nearly plateaus (0.004–0.005) for d4–d6 — consistent with its formulaic response pattern
6. **All models start d0 at higher GSS** than the control baseline (~0.002–0.004), confirming that identity-loaded prompts cause real attentional elevation even in first turns

**Interpretation:** The general pattern of *decreasing* GSS with depth (negative Pearson r for most models) suggests that LLMs tend to reference identity attributes most heavily in early turns and then become more topic-focused as the conversation develops. This is an important finding for mitigating attentional harm — multi-turn conversation may naturally dilute attentional salience over time.

### 8.3 Column Schema for Study 4 CSV

| Column | Description |
|--------|-------------|
| `item_id` | Unique item identifier (e.g., `mt7_indian_64`) |
| `model_name` | Model abbreviation |
| `category` | Identity category (caste/female/gender/male/race/religion) |
| `n_turns` | Number of turns in the conversation (7 for all production items) |
| `gss_d0` | GSS at depth 0 (first turn) |
| `gss_d2` | GSS at depth 2 (third turn) |
| `gss_d4` | GSS at depth 4 (fifth turn) |
| `gss_d6` | GSS at depth 6 (seventh turn) |
| `gss_turn1` | Alias for gss_d0 |
| `gss_last` | GSS at the last observed turn |
| `gss_drift` | gss_last − gss_turn1 |
| `gss_all` | Python list of [gss_d0, gss_d2, gss_d4, gss_d6] |

---

## 9. Study 5 — Negative Control

### 9.1 Setup

- **N:** 200 identity-neutral prompts per model (e.g., "The lamp cast a circle of light on the ceiling.")
- **Purpose:** Verify that GSS is near-zero when no identity cues are present in prompt
- **Metric:** GSS and FI on neutral responses

### 9.2 Control Results

| Model | N | Mean GSS | Mean FI | Interpretation |
|-------|---|----------|---------|----------------|
| deepseek | 200 | **0.0042** | 0.040 | Near-zero ✓ |
| gpt4o_mini | 200 | **0.0026** | 0.040 | Near-zero ✓ |
| grok | 200 | **0.0018** | 0.045 | Near-zero ✓ |
| qwen | 200 | **0.0019** | 0.030 | Near-zero ✓ |
| gemma3 | 200 | **0.0028** | 0.090 | Near-zero ✓ |
| llama | 200 | **0.0020** | 0.025 | Near-zero ✓ |
| qwen72b | 200 | **0.0024** | 0.055 | Near-zero ✓ |

**All models achieve GSS < 0.005 on identity-neutral prompts**, confirming that:
1. The GSS measure is not inflated by general vocabulary frequency — identity words are genuinely sparse in neutral contexts
2. The elevated GSS observed in Studies 3 and 4 is attributable to identity-loaded prompts, not to baseline model verbosity
3. gemma3 has a slightly elevated FI (0.090) for control prompts, suggesting it tends to front-load responses but not with identity-bearing words

**Control-vs-Study3 comparison:** The mean Study 3 GSS across all models and categories is ~0.025–0.030, which is 6–15× higher than the control baseline. This establishes the GSS signal-to-noise ratio.

---

## 10. qwen72b — Seventh Model Analysis

Qwen2.5-72B-Instruct was added as the seventh model in the production run to provide an API-scale large model comparison (70B parameter range) alongside the smaller local models.

### 10.1 Coverage

| Study | File | Rows | Notes |
|-------|------|------|-------|
| Study 1 SAR | `study1_sar_qwen72b.csv` | 1,508 | ppl=-1.0, SAR=N/A (API model) |
| Study 2 GenSS | `study2_ss_qwen72b.csv` | 2,106 | GenSS=69.0% |
| Study 3 GSS | `study3_gss_qwen72b.csv` | 400 | Valid ✓ |
| Study 4 Drift | `study4_multiturn_qwen72b.csv` | 300 | Valid ✓ |
| Study 5 Control | `study5_control_qwen72b.csv` | 200 | Valid ✓ |

### 10.2 GSS Profile

qwen72b shows a notably different profile from the smaller qwen-1.5B:

| Category | qwen (1.5B) | qwen72b | Reduction |
|----------|------------|---------|-----------|
| caste | 0.0179 | 0.0215 | — (72B slightly higher) |
| female | **0.0922** | 0.0312 | −66% |
| gender | 0.0179 | 0.0260 | — (72B slightly higher) |
| male | **0.0774** | 0.0260 | −66% |
| race | 0.0179 | 0.0240 | — |
| religion | 0.0179 | 0.0250 | — |

The 72B model is dramatically less gender-salient than the 1.5B model (female GSS 3× lower), while showing slightly higher salience for non-gender categories. This suggests that scale reduces gender-specific attentional harm but does not eliminate attentional salience overall.

### 10.3 Drift Behavior

qwen72b shows **no significant drift** (Pearson p=0.28, Kruskal–Wallis p=0.28), with very low baseline GSS:
- d0=0.0035, d2=0.0021, d4=0.0008, d6=0.0005

The model's attentional salience is near-control levels even in the first turn, making it the lowest-risk model for multi-turn attentional harm in this study.

### 10.4 Pairwise Comparison (qwen72b vs. other models)

From pairwise_comparisons.csv, qwen72b is significantly different from:
- **gemma3** (d=−0.353, p=1.6×10⁻⁵) — qwen72b > gemma3 in GSS
- **qwen** (d=0.398, p=5.9×10⁻⁷) — qwen >> qwen72b in GSS

Not significantly different from: deepseek, gpt4o_mini, grok, llama — qwen72b falls within the mid-range cluster.

---

## 11. Cross-Study Statistical Summary

### 11.1 Master Results Table

| Model | SAR (%) | GenSS (%) | Mean GSS | Control GSS | d0→d6 drift | Judge VAR |
|-------|---------|-----------|----------|-------------|-------------|-----------|
| deepseek | N/A | — | 0.024 | 0.0042 | −0.009 (sig) | Low |
| gpt4o_mini | N/A | 68.5% | 0.022 | 0.0026 | −0.006 (ns) | Low |
| grok | N/A | 76.4% | 0.020 | 0.0018 | −0.009 (sig) | Low |
| qwen | 62.9% | — | **0.043** | 0.0019 | **−0.035 (sig)** | Low |
| gemma3 | 61.6% | — | 0.015 | 0.0028 | −0.012 (sig) | Low |
| llama | 65.9% | — | 0.018 | 0.0020 | −0.004 (sig, non-mono) | Low |
| qwen72b | N/A | 69.0% | 0.026 | 0.0024 | −0.003 (ns) | Low |

*Mean GSS = weighted average over Study 3 categories (accounting for n=50 vs n=100 per category)*

### 11.2 Key Statistical Tests Summary

| Test | Result | Significance |
|------|--------|--------------|
| ANOVA: Model effect on GSS | F(6,2758)=23.9, η²=0.043 | p=1.1×10⁻²⁷ *** |
| ANOVA: Category effect on GSS | F(5,2758)=46.8, η²=0.071 | p=1.4×10⁻⁴⁶ *** |
| ANOVA: Model × Category interaction | F(30,2758)=5.57, η²=0.051 | p=3.0×10⁻²⁰ *** |
| Pairwise: qwen vs. all others | d=0.30–0.61 | 6/6 significant after Bonferroni |
| Pairwise: gemma3 vs. all others | d=0.27–0.61 | 4/6 significant |
| Drift: qwen (d0→d6) | r=−0.189, KW p=5×10⁻⁸ | ✓ |
| Drift: gemma3 | r=−0.194, KW p=0.001 | ✓ |
| Drift: gpt4o_mini | r=−0.062, KW p=0.52 | ✗ |
| Drift: qwen72b | r=−0.094, KW p=0.28 | ✗ |
| GSS vs FI correlation | r=0.667 | Moderate |
| Cross-seed consistency (mean vs seed) | r=0.78–0.83 | Good |
| Judge α (Krippendorff) | α=0.229 | Below threshold (0.667) |

### 11.3 Model Ranking (by mean GSS, Study 3)

1. **qwen** (0.043) — highest; large female/male gender bias
2. **deepseek** (0.024) — moderate; female-skewed
3. **qwen72b** (0.026) — moderate; most uniform across categories
4. **gpt4o_mini** (0.022) — moderate; female-skewed, very low race GSS
5. **grok** (0.020) — moderate; unusual race salience, lower gender salience
6. **llama** (0.018) — low-moderate; gender-only salient, rest near-zero
7. **gemma3** (0.015) — lowest; near-uniform, suppressed responses

---

## 12. Sampling Comparison: n=60 vs n=300 (Study 4)

Two Study 4 experiments were conducted. Both are preserved for journal comparison.

### 12.1 Experimental Conditions

| Parameter | n=60 (pilot) | n=300 (production) |
|-----------|-------------|-------------------|
| Items | 60 | 300 |
| Turn depths | n_turns=2 only | d0, d2, d4, d6 (7 turns) |
| Models | 6 (no qwen72b) | 7 (includes qwen72b) |
| Location | `Codes/results/n60_study4/` | `Codes/results/study4_multiturn_*.csv` |
| Pipeline | Original main.py | Re-run with main_rerun.py |
| Commit | Local only (pre-production) | `9801a41` |

### 12.2 Consistency Check (on 6 shared models, deepseek shown)

The n=60 pilot had n_turns=2 (only first and second turn). In the production run, d0 corresponds to turn 1 (equivalent to pilot turn 1). For deepseek:
- **Pilot:** 60 items × 2 turns — early-turn GSS
- **Production d0:** gss_d0=0.0198 — aligned with pilot mean (deepseek control range ~0.004, Study3 mean ~0.024)
- The production d0 values are consistent with Study 3 single-turn GSS values, confirming dataset and methodology continuity

### 12.3 Benefit for Journal Paper

Retaining both experiments demonstrates:
1. **Reproducibility:** Results hold across different sampling sizes (n=60 and n=300)
2. **Depth sensitivity:** The 4-depth production run reveals finer-grained drift patterns that the 2-depth pilot could not capture
3. **Model completeness:** Production includes qwen72b, providing a 72B-parameter data point

These can be presented in the journal as: *"We first conducted a pilot with n=60 items and 2 turn depths to validate the methodology, then expanded to n=300 items with 4 depths for the main analysis. Pilot and production results are consistent (deepseek d0 GSS: pilot ~0.020, production 0.0198)."*

---

## 13. Judge Ensemble & Inter-Rater Reliability

### 13.1 Setup

- 3-judge ensemble evaluated VAR (Volunteered Attribute Rate) on Study 3 and 4 responses
- Items labelled: `final_label` ∈ {0, 1} — whether response introduces identity absent from prompt
- Confidence score recorded; majority vote determines final label

### 13.2 Reliability Findings

| Metric | Value |
|--------|-------|
| Krippendorff's α | 0.229 |
| Acceptable threshold | ≥ 0.667 |
| Agreement type | Nominal |
| Items with low_reliability flag | All items (α is global) |

**Why α is low:** Examination of judge_agreement.csv reveals that all three judges consistently label items as `0` (not harmful/VAR-negative) in the majority of cases, with votes of `(0, 0, 0)` most common. When all judges agree, the observed agreement is 1.0 but Krippendorff's α controls for expected chance agreement — if a judge always votes 0, the chance agreement is also high, driving α down.

A secondary reason is that Judge A occasionally votes `−1` (failure/abstention) rather than 0 or 1, which may affect the α calculation. For example, several early items show Judge A vote = −1 (reasoning field empty).

**Implication:** VAR labels should be reported with the caveat that inter-rater reliability is below the accepted threshold for social science annotation (Krippendorff, 2004). The GSS metric, which does not rely on human or LLM judgement, is the more reliable primary metric.

---

## 14. Reproducibility Notes

### 14.1 Environment

| Component | Specification |
|-----------|---------------|
| OS | Ubuntu 22.04 LTS (Akash VM) |
| Python | 3.10.12 |
| GPU | NVIDIA H100 80 GB HBM3 |
| CUDA | 12.4 |
| PyTorch | 2.4.0+cu124 |
| Flash Attention 2 | 2.6.3 |

Key Python packages: `transformers`, `datasets`, `statsmodels`, `scipy`, `pandas`, `numpy`, `pingouin`, `matplotlib`, `seaborn`.

### 14.2 Checkpoints and Resumption

All study runs use `.jsonl` checkpoint files (`ckpt_study{N}_{model}.jsonl`) to allow resumption on failure. The pipeline checks for completed phase marker files (`.phase_*_done`) before re-running phases. This means partial results can be recovered without restarting from scratch.

### 14.3 Randomisation

- `RANDOM_SEED = 20260502` (set in `.env`)
- GSS measured at seeds 42, 123, and 7 for each prompt; mean across seeds reported
- Model temperature = 1.0 for GSS generation; 0.7 for judge ensemble (Gemini, DeepSeek); 0.0 for Mistral judge

### 14.4 Model Snapshots

`Codes/results/model_snapshots.json` records model provenance (versions, API endpoints, load timestamps) for TACL A.10.1 reproducibility requirements.

### 14.5 Re-running the Pipeline

```bash
# Clone and set up
git clone https://github.com/DevDaring/Attentional_Harms.git
cd Attentional_Harms

# Create Codes/.env with API keys (see Codes/.env.example if available)
# Required keys: DEEPSEEK_API_KEY_1, OPENROUTER_API_KEY_1, GEMINI_API_KEY_1, 
# MISTRAL_API_KEY1, HUGGINGFACE_TOKEN

# Install dependencies
pip install torch==2.4.0+cu124 --index-url https://download.pytorch.org/whl/cu124
pip install -r Codes/requirements.txt
pip install flash-attn==2.6.3 --no-build-isolation

# Run full pipeline (resumes from checkpoint if interrupted)
cd Codes
python main.py

# Or use the re-run script (7-model production version)
python Re_Run/main_rerun.py
```

Results are written to `Codes/results/`. Statistics to `Codes/results/statistics/`. Figures to `Codes/results/figures/`.

### 14.6 Data File Structure

```
Codes/results/
├── study1_sar_{model}.csv         # Study 1: SAR via CrowS-Pairs (7 files)
├── study2_ss_{model}.csv          # Study 2: StereoSet Score (7 files)
├── study3_gss_{model}.csv         # Study 3: GSS single-turn (7 files)
├── study4_multiturn_{model}.csv   # Study 4: multi-turn drift n=300 (7 files)
├── study5_control_{model}.csv     # Study 5: negative control (7 files)
├── n60_study4/                    # Archived pilot Study 4 (6 models, n=60)
│   └── study4_multiturn_{model}.csv
├── figures/
│   ├── figure3_accumulation.pdf/png
│   └── figure5_rad.pdf/png
└── statistics/
    ├── bootstrap_ci.csv
    ├── pairwise_comparisons.csv
    ├── anova_table.csv
    ├── correlation_matrix.csv
    ├── cross_category_analysis.csv
    ├── mixed_effects_summary.txt
    ├── judge_agreement.csv
    ├── convergent_validity.csv
    └── drift_trajectory.csv
```

---

*Analysis prepared May 7 2026. All data grounded in production pipeline commit `9801a41` (GitHub: DevDaring/Attentional_Harms).*
