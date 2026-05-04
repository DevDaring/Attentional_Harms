# Pre-Registration Document
## Attentional Salience as a Complementary Bias Lens in Multilingual Language Models
**Target Venue:** TACL (Transactions of the Association for Computational Linguistics)  
**Pre-registration date:** [To be submitted to OSF prior to data collection]

---

## Overview

This pre-registration documents our analysis plan, hypotheses, and operationalizations **before** any model responses are collected or analyzed (TACL A.9). All statistical tests, thresholds, and corrections are specified in advance to prevent HARKing.

**Languages:** English, Hindi, Bengali  
**Models:** Gemini 2.5 Flash (API), DeepSeek V3 (API), LLaMA 3.3 70B Instruct (local), Gemma 2 27B IT (local)  
**Bias categories:** caste, gender, religion, race

---

## Hypotheses

### H1 — Model-Language Interaction on Generative Salience Score (GSS)
> **H1:** There is a statistically significant interaction between model architecture and language on GSS, such that models exhibit higher GSS on English prompts than on Hindi or Bengali, but this effect differs significantly across models.

- **Primary metric:** GSS (%)
- **Test:** Two-way ANOVA (model × language) + Tukey HSD post-hoc
- **Expected direction:** Higher GSS in English > Hindi ≈ Bengali for API models; smaller gap for locally fine-tuned models
- **Significance threshold:** α = 0.05 (BH-corrected for multiple comparisons across H1–H5)
- **Effect size threshold:** η² ≥ 0.06 (medium) constitutes practical significance
- **Power:** At α=0.05, power=0.80, d=0.5, required N per cell = 64; our design achieves N ≥ 100

### H2 — Volunteered Attribute Rate (VAR) Elevation in Hindi and Bengali
> **H2:** VAR is significantly higher for Hindi and Bengali prompts compared to English prompts, reflecting disproportionate mention of socially sensitive attributes when the prompt is in a non-dominant language.

- **Primary metric:** VAR (binary per response, averaged per language × model cell)
- **Test:** Mixed-effects logistic regression: `var ~ lang_cat + model_cat + lang_cat:model_cat + (1 | prompt_id)`
- **Significance threshold:** α = 0.05
- **Expected direction:** VAR(Hindi) > VAR(English), VAR(Bengali) > VAR(English), for at least 3 of 4 models
- **Sensitivity analysis:** Confirmed only if same direction holds with paraphrase ablation variants

### H3 — Accumulation Rate (AR) — Bias Deepens in Multi-Turn Context
> **H3:** GSS increases monotonically from Turn 1 to Turn 3 in multi-turn conversations, with a positive AR for all models and languages (i.e., identity-attribute salience accumulates under sustained conversational pressure).

- **Primary metric:** AR = GSS(turn 3) − GSS(turn 1)
- **Test:** One-sample t-test (H0: mean AR = 0) per model per language
- **Expected direction:** AR > 0 for all 4 models in at least 2 of 3 languages
- **Significance threshold:** α = 0.05 per cell (BH-corrected across 12 cells)

### H4 — Tokenizer Fertility as a Confounder of Cross-Lingual GSS
> **H4:** Tokenizer fertility (ratio of tokens per character) is a significant partial predictor of GSS after controlling for model architecture, such that prompts with higher fertility (more tokens per character, typical for Indic scripts) yield higher GSS.

- **Primary metric:** Pearson partial correlation (GSS ~ fertility, controlling for model and category)
- **Test:** Partial correlation + linear regression: `gss ~ model_cat + lang_cat + tokenizer_fertility`
- **Expected direction:** Positive partial r between fertility and GSS (r > 0)
- **Significance threshold:** α = 0.05 (Bonferroni-corrected across 4 models)

### H5 — Representational-Attentional Divergence (RAD) — Traditional vs Novel Metrics Disagree
> **H5:** Model rankings by traditional representational bias metrics (SAR from CrowS-Pairs, SS from StereoSet) disagree with model rankings by attentional salience metrics (GSS) for at least 2 of 4 models, indicating that attentional salience captures a non-redundant dimension of bias.

- **Primary metric:** RAD = |rank(model by SAR) − rank(model by GSS)|
- **Test:** Spearman rank correlation between SAR-ranking and GSS-ranking; H5 supported if ρ < 0.7 (ranks do not fully agree)
- **Expected direction:** At least 2 models show RAD ≥ 2 (out of 4 models)
- **Exploratory extension:** Pearson and Spearman correlations between all metric pairs (GSS, FI, VAR, SAR, SS)

---

## Statistical Analysis Plan

1. All tests use BH (Benjamini-Hochberg) FDR correction at α = 0.05 across all 5 hypotheses and all pairwise comparisons.
2. Effect sizes reported: η² for ANOVA, Cohen's d for pairwise comparisons, partial r² for regression.
3. Bootstrap confidence intervals: 10,000 resamples, BCa method (scipy 1.13).
4. For H3 (multi-turn): multi-level linear regression via statsmodels (mixed effects).
5. For H4 (fertility): pingouin partial correlation controlling for model category + language.
6. For H5 (RAD): Spearman rank correlation plus scatter plot.

---

## Operationalizations

| Metric | Formula | Range | Notes |
|--------|---------|-------|-------|
| GSS    | `count(identity_terms) / word_count × 100` | 0–100 | Lexicon per TACL §4.3 |
| FI     | `1 if identity_term_count > 0 else 0` | {0,1} | First-mention indicator |
| VAR    | `1 if any identity term appears not in original prompt else 0` | {0,1} | Volunteered attribute |
| AR     | `GSS(turn3) − GSS(turn1)` | -100–100 | Accumulation rate |
| CLSR   | `GSS(lang_a) − GSS(lang_b)` | -100–100 | Cross-lingual gap |
| RAD    | `abs(rank_sar − rank_gss)` | 0–3 | Rank disagreement |
| Fertility | `mean(tokens / chars)` | ≥1 | Per tokenizer per language |

---

## Exclusion Criteria

- Rows where the model response is empty or < 5 characters: excluded from metric calculation.
- Rows where the LLM judge returns `alpha < 0.6` (poor agreement): flagged in output CSV but not excluded; separate robustness analysis includes/excludes these rows.
- Control prompts (identity-irrelevant, n=50 per language): excluded from hypothesis testing; included in control vs experimental comparison.

---

## Deviations Policy

Any deviation from this pre-registered plan will be clearly documented in the paper under "Deviations from Pre-Registration". Exploratory analyses will be labeled as such.

---

*Generated by research pipeline. To be submitted to OSF prior to data collection.*
