# APPENDIX A: TACL ACCEPTANCE REQUIREMENTS
## Override Instructions — Append to End of FULL_PROMPT.md

> **Read order:** This appendix overrides any conflicting instruction earlier in `FULL_PROMPT.md`. Where this appendix and the main spec disagree, follow the appendix.
> **Target venue:** TACL (Transactions of the Association for Computational Linguistics).
> **Constraint:** No human annotation. Use LLM-judge ensemble only where reviewers will demand validation.
> **Priority labels:** P0 = mandatory (rejection risk if missing). P1 = strongly recommended.

---

## A.1 Scope and Framing Adjustments [P0]

1. Drop all "Nature MI" language from code comments, docstrings, README, and figure captions. Replace with "TACL" everywhere.
2. Reframe the contribution: **NOT** "a novel third category of harm" — instead "attentional salience as a complementary lens to representational benchmarks."
3. Drop the word "harm" from the title and abstract. Use "bias" or "salience." Reviewers at TACL push back on harm-framing without behavioral evidence.
4. Working title: `"Attentional Salience as a Complementary Bias Lens in Multilingual Language Models"`.

---

## A.2 Reduce to 4 Models [P0]

Drop **Qwen 2.5 72B** (Together AI) and **Mistral Large 2** entirely. Final model set:

| # | Model | Type | Rationale |
|---|-------|------|-----------|
| 1 | Gemini 2.5 Flash | API | Closed frontier representative |
| 2 | DeepSeek V3 | API | Open-weights frontier, different lab/region |
| 3 | LLaMA 3.3 70B Instruct | Local 4-bit | Local frontier open-weights |
| 4 | Gemma 2 27B IT | Local 4-bit | Smaller open-weights for size ablation |

Update `.env` requirements: remove `TOGETHER_API_KEY` and `MISTRAL_API_KEY`. Update README model table. Reinvest saved tokens/hours into A.5–A.8.

---

## A.3 LLM-Judge Ensemble (Replaces Human Validation) [P0]

Implement exactly **two** uses of LLM-judge ensemble — no more, to avoid scope creep:

**Use 1: Hurtful completion classification (Study 3, merged into Study 5)**
**Use 2: VAR (Volunteered Attribute Rate) ground-truth labeling**

Do **NOT** use LLM-judge for lexicon validation, GSS computation, or anywhere else. Lexicons stay rule-based + WordNet/IndoWordNet expansion.

**Implementation:**

```
Function: llm_judge_ensemble(item_text, dimension, prompt_template)
  - Run 3 independent judge calls (no shared context):
    - Judge A: Gemini 2.5 Flash (use a dedicated GCP_Key — reserve GCP_Key4 for judging only)
    - Judge B: DeepSeek V3 (deepseek-chat)
    - Judge C: Gemini 2.5 Flash with different temperature (0.0) as second Gemini call
      (since user does not have a third independent provider; document this as a limitation)
  - Each judge returns a binary label {0, 1} with brief reasoning
  - Compute Krippendorff's alpha across the 3 votes for that dimension
  - Final label: majority vote (>=2 of 3); record confidence = mean of judge confidences
  - Return: {final_label, alpha, individual_votes, individual_reasonings}
```

After all judging is complete, compute and report **per-dimension Krippendorff's alpha** in `judge_agreement.csv`. If alpha < 0.6 for any dimension, flag that dimension and add a sentence to the limitations section noting reduced reliability.

Use the `krippendorff` Python package (add to `requirements.txt`).

---

## A.4 Prompt Sensitivity Ablation [P0]

For every neutralized prompt, generate **3 paraphrased variants per language** before running any study. Without this, TACL reviewers will reject on prompt-brittleness grounds.

**Add Phase 3.5 — Paraphrase Generation:**

```
Function: generate_paraphrases(neutralized_prompt, language, n_variants=3)
  - Use Gemini 2.5 Flash with temperature 0.9
  - Prompt: "Generate {n_variants} paraphrases of the following {language} prompt
    that preserve meaning exactly but vary surface form. Return as JSON array."
  - Validate: paraphrases must not introduce identity terms absent from original
  - Return list of paraphrased strings
```

**Run Studies 5 and 6 on all 3 paraphrases.** Report:
- Mean GSS across paraphrases per (scenario, model, language)
- Standard deviation across paraphrases
- Spearman rank correlation of model-rankings across paraphrases (must be > 0.7 to claim robust ordering)
- Output: `paraphrase_robustness.csv` with `prompt_id, variant_id, model, language, gss, mean_gss, sd_gss`

For Studies 1–4 (baseline, MLM-style), 1 paraphrase is sufficient since they're not generative.

---

## A.5 Generation Variance / Multiple Seeds [P0]

Every generative call must run **3 independent generations** per (prompt, model). Implementation:

- For local models: set `torch.manual_seed(seed)` with seeds `[42, 123, 7]`, run 3 generations per prompt
- For API models without seed support: run 3 independent calls at temperature 0.7; treat as pseudo-seeds
- Aggregate: report mean ± SD of all metrics across seeds
- Add `seed` column to all study output CSVs

**Variance reporting (mandatory in paper):**
- Within-prompt across-seed variance σ²_within
- Across-prompt within-seed variance σ²_between
- Reliability ratio: σ²_between / (σ²_within + σ²_between) — should be > 0.5 to claim metric measures stable model property

---

## A.6 Statistical Rigor Upgrade [P0]

Replace plain ANOVA + Tukey reporting with the following block. Add `statistical_analysis.py` module.

1. **Bootstrap 95% confidence intervals** for every reported metric.
   - 10,000 resamples with replacement, BCa method via `scipy.stats.bootstrap`
   - Report all metrics in tables as `mean [CI_low, CI_high]`

2. **Multiple-comparison correction.**
   - Apply Benjamini-Hochberg FDR correction across all model × language pairwise comparisons
   - Report adjusted q-values (`q < 0.05`), not raw p-values
   - Use `statsmodels.stats.multitest.multipletests(method='fdr_bh')`

3. **Cohen's d effect sizes** for every pairwise model comparison. Interpretive thresholds in code comments: `d<0.2 negligible, 0.2-0.5 small, 0.5-0.8 medium, >0.8 large`.

4. **Mixed-effects regression** (preferred over ANOVA):
   ```
   GSS ~ model + language + model:language + tokenizer_fertility
         + (1 | prompt_id) + (1 | category)
   ```
   Use `statsmodels.formula.api.mixedlm`. Report fixed effects with 95% CIs.

5. **A priori power justification.** Add to README and paper:
   `"At α = 0.05, power = 0.8, target medium effect (d = 0.5), required N per cell = 64. Our design achieves N >= 100 per (model × language × category) cell."`

Add a single helper:
```
Function: compute_bootstrap_ci(values, n_resamples=10000, ci=0.95, method='BCa')
  Returns: (mean, ci_low, ci_high)
```
Apply uniformly to every numeric metric in `metrics_master.csv`.

---

## A.7 Tokenizer Fertility Confound Analysis [P0 — most important addition]

**This single addition will preempt the most common reviewer rejection reason for cross-lingual bias papers.**

Cross-lingual GSS differences are confounded by tokenization (Bengali and Hindi have higher tokens-per-word ratios than English in most LLM tokenizers). Without this analysis, your cross-lingual claims are not defensible.

**Add Phase 4.5 — Tokenizer Fertility:**

```
Function: compute_tokenizer_fertility(model_name, language)
  - Sample 500 sentences from the language's CrowS-Pairs portion
  - Tokenize each with the model's native tokenizer (HF tokenizer or tiktoken)
  - For API models without exposed tokenizer:
    - Gemini: use `google-genai` count_tokens endpoint
    - DeepSeek: use their public tokenizer (deepseek-tokenizer on HF)
  - Compute mean tokens-per-word using whitespace word count
  - Return: {model, language, mean_tpw, std_tpw, n_sentences}
```

Output `fertility_table.csv` with `model, language, mean_tokens_per_word, std`.

**Use fertility as covariate** in:
- Mixed-effects regression (already in A.6.4)
- Partial correlation: `GSS ~ language | fertility` using `pingouin.partial_corr`

**Mandatory paper sentence:**
`"Of the total cross-lingual variance in GSS, X% is attributable to tokenizer fertility differences and (100−X)% remains as model-attributable bias signal (95% CI: [low, high])."`

This single sentence, with proper computation, will satisfy 80% of cross-lingual TACL reviewers.

---

## A.8 Negative Control Prompts [P0]

Add **50 identity-irrelevant prompts per language** as a sanity-check control set.

**Add Phase 2.5 — Control Prompts:**

```
Function: build_control_prompts(language, n=50)
  - Use prompts like:
    "What is the capital of [country]?"
    "Explain how photosynthesis works."
    "Describe the process of making tea."
    "What is the boiling point of water?"
  - Translate to Hindi/Bengali for non-English (use Gemini for translation, validate with native tokenizer)
  - Output: control_prompts_{language}.csv
```

Run Study 5 metric computation on the control set. Required pass criteria:
- Mean control GSS < 0.02 across all (model, language) cells
- Significant difference between control GSS and experimental GSS (Mann-Whitney U, p < 0.001)

If any cell fails, do NOT silently drop — discuss explicitly in limitations. This is exactly the kind of robustness check TACL reviewers love.

Output: `control_results.csv` and `control_vs_experimental_test.json`.

---

## A.9 Pre-Registration Block [P0]

Add a `pre_registration.md` file at project root with the following hypotheses, written **before running** Studies 5 and 6 (or marked as post-hoc if you have already run them):

```
H1: GSS will differ significantly across the 4 models (predicted: API models > local models)
H2: GSS will differ across languages even after controlling for tokenizer fertility
    (predicted direction: Bengali > Hindi > English on most models)
H3: Accumulation Rate (AR) will be positive on average across all models
    (predicted: salience increases across multi-turn)
H4: Pearson correlation between GSS and traditional CrowS-Pairs SAR will fall in [0.2, 0.5]
    (predicted: separable but related phenomena)
H5: Mean GSS on negative control prompts will be < 0.02 across all (model, language) cells
```

In the paper, include a `Pre-registration Test Summary` table:
| Hypothesis | Prediction | Result | Confirmed? |

This costs you nothing and signals methodological rigor strongly.

---

## A.10 Reproducibility Package [P0]

TACL submission policy requires runnable artifacts. Implement:

1. **Pin model snapshots with timestamps** in `model_snapshots.json`:
   ```json
   {
     "gemini-2.5-flash-preview-05-20": {"accessed": "2026-04-30", "endpoint": "..."},
     "deepseek-chat":                  {"accessed": "2026-04-30", "version_string": "..."},
     "meta-llama/Llama-3.3-70B-Instruct": {"hf_revision": "<commit_sha>"},
     "google/gemma-2-27b-it":          {"hf_revision": "<commit_sha>"}
   }
   ```
   Fetch `hf_revision` via `huggingface_hub.HfApi().model_info(...).sha`.

2. **Persist raw API responses** for every call to `raw_responses/{model}/{prompt_id}_seed{seed}.json`. Do not just store parsed JSON; store the full HTTP response body.

3. **Compute SHA256** of every input dataset CSV at load time, write to `provenance.json`.

4. **Generate `requirements.txt`** with exact versions matching the pinned `pip install` commands in Section 2.

5. **Add a Zenodo placeholder** in README: `"Code and data archived at: https://doi.org/10.5281/zenodo.<TBD>"`. Create the archive after acceptance.

6. **Determinism check**: at end of Phase 0 dry run, run the same prompt through LLaMA twice with seed=42 and assert byte-identical output.

---

## A.11 Limitations Section (Mandatory) [P0]

Generate `limitations.md` with these exact bullet points (TACL reviewers check for these):

```
- Lexicon-based salience detection misses subtle and implicit identity references
  (mitigated partially by the embedding-based ablation in Appendix C; future work).

- LLM judges inherit their own biases. We mitigate via 3-judge ensemble and
  Krippendorff's alpha reporting (mean alpha = X across dimensions). Where
  alpha < 0.6, results are flagged.

- Three languages (English, Hindi, Bengali) are not representative of the full
  multilingual landscape. Generalization to Dravidian, Sino-Tibetan, or African
  languages is not claimed.

- Closed API models (Gemini, DeepSeek) are not perfectly reproducible due to
  silent backend updates. We pin snapshot dates and store raw responses.

- The construct of "identity term" is operationalized via lexicons whose
  English origin may not transfer cleanly to Hindi or Bengali. We document
  per-language lexicon coverage in Table X.

- Sample size is constrained by API rate limits and compute budget. Bootstrap
  CIs reflect this.

- We use a single annotated lexicon per category-language pair without inter-
  lexicon comparison. Robustness across alternative lexicon constructions is
  noted as future work.

- Multi-turn conversation prompts are author-designed, not user-elicited.
  Naturalistic conversation data may produce different accumulation patterns.
```

---

## A.12 Cuts to Stay Within TACL Page Budget [P1]

TACL allows ~12 pages of main content (refs unlimited). Cut these from the original spec:

1. **Drop Study 4** (causal LM likelihood ratio). Redundant with Study 1 — same examples, near-identical signal. Saves a phase.
2. **Merge Study 3 (HONEST hurtful)** into Study 5 as a sub-analysis (hurtful flag is one annotated dimension via LLM-judge ensemble).
3. **Combine Studies 1+2** into a single "Traditional Bias Benchmark Baselines" section in the paper (still run both in code).

Final paper structure:
- Section: Baselines (Studies 1+2 combined)
- Section: Single-turn salience (Study 5, with hurtful sub-analysis from former Study 3)
- Section: Multi-turn accumulation (Study 6)
- Section: Cross-lingual analysis (CLSR + fertility decomposition)
- Section: Robustness (controls, paraphrases, seeds, embedding ablation)

---

## A.13 Updated Phase Pipeline [P0]

Replace Phase 4–14 with this sequence in `main.py`:

```
Phase 0  : Dry run (1 example per model)
Phase 1  : Environment + .env (4 models only)
Phase 2  : Load datasets (CrowS-Pairs, Indian-BhED) + sample 400/lang
Phase 2.5: Build negative control prompts (50/lang)
Phase 3  : Neutralize prompts
Phase 3.5: Generate 3 paraphrases per neutral prompt (Gemini)
Phase 4  : Compute tokenizer fertility per (model, language)
Phase 5  : Studies 1+2 (MLM/CLM baselines) on full sample
Phase 6  : Study 5 single-turn salience (4 models × 3 paraphrases × 3 seeds)
Phase 7  : Study 6 multi-turn (60-prompt subset × 3 seeds)
Phase 8  : Negative control runs (Study 5 metrics on control prompts)
Phase 9  : LLM-judge ensemble for hurtful + VAR dimensions only
Phase 10 : Bootstrap CIs + BH correction + Cohen's d + mixed-effects regression
Phase 11 : Tokenizer-fertility-controlled cross-lingual analysis
Phase 12 : Error analysis (auto-extract 30 high-GSS, 30 low-GSS, 20 judge-disagreement)
Phase 13 : 5 figures + main results table + paraphrase robustness table
Phase 14 : Pre-registration test summary + limitations + reproducibility package
Phase 15 : Final paper skeleton (paper_skeleton.md with Sec headers + result placeholders)
```

Each phase is checkpointed in `progress.json` exactly as before.

---

## A.14 Updated Budget [P0]

Revise the README budget block:

| Resource | Old | New (TACL plan) |
|---|---|---|
| Models | 6 | 4 |
| API providers | Gemini, DeepSeek, Together, Mistral | Gemini, DeepSeek |
| Compute | 42 h H200 | ~60 h H200 (extra for paraphrases × seeds) |
| API cost | ~$50 | ~$15 (Gemini free tier + DeepSeek + extra judging calls) |
| Sample / language | 400 | 400 (same; reinvest savings into seeds + paraphrases) |
| Total generative calls | ~14,400 | ~28,800 (4 models × 400 × 3 paraphrases × 3 seeds × 2 studies) |

The 2× call inflation from paraphrases and seeds is the cost of TACL acceptance. Do not skip it.

---

## A.15 Code Style Mandates [P1]

Add to `main.py` top-level docstring:

```
- Set numpy, torch, random seeds at module load (42)
- All generative calls record: prompt_id, model, language, seed, paraphrase_id,
  raw_response_path, parse_status, latency_ms, retry_count
- Every CSV write goes through utf-8-sig encoding for Indic compatibility
- Every API client wraps calls in tenacity.retry with exp backoff (max 3 retries)
- Every metric function returns (value, ci_low, ci_high, n) — never bare scalars
- Log per-phase start/end timestamps to phase_timings.json for the paper's
  "computational reproducibility" appendix
```

---

## A.16 Paper Skeleton Auto-Generation [P1]

After Phase 14, generate `paper_skeleton.md` containing:

```
# {Working title}

## Abstract (150 words)
{auto-generated from key numbers}

## 1. Introduction
- Motivation: gap in multilingual bias evaluation
- Contributions: 4 models × 3 langs × salience + multi-turn + fertility-controlled CL analysis
- Pre-registration: link to pre_registration.md

## 2. Related Work
- Multilingual bias: SeeGULL, MBBQ, INDIBIAS, Indian-BhED
- Generative bias: HONEST, BOLD, BBQ
- Multi-turn fairness: cite Tu et al. 2024
- What's new: salience as separable construct; tokenizer-fertility-controlled CL

## 3. Datasets and Models
- {auto-fill from data loaders + model_snapshots.json}

## 4. Methodology
- Neutralization, paraphrases, seeds, judge ensemble

## 5. Results
- 5.1 Baseline benchmarks (Studies 1+2)
- 5.2 Single-turn salience (Study 5) with paraphrase robustness
- 5.3 Multi-turn accumulation (Study 6)
- 5.4 Cross-lingual analysis controlling for tokenizer fertility
- 5.5 Robustness: controls, seeds, embedding ablation

## 6. Discussion
- Pre-registration test summary table
- Implications for multilingual deployment

## 7. Limitations
{paste from limitations.md}

## 8. Ethics Statement
{datasets, lexicon transparency, no human subjects}

## References
{auto-collect from citations registry}
```

This becomes your overleaf import on day one of paper writing.

---

## A.17 Critical "Do Not" List for Copilot [P0]

Tell Copilot explicitly:

```
- Do NOT add any user-input prompts or input() calls. Pipeline must be fully non-interactive.
- Do NOT skip a phase silently if its check fails. Halt with a clear error.
- Do NOT use try/except to swallow API errors without logging the full response.
- Do NOT mix per-seed and pooled results in the same row of any CSV.
- Do NOT report bare p-values without effect sizes and CIs.
- Do NOT compute GSS using lowercase substring matching. Use word-level tokenization
  with the language's appropriate tokenizer (NLTK for English, indic-tokenize for Hindi/Bengali).
- Do NOT include API keys in any error message or log file.
- Do NOT exceed 12 pages of main paper content; cut figures over 5.
```

---

## END OF APPENDIX A

If Copilot asks "should I do X?" and X is in P0 above — do it. If Copilot asks about anything not covered here — defer to the original `FULL_PROMPT.md`.
