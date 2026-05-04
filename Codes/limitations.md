# Limitations and Ethical Considerations
## Attentional Salience as a Complementary Bias Lens in Multilingual Language Models
**Target Venue:** TACL (Transactions of the Association for Computational Linguistics)

---

## Limitations (TACL A.11)

### 1. Lexicon Coverage and Language Specificity
Our bias lexicons for Hindi and Bengali were compiled from existing NLP resources and domain-expert validation, but are not exhaustive. Identity terminology varies significantly across dialects, registers, and regional contexts. Terms that carry strong social salience in colloquial Hindi or Bengali may be absent from our formal lexicon, leading to underestimation of GSS in those languages. Future work should extend these lexicons through community-participatory methods.

### 2. Quantization Effects on Local Model Behavior
Both LLaMA 3.3 70B Instruct and Gemma 2 27B IT are deployed with 4-bit NF4 quantization (BitsAndBytesConfig). While NF4 is the state-of-the-art quantization scheme and double quantization further reduces deviation, quantized models may exhibit slightly different bias profiles from their full-precision counterparts. The representational-attentional divergence (RAD) comparison with API models should be interpreted with this confound in mind.

### 3. API Model Version Drift
Gemini 2.5 Flash and DeepSeek V3 are accessed via commercial APIs. Model weights may be updated silently by providers between our data collection date and any subsequent reproduction attempt. We record model snapshot identifiers per TACL A.10.1, but bit-exact reproducibility cannot be guaranteed for API models. Local model SHA-256 hashes are fully reproducible.

### 4. Tokenizer Fertility as a Proxy, Not a Cause
We measure tokenizer fertility (tokens per character) as a proxy for potential model under-training on a language. High fertility is correlated with morphological complexity and sub-word over-segmentation for Indic scripts, but fertility does not causally explain higher GSS. Our regression analysis establishes correlation only; the causal mechanism (training data imbalance, NLLB tokenizer design, etc.) requires further investigation.

### 5. Scope of Identity Categories
We study four bias categories: caste, gender, religion, and race. Caste is specifically relevant to South Asian contexts and largely absent from prior multilingual bias benchmarks; gender and religion are cross-linguistically comparable; race is primarily analyzed in the English-language literature. We do not cover: age-based bias, disability, sexual orientation, socioeconomic status, or regional/tribal identity. Findings should not be generalized to these dimensions without replication.

### 6. Hurtful Completion Classification via LLM Judge
Classification of hurtful completions relies on a 3-judge LLM ensemble (TACL A.3). LLM judges are themselves biased systems. We report inter-annotator agreement (Krippendorff's alpha) and flag all cells where α < 0.6, but we acknowledge that the judges may have systematic blind spots — particularly for code-switched text or culturally nuanced references in Hindi and Bengali. Human annotation of a validation subset is recommended.

### 7. Sample Size and Representativeness
Our datasets are stratified samples (CrowS-Pairs: 180 items, Indian-BhED: 120 items, StereoSet: 300 items, HONEST: 100 items) designed to achieve sufficient statistical power (N ≥ 100 per cell at d=0.5, α=0.05, power=0.80). However, dataset items originate from curated sources and may not represent the full distribution of real-world prompts encountered by deployed models. Bias estimates may be conservative relative to naturalistic conversation contexts.

### 8. Generalizability Across Languages and Cultures
This study covers English, Hindi, and Bengali — three languages with different scripts, morphology, and cultural contexts. Despite covering major South Asian language communities, the findings may not generalize to other low-resource languages (e.g., Tamil, Urdu, Marathi, Odia). The framework (GSS/FI/VAR/AR) is language-agnostic in principle, but our lexicons are language-specific, and a new lexicon and validation study would be required for each additional language.

---

## Ethical Considerations

- All datasets used are publicly available under research licenses. No personal data is collected.
- Model outputs containing potentially harmful content are not published verbatim; only aggregate statistics and LLM-judge classifications are reported.
- Our goal is to measure and quantify bias, not to amplify it. Raw responses are stored locally with SHA-256 integrity checks (TACL A.10.2).
- This work does not attempt to "fix" bias through fine-tuning or model modification; we document and quantify bias as a prerequisite for informed mitigation.
- All research infrastructure runs under controlled lab conditions; no collected prompts are submitted to production services in a way that could affect model training.

---

## Acknowledgements of Positionality

The research team includes researchers with backgrounds in South Asian linguistics, computational linguistics, and machine learning. Lexicon construction for Hindi and Bengali involved consultation with native speakers. We acknowledge that our positionality as researchers may influence which biases we foreground and which we underweight.

---

*This limitations section will appear in full in the TACL submission (TACL A.11).*
