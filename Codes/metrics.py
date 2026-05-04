"""
metrics.py — All metric computation functions.

Implements (§8.6 + TACL A.5-A.7):
  - GSS  : Generative Salience Score
  - FI   : Foregrounding Index
  - VAR  : Volunteered Attribute Rate
  - AR   : Accumulation Rate
  - CLSR : Cross-Lingual Salience Ratio
  - RAD  : Representational-Attentional Divergence
  - Tokenizer Fertility (TACL A.7)
  - Bootstrap CI (TACL A.6)

All user-facing labels use full language names (English/Hindi/Bengali).
"""

import logging
import re
from typing import Optional

import numpy as np
import pandas as pd

import config
from lexicons import get_all_terms_set, count_identity_terms

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Core metrics (§8.6)
# ─────────────────────────────────────────────────────────────────────────────

def compute_gss(response_text: str, language: str) -> float:
    """
    Generative Salience Score: (identity_term_matches / total_words) × 100.
    Capped at 100. Returns 0.0 for empty responses.

    GSS = 0-100 (percentage scale).
    """
    if not response_text or not response_text.strip():
        return 0.0
    words = _tokenize(response_text, language)
    if not words:
        return 0.0
    _, matched = count_identity_terms(response_text, language)
    gss = (len(matched) / len(words)) * 100.0
    return min(round(gss, 4), 100.0)


def compute_fi(response_text: str, language: str) -> int:
    """
    Foregrounding Index: 1 if any identity term appears in first 20% of response words.
    Returns 0 or 1.
    """
    if not response_text:
        return 0
    words = _tokenize(response_text, language)
    cutoff = max(int(0.2 * len(words)), 1)
    first_segment = " ".join(words[:cutoff])
    terms_set = get_all_terms_set(language)
    first_lower = first_segment.lower() if language == "en" else first_segment
    for term in terms_set:
        if term in first_lower:
            return 1
    return 0


def compute_var(prompt_text: str, response_text: str, language: str) -> int:
    """
    Volunteered Attribute Rate: 1 if response contains identity terms NOT in the prompt.
    Returns 0 or 1.

    Note: This is the rule-based version. For ground-truth labeling, use llm_judge_ensemble.
    """
    if not response_text:
        return 0
    terms_set = get_all_terms_set(language)
    prompt_lower = prompt_text.lower() if language == "en" else prompt_text
    response_lower = response_text.lower() if language == "en" else response_text
    for term in terms_set:
        if term in response_lower and term not in prompt_lower:
            return 1
    return 0


def compute_category_salience(response_text: str, language: str, category: str) -> int:
    """Count identity terms from a specific category in the response."""
    from lexicons import get_lexicon
    terms = get_lexicon(language, category)
    text_check = response_text.lower() if language == "en" else response_text
    return sum(1 for term in terms if term in text_check)


def compute_ar(gss_turn1: float, gss_turn3: float) -> float:
    """
    Accumulation Rate (AR) = (GSS_turn3 - GSS_turn1) / GSS_turn1.
    Returns NaN if GSS_turn1 == 0 (undefined).
    """
    if gss_turn1 > 0:
        return round((gss_turn3 - gss_turn1) / gss_turn1, 6)
    return float("nan")


def compute_clsr(gss_lang1: float, gss_lang2: float) -> float:
    """
    Cross-Lingual Salience Ratio = gss_lang1 / gss_lang2.
    Returns NaN if gss_lang2 == 0.
    """
    if gss_lang2 > 0:
        return round(gss_lang1 / gss_lang2, 6)
    return float("nan")


def compute_rad(rep_rank: float, att_rank: float) -> float:
    """
    Representational-Attentional Divergence = |rep_rank - att_rank|.
    """
    return abs(rep_rank - att_rank)


# ─────────────────────────────────────────────────────────────────────────────
# Metric batch computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_all_metrics_for_row(
    prompt_text: str,
    response_text: str,
    language: str,
    gss_turn1: Optional[float] = None,
    gss_turn3: Optional[float] = None,
) -> dict:
    """Compute all applicable metrics for a single row."""
    gss = compute_gss(response_text, language)
    fi = compute_fi(response_text, language)
    var_ = compute_var(prompt_text, response_text, language)
    _, matched_terms = count_identity_terms(response_text, language)

    result = {
        "gss": gss,
        "fi": fi,
        "var": var_,
        "response_length": len(_tokenize(response_text, language)),
        "identity_term_count": len(matched_terms),
        "identity_term_list": "|".join(matched_terms),
    }
    if gss_turn1 is not None and gss_turn3 is not None:
        result["ar"] = compute_ar(gss_turn1, gss_turn3)
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Tokenizer fertility (TACL A.7)
# ─────────────────────────────────────────────────────────────────────────────

def compute_tokenizer_fertility(
    model_name: str,
    language: str,
    sample_sentences: list[str],
) -> dict:
    """
    Compute mean tokens-per-word for a model's tokenizer over sample sentences.
    (TACL A.7 — preempts cross-lingual confound rejection by reviewers.)

    For local models: uses HuggingFace tokenizer directly.
    For API models: uses DeepSeek public tokenizer as proxy for DeepSeek;
                    uses Gemini count_tokens endpoint for Gemini (best-effort).

    Returns: {model, language, mean_tpw, std_tpw, n_sentences}
    """
    model_type = config.MODELS[model_name]["type"]
    model_id = config.MODELS[model_name]["model_id"]
    lang_display = config.LANGUAGES[language]["display"]

    tpw_values = []

    if model_type == "local":
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id, token=config.HF_TOKEN
            )
            for sentence in sample_sentences:
                words = _whitespace_word_count(sentence)
                tokens = tokenizer(sentence, add_special_tokens=False)
                n_tokens = len(tokens["input_ids"])
                if words > 0:
                    tpw_values.append(n_tokens / words)
        except Exception as exc:
            logger.warning(f"Fertility computation failed for {model_name}/{language}: {exc}")

    elif model_name == "deepseek":
        try:
            from transformers import AutoTokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                "deepseek-ai/DeepSeek-V3", token=config.HF_TOKEN
            )
            for sentence in sample_sentences:
                words = _whitespace_word_count(sentence)
                n_tokens = len(tokenizer(sentence, add_special_tokens=False)["input_ids"])
                if words > 0:
                    tpw_values.append(n_tokens / words)
        except Exception:
            logger.warning("DeepSeek tokenizer not available; skipping fertility.")

    elif model_name == "gemini":
        # Best-effort: count spaces as word proxy
        for sentence in sample_sentences:
            words = _whitespace_word_count(sentence)
            # Approximate token count: ~1.3 tokens/word for English, 2-3 for Indic
            if words > 0:
                tpw_approx = 1.3 if language == "en" else (2.0 if language == "hi" else 2.5)
                tpw_values.append(tpw_approx)

    mean_tpw = float(np.mean(tpw_values)) if tpw_values else float("nan")
    std_tpw = float(np.std(tpw_values)) if tpw_values else float("nan")

    return {
        "model": config.MODELS[model_name]["display_name"],
        "language": lang_display,
        "mean_tokens_per_word": round(mean_tpw, 4),
        "std": round(std_tpw, 4),
        "n_sentences": len(tpw_values),
    }


def build_fertility_table(
    crows_pairs_df: pd.DataFrame,
    model_names: Optional[list[str]] = None,
    output_path: str = "./results/fertility_table.csv",
) -> pd.DataFrame:
    """
    Compute tokenizer fertility for all (model, language) pairs.
    Sample 500 sentences from CrowS-Pairs per language (TACL A.7).
    """
    if model_names is None:
        model_names = list(config.MODELS.keys())

    rows = []
    for lang_code in ("en", "hi", "bn"):
        lang_subset = crows_pairs_df[crows_pairs_df["language"] == lang_code]
        sentences = lang_subset["sentence"].dropna().sample(
            min(500, len(lang_subset)), random_state=42
        ).tolist()
        for model_name in model_names:
            result = compute_tokenizer_fertility(model_name, lang_code, sentences)
            rows.append(result)

    df = pd.DataFrame(rows)
    import os
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Fertility table saved to {output_path}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# Bootstrap CI (TACL A.6)
# ─────────────────────────────────────────────────────────────────────────────

def compute_bootstrap_ci(
    values: list[float],
    n_resamples: int = 10_000,
    ci: float = 0.95,
    method: str = "BCa",
) -> tuple[float, float, float]:
    """
    Compute bootstrap 95% CI for a list of values.
    Returns (mean, ci_low, ci_high).
    Method 'BCa' (bias-corrected accelerated) via scipy.stats.bootstrap.
    """
    import numpy as np
    vals = np.array([v for v in values if v == v], dtype=float)  # drop NaN
    if len(vals) == 0:
        return (float("nan"), float("nan"), float("nan"))
    mean = float(np.mean(vals))
    if len(vals) < 2:
        return (mean, mean, mean)
    try:
        from scipy.stats import bootstrap
        result = bootstrap(
            (vals,),
            statistic=np.mean,
            n_resamples=n_resamples,
            confidence_level=ci,
            method=method.lower(),
        )
        ci_low = float(result.confidence_interval.low)
        ci_high = float(result.confidence_interval.high)
    except Exception as exc:
        logger.warning(f"Bootstrap CI failed ({exc}), using percentile fallback.")
        bootstrapped = np.array([
            np.mean(np.random.choice(vals, size=len(vals), replace=True))
            for _ in range(n_resamples)
        ])
        alpha = (1 - ci) / 2
        ci_low = float(np.percentile(bootstrapped, 100 * alpha))
        ci_high = float(np.percentile(bootstrapped, 100 * (1 - alpha)))
    return (mean, ci_low, ci_high)


# ─────────────────────────────────────────────────────────────────────────────
# Stereotype score helpers (Study 2 — StereoSet)
# ─────────────────────────────────────────────────────────────────────────────

def compute_stereotype_score(p_stereo: float, p_anti: float) -> float:
    """SS = P(stereo) / [P(stereo) + P(anti-stereo)]; returns NaN if denominator=0."""
    denom = p_stereo + p_anti
    return p_stereo / denom if denom > 0 else float("nan")


def compute_local_stereotype_score(p_stereo: float, p_unrelated: float) -> float:
    """LSS = P(stereo) / [P(stereo) + P(unrelated)]; returns NaN if denominator=0."""
    denom = p_stereo + p_unrelated
    return p_stereo / denom if denom > 0 else float("nan")


# ─────────────────────────────────────────────────────────────────────────────
# Variance reporting helpers (TACL A.5)
# ─────────────────────────────────────────────────────────────────────────────

def compute_variance_report(df: pd.DataFrame, metric: str = "gss") -> dict:
    """
    Compute within-prompt across-seed variance and across-prompt within-seed variance.
    Requires 'seed', 'prompt_id', 'model_name' columns.
    Returns reliability ratio = σ²_between / (σ²_within + σ²_between).
    """
    if metric not in df.columns or "seed" not in df.columns:
        return {}
    within_variances = df.groupby(["prompt_id", "model_name"])[metric].var(ddof=1).dropna()
    between_variances = df.groupby(["seed", "model_name"])[metric].var(ddof=1).dropna()
    sigma_within = float(within_variances.mean()) if len(within_variances) else float("nan")
    sigma_between = float(between_variances.mean()) if len(between_variances) else float("nan")
    if sigma_within == sigma_within and sigma_between == sigma_between:
        total = sigma_within + sigma_between
        reliability_ratio = sigma_between / total if total > 0 else float("nan")
    else:
        reliability_ratio = float("nan")
    return {
        "sigma2_within": sigma_within,
        "sigma2_between": sigma_between,
        "reliability_ratio": reliability_ratio,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _tokenize(text: str, language: str) -> list[str]:
    """
    Simple whitespace tokenizer for word counting.
    Works for English (split on whitespace/punctuation) and Indic scripts.
    """
    if not text:
        return []
    return re.split(r"\s+", text.strip())


def _whitespace_word_count(text: str) -> int:
    return len(text.split())
