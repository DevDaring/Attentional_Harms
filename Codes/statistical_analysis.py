"""
statistical_analysis.py — All inferential statistics for the TACL paper.

English-only experiment; grouping variables are model_name and category
(not language). 6 models: deepseek, gpt4o_mini, grok, qwen, gemma3, llama.

Tests performed:
  1. Descriptive stats + bootstrap 95% CI per model × category
  2. Mixed-effects regression: GSS ~ model + category + model:category + tokenizer_fertility
  3. Krippendorff's alpha for LLM judge agreement (computed in llm_judge.py)
  4. Two-way ANOVA: factors = model_name × category
  5. Pearson r between tokenizer fertility and GSS across models
  6. Effect sizes (Cohen's d between model pairs, η² for ANOVA)
  7. Bonferroni-corrected pairwise model comparisons
"""

import logging
import os
from typing import Optional

import numpy as np
import pandas as pd
import scipy.stats as stats

import config

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# 1. Bootstrap confidence intervals
# ─────────────────────────────────────────────────────────────────────────────

def bootstrap_ci_for_all_metrics(
    df: pd.DataFrame,
    metric_cols: list[str],
    group_cols: Optional[list[str]] = None,
    n_boot: int = 2000,
    ci: float = 0.95,
    seed: int = config.RANDOM_SEED,
) -> pd.DataFrame:
    """
    Compute bootstrap 95% CI for each metric, grouped by model_name and category.
    """
    if group_cols is None:
        group_cols = ["model_name", "category"]
    rng = np.random.default_rng(seed)
    rows = []
    alpha = 1.0 - ci

    for keys, grp in df.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        for col in metric_cols:
            vals = grp[col].dropna().values
            if len(vals) == 0:
                rows.append({**base, "metric": col, "mean": np.nan, "ci_lo": np.nan, "ci_hi": np.nan, "n": 0})
                continue
            mu = float(vals.mean())
            boot = np.array([rng.choice(vals, size=len(vals), replace=True).mean() for _ in range(n_boot)])
            ci_lo = float(np.percentile(boot, 100 * alpha / 2))
            ci_hi = float(np.percentile(boot, 100 * (1 - alpha / 2)))
            rows.append({**base, "metric": col, "mean": mu, "ci_lo": ci_lo, "ci_hi": ci_hi, "n": len(vals)})

    return pd.DataFrame(rows)


# ─────────────────────────────────────────────────────────────────────────────
# 2. Mixed-effects regression
# ─────────────────────────────────────────────────────────────────────────────

def run_mixed_effects_regression(
    df: pd.DataFrame,
    outcome: str = "gss_mean",
) -> Optional[object]:
    """
    Mixed-effects regression:
      GSS ~ model_cat + category_cat + model_cat:category_cat + tokenizer_fertility_en

    Grouping variable: category (random intercept per category).
    Returns fitted model object (statsmodels MixedLM), or None if package unavailable.
    """
    try:
        import statsmodels.formula.api as smf
    except ImportError:
        logger.warning("statsmodels not available — skipping mixed-effects regression")
        return None

    work = df.copy()
    for col in ("model_name", "category"):
        if col not in work.columns:
            logger.warning(f"Column '{col}' missing from DataFrame for regression")
            return None
    if "tokenizer_fertility_en" not in work.columns:
        work["tokenizer_fertility_en"] = np.nan
    if outcome not in work.columns:
        logger.warning(f"Outcome column '{outcome}' not found")
        return None

    work["model_cat"]    = work["model_name"].astype("category")
    work["category_cat"] = work["category"].astype("category")
    work = work.dropna(subset=[outcome])

    formula = (
        f"{outcome} ~ model_cat + category_cat "
        f"+ model_cat:category_cat + tokenizer_fertility_en"
    )
    try:
        md = smf.mixedlm(formula, work, groups=work["category_cat"])
        result = md.fit(reml=False)
        logger.info(
            f"Mixed-effects regression: AIC={result.aic:.2f}  "
            f"Log-likelihood={result.llf:.2f}"
        )
        return result
    except Exception as exc:
        logger.error(f"Mixed-effects regression failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 3. Two-way ANOVA (model × category)
# ─────────────────────────────────────────────────────────────────────────────

def run_anova(
    df: pd.DataFrame,
    outcome: str = "gss_mean",
) -> Optional[pd.DataFrame]:
    """
    Two-way balanced ANOVA: factors = model_name × category.
    Returns ANOVA table or None.
    """
    try:
        from statsmodels.formula.api import ols
        from statsmodels.stats.anova import anova_lm
    except ImportError:
        logger.warning("statsmodels not available — skipping ANOVA")
        return None

    work = df.copy()
    if outcome not in work.columns:
        return None
    work = work.dropna(subset=[outcome])
    work["model_cat"]    = work["model_name"].astype("category")
    work["category_cat"] = work["category"].astype("category")

    formula = f"{outcome} ~ C(model_cat) + C(category_cat) + C(model_cat):C(category_cat)"
    try:
        fit = ols(formula, work).fit()
        table = anova_lm(fit, typ=2)
        # Compute η² from SS
        total_ss = table["sum_sq"].sum()
        table["eta_squared"] = table["sum_sq"] / total_ss
        logger.info(f"ANOVA completed. η² summary:\n{table['eta_squared'].to_string()}")
        return table
    except Exception as exc:
        logger.error(f"ANOVA failed: {exc}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# 4. Tokenizer fertility correlation
# ─────────────────────────────────────────────────────────────────────────────

def compute_partial_correlation_fertility(
    df: pd.DataFrame,
    fertility_col: str = "tokenizer_fertility_en",
    outcome_col: str = "gss_mean",
    control_col: str = "category",
) -> dict:
    """
    Measure how much of the cross-model GSS variance is attributable to
    tokenizer fertility differences (English only).

    Returns {pearson_r, pearson_p, spearman_r, spearman_p, n}.
    """
    work = df.dropna(subset=[fertility_col, outcome_col])
    if len(work) < 3:
        return {"pearson_r": np.nan, "pearson_p": np.nan,
                "spearman_r": np.nan, "spearman_p": np.nan, "n": len(work)}

    # Aggregate to model level (mean GSS and mean fertility per model)
    model_agg = (
        work.groupby("model_name")[[fertility_col, outcome_col]]
        .mean()
        .reset_index()
    )
    x = model_agg[fertility_col].values
    y = model_agg[outcome_col].values

    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    return {
        "pearson_r":  float(pr), "pearson_p":  float(pp),
        "spearman_r": float(sr), "spearman_p": float(sp),
        "n": len(model_agg),
    }


# ─────────────────────────────────────────────────────────────────────────────
# 5. Effect sizes & pairwise model comparisons
# ─────────────────────────────────────────────────────────────────────────────

def _cohens_d(a: np.ndarray, b: np.ndarray) -> float:
    """Cohen's d (pooled SD)."""
    n1, n2 = len(a), len(b)
    if n1 < 2 or n2 < 2:
        return float("nan")
    pooled = np.sqrt(((n1 - 1) * np.var(a, ddof=1) + (n2 - 1) * np.var(b, ddof=1)) / (n1 + n2 - 2))
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else float("nan")


def pairwise_model_comparisons(
    df: pd.DataFrame,
    outcome: str = "gss_mean",
    correction: str = "bonferroni",
) -> pd.DataFrame:
    """
    Welch's t-test for each pair of models, with Bonferroni correction.
    Returns DataFrame with columns: model_a, model_b, t, p_raw, p_corrected, cohens_d, significant.
    """
    models = sorted(df["model_name"].dropna().unique())
    pairs = [(a, b) for i, a in enumerate(models) for b in models[i + 1:]]
    rows = []
    for a, b in pairs:
        x = df[df["model_name"] == a][outcome].dropna().values
        y = df[df["model_name"] == b][outcome].dropna().values
        if len(x) < 2 or len(y) < 2:
            rows.append({"model_a": a, "model_b": b, "t": np.nan, "p_raw": np.nan,
                         "p_corrected": np.nan, "cohens_d": np.nan, "significant": False})
            continue
        t_stat, p_raw = stats.ttest_ind(x, y, equal_var=False)
        rows.append({
            "model_a": a, "model_b": b,
            "t": float(t_stat), "p_raw": float(p_raw),
            "cohens_d": _cohens_d(x, y),
            "significant": False,
        })

    df_pairs = pd.DataFrame(rows)
    if len(df_pairs) == 0:
        return df_pairs
    n_tests = len(df_pairs)
    if correction == "bonferroni":
        df_pairs["p_corrected"] = (df_pairs["p_raw"] * n_tests).clip(upper=1.0)
    else:
        df_pairs["p_corrected"] = df_pairs["p_raw"]
    df_pairs["significant"] = df_pairs["p_corrected"] < 0.05
    return df_pairs


# ─────────────────────────────────────────────────────────────────────────────
# 6. Correlation matrix (cross-metric)
# ─────────────────────────────────────────────────────────────────────────────

def compute_correlation_matrix(
    df: pd.DataFrame,
    metrics: Optional[list[str]] = None,
    output_path: str = "./results/statistics/correlation_matrix.csv",
) -> pd.DataFrame:
    """
    Pearson correlation matrix across all numeric metrics.
    """
    if metrics is None:
        metrics = [c for c in df.select_dtypes(include=[np.number]).columns
                   if c not in ("item_id",)]
    sub = df[metrics].dropna()
    corr = sub.corr(method="pearson")
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    corr.to_csv(output_path, encoding="utf-8-sig")
    return corr


# ─────────────────────────────────────────────────────────────────────────────
# 7. Save all statistics to disk
# ─────────────────────────────────────────────────────────────────────────────

def run_all_statistics(
    study3_df: pd.DataFrame,
    output_dir: str = config.STATS_DIR,
) -> dict:
    """
    Run all statistical tests on Study 3 results and save outputs.
    Returns dict of result objects.
    """
    os.makedirs(output_dir, exist_ok=True)
    results: dict = {}

    # Bootstrap CIs
    ci_df = bootstrap_ci_for_all_metrics(
        study3_df, metric_cols=["gss_mean", "fi_mean"]
    )
    ci_df.to_csv(os.path.join(output_dir, "bootstrap_ci.csv"), index=False, encoding="utf-8-sig")
    results["bootstrap_ci"] = ci_df
    logger.info(f"Bootstrap CI saved ({len(ci_df)} rows)")

    # Mixed-effects regression
    reg = run_mixed_effects_regression(study3_df)
    if reg is not None:
        try:
            with open(os.path.join(output_dir, "mixed_effects_summary.txt"), "w") as f:
                f.write(reg.summary().as_text())
        except Exception:
            pass
    results["mixed_effects"] = reg

    # ANOVA
    anova_table = run_anova(study3_df)
    if anova_table is not None:
        anova_table.to_csv(os.path.join(output_dir, "anova_table.csv"), encoding="utf-8-sig")
    results["anova"] = anova_table

    # Pairwise comparisons
    pairs_df = pairwise_model_comparisons(study3_df)
    pairs_df.to_csv(os.path.join(output_dir, "pairwise_comparisons.csv"), index=False, encoding="utf-8-sig")
    results["pairwise"] = pairs_df

    # Fertility correlation
    if "tokenizer_fertility_en" in study3_df.columns:
        fert = compute_partial_correlation_fertility(study3_df)
        pd.DataFrame([fert]).to_csv(
            os.path.join(output_dir, "fertility_correlation.csv"), index=False, encoding="utf-8-sig"
        )
        results["fertility_correlation"] = fert

    # Correlation matrix
    num_cols = [c for c in ("gss_mean", "fi_mean", "gss_seed42", "gss_seed123", "gss_seed7")
                if c in study3_df.columns]
    if num_cols:
        corr = compute_correlation_matrix(
            study3_df, metrics=num_cols,
            output_path=os.path.join(output_dir, "correlation_matrix.csv"),
        )
        results["correlation_matrix"] = corr

    logger.info(f"All statistics written to {output_dir}")
    return results

