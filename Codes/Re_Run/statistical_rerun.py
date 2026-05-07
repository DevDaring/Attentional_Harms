"""
statistical_rerun.py — Extended statistical analysis for the 7-model Re_Run.

Extends statistical_analysis.py with:
  1. 7 models → 21 pairwise comparisons (C(7,2)=21 vs. original 15)
  2. Drift trajectory analysis: Kruskal-Wallis H + Pearson r(depth, GSS) per model
     Confirms monotonic accumulation with test statistics suitable for paper
  3. gen_sar_mean and gen_ss_mean added to bootstrap CI and correlation matrix
     (convergent validity: do GenSAR/GenSS correlate with GSS across models?)
  4. n_a, n_b columns in pairwise table (power transparency)
  5. All outputs overwrite files in results/statistics/

Usage (from Codes/ directory):
  python Re_Run/statistical_rerun.py   # standalone dry-run
"""

import logging
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import scipy.stats as stats

import config
from statistical_analysis import (
    bootstrap_ci_for_all_metrics,
    run_mixed_effects_regression,
    run_anova,
    pairwise_model_comparisons,
    compute_correlation_matrix,
)
from studies import compute_cross_category_analysis

logger = logging.getLogger(__name__)


# ── Drift trajectory analysis ─────────────────────────────────────────────────

def run_drift_trajectory_anova(
    all_study4_df: pd.DataFrame,
    output_dir: str,
) -> pd.DataFrame:
    """
    Per-model drift trajectory statistics.

    For each model with gss_d0/d2/d4/d6 columns:
      - Kruskal-Wallis H test across 4 depth groups
        (non-parametric equivalent of one-way ANOVA; robust to GSS non-normality)
      - Pearson r between depth value (0/2/4/6) and per-row GSS
        (tests linearity of accumulation)
      - Mean GSS at each depth (for Figure 6 trajectory plot)

    Outputs: results/statistics/drift_trajectory.csv
    """
    depth_cols = ["gss_d0", "gss_d2", "gss_d4", "gss_d6"]
    depth_vals = [0, 2, 4, 6]

    missing = [c for c in depth_cols if c not in all_study4_df.columns]
    if missing:
        logger.warning(
            f"Drift trajectory ANOVA skipped: columns {missing} not in Study4 DataFrame. "
            "Run study4_expanded.py first."
        )
        return pd.DataFrame()

    rows = []
    for model in sorted(all_study4_df["model_name"].dropna().unique()):
        sub = all_study4_df[all_study4_df["model_name"] == model]

        # Per-depth distributions
        gss_groups = [sub[c].dropna().values for c in depth_cols]
        means      = [
            float(np.nanmean(g)) if len(g) > 0 else float("nan")
            for g in gss_groups
        ]

        # Kruskal-Wallis across 4 depth groups
        valid_groups = [g for g in gss_groups if len(g) >= 2]
        if len(valid_groups) >= 2:
            try:
                h_stat, kw_p = stats.kruskal(*valid_groups)
            except Exception:
                h_stat, kw_p = float("nan"), float("nan")
        else:
            h_stat, kw_p = float("nan"), float("nan")

        # Pearson r(depth_value, gss) — stacks all (depth, gss) pairs from all items
        depths_rep = np.concatenate(
            [np.full(len(g), d) for d, g in zip(depth_vals, gss_groups)]
        )
        gss_rep = np.concatenate(gss_groups)
        if len(gss_rep) >= 4:
            try:
                r_val, r_p = stats.pearsonr(depths_rep, gss_rep)
            except Exception:
                r_val, r_p = float("nan"), float("nan")
        else:
            r_val, r_p = float("nan"), float("nan")

        rows.append({
            "model_name":      model,
            "gss_d0_mean":     means[0],
            "gss_d2_mean":     means[1],
            "gss_d4_mean":     means[2],
            "gss_d6_mean":     means[3],
            "pearson_r_depth": float(r_val),
            "pearson_p_depth": float(r_p),
            "kruskal_H":       float(h_stat),
            "kruskal_p":       float(kw_p),
            "n_items":         int(len(sub)),
            "significant":     bool(kw_p < 0.05) if not np.isnan(kw_p) else False,
        })

    df_traj = pd.DataFrame(rows)
    if len(df_traj) > 0:
        out_path = os.path.join(output_dir, "drift_trajectory.csv")
        df_traj.to_csv(out_path, index=False, encoding="utf-8-sig")
        sig_models = df_traj[df_traj["significant"]]["model_name"].tolist()
        logger.info(
            f"Drift trajectory: {len(df_traj)} models  "
            f"significant={sig_models}  → {out_path}"
        )
    return df_traj


# ── Convergent validity: GenSAR/GenSS vs. GSS correlation ────────────────────

def compute_convergent_validity(
    study3_all_df: pd.DataFrame,
    study1_all_df: pd.DataFrame,
    study2_all_df: pd.DataFrame,
    output_dir: str,
) -> pd.DataFrame:
    """
    Compute Pearson r between model-level mean GenSAR and mean GSS,
    and between model-level mean GenSS and mean GSS.

    This is the key convergent validity test: if GSS correlates with
    established bias metrics (GenSAR, GenSS) across models, it validates
    the attentional salience construct.
    """
    # Aggregate to model level
    gss_by_model = (
        study3_all_df.groupby("model_name")["gss_mean"]
        .mean()
        .reset_index()
        .rename(columns={"gss_mean": "mean_gss"})
    )

    rows = []
    for metric_name, source_df, col in [
        ("GenSAR", study1_all_df, "gen_sar"),
        ("GenSS",  study2_all_df, "gen_ss"),
    ]:
        if source_df is None or len(source_df) == 0 or col not in source_df.columns:
            continue
        metric_by_model = (
            source_df.groupby("model_name")[col]
            .mean()
            .reset_index()
            .rename(columns={col: f"mean_{metric_name.lower()}"})
        )
        merged = gss_by_model.merge(metric_by_model, on="model_name", how="inner")
        if len(merged) < 3:
            continue
        x = merged["mean_gss"].values
        y = merged[f"mean_{metric_name.lower()}"].values
        try:
            r, p = stats.pearsonr(x, y)
            sr, sp = stats.spearmanr(x, y)
        except Exception:
            r = p = sr = sp = float("nan")
        rows.append({
            "comparison":  f"GSS vs {metric_name}",
            "pearson_r":   float(r),
            "pearson_p":   float(p),
            "spearman_r":  float(sr),
            "spearman_p":  float(sp),
            "n_models":    len(merged),
        })

    df_cv = pd.DataFrame(rows)
    if len(df_cv) > 0:
        out_path = os.path.join(output_dir, "convergent_validity.csv")
        df_cv.to_csv(out_path, index=False, encoding="utf-8-sig")
        logger.info(f"Convergent validity → {out_path}")
    return df_cv


# ── Main stats orchestrator ───────────────────────────────────────────────────

def run_all_stats_rerun(
    study3_all_df: pd.DataFrame,
    study4_all_df: pd.DataFrame,
    study1_all_df: pd.DataFrame = None,
    study2_all_df: pd.DataFrame = None,
    output_dir: str = None,
) -> dict:
    """
    Full statistical pipeline for the 7-model re-run.

    Runs all original tests (bootstrap CI, mixed-effects regression, ANOVA,
    pairwise comparisons, correlation matrix, cross-category analysis) plus:
      - Drift trajectory ANOVA (Study 4)
      - Convergent validity (GenSAR/GenSS vs GSS)

    All outputs overwrite files in results/statistics/.
    """
    if output_dir is None:
        output_dir = config.STATS_DIR
    os.makedirs(output_dir, exist_ok=True)

    results: dict = {}
    n_models = study3_all_df["model_name"].nunique() if len(study3_all_df) > 0 else 0
    logger.info(
        f"Stats rerun: {len(study3_all_df)} Study3 rows, "
        f"{n_models} models, {len(study4_all_df)} Study4 rows"
    )

    # 1. Bootstrap CIs — add gen_sar_mean / gen_ss_mean if available
    metric_cols = ["gss_mean", "fi_mean"]
    for extra_col in ("gen_sar_mean", "gen_ss_mean"):
        if extra_col in study3_all_df.columns:
            metric_cols.append(extra_col)

    ci_df = bootstrap_ci_for_all_metrics(study3_all_df, metric_cols=metric_cols)
    ci_df.to_csv(
        os.path.join(output_dir, "bootstrap_ci.csv"),
        index=False, encoding="utf-8-sig",
    )
    results["bootstrap_ci"] = ci_df
    logger.info(f"Bootstrap CI: {len(ci_df)} rows")

    # 2. Mixed-effects regression
    reg = run_mixed_effects_regression(study3_all_df)
    if reg is not None:
        try:
            with open(os.path.join(output_dir, "mixed_effects_summary.txt"), "w") as f:
                f.write(reg.summary().as_text())
        except Exception as exc:
            logger.warning(f"Could not write mixed-effects summary: {exc}")
    results["mixed_effects"] = reg

    # 3. Two-way ANOVA (model × category) — now 7 models
    anova_table = run_anova(study3_all_df)
    if anova_table is not None:
        anova_table.to_csv(
            os.path.join(output_dir, "anova_table.csv"),
            encoding="utf-8-sig",
        )
    results["anova"] = anova_table

    # 4. Pairwise comparisons — 21 pairs for 7 models
    pairs_df = pairwise_model_comparisons(study3_all_df)
    # Annotate with sample sizes for transparency / power reporting in paper
    n_per_model = (
        study3_all_df.groupby("model_name")["gss_mean"]
        .count()
        .to_dict()
    )
    pairs_df["n_a"] = pairs_df["model_a"].map(n_per_model)
    pairs_df["n_b"] = pairs_df["model_b"].map(n_per_model)
    pairs_df.to_csv(
        os.path.join(output_dir, "pairwise_comparisons.csv"),
        index=False, encoding="utf-8-sig",
    )
    results["pairwise"] = pairs_df
    logger.info(f"Pairwise comparisons: {len(pairs_df)} pairs (expected 21 for 7 models)")

    # 5. Drift trajectory ANOVA (Study 4)
    if len(study4_all_df) > 0:
        traj_df = run_drift_trajectory_anova(study4_all_df, output_dir)
        results["drift_trajectory"] = traj_df

    # 6. Convergent validity (GenSAR/GenSS vs GSS)
    if study1_all_df is not None or study2_all_df is not None:
        cv_df = compute_convergent_validity(
            study3_all_df,
            study1_all_df if study1_all_df is not None else pd.DataFrame(),
            study2_all_df if study2_all_df is not None else pd.DataFrame(),
            output_dir,
        )
        results["convergent_validity"] = cv_df

    # 7. Correlation matrix — extended to include gen_sar/gen_ss columns
    num_cols = [
        c for c in (
            "gss_mean", "fi_mean",
            "gss_seed42", "gss_seed123", "gss_seed7",
            "gen_sar_mean", "gen_ss_mean",
        )
        if c in study3_all_df.columns
    ]
    if num_cols:
        corr = compute_correlation_matrix(
            study3_all_df,
            metrics=num_cols,
            output_path=os.path.join(output_dir, "correlation_matrix.csv"),
        )
        results["correlation_matrix"] = corr

    # 8. Cross-category pivot (models × bias categories)
    cross = compute_cross_category_analysis(
        study3_all_df,
        output_path=os.path.join(output_dir, "cross_category_analysis.csv"),
    )
    results["cross_category"] = cross

    logger.info(f"All statistics (7-model re-run) written to {output_dir}")
    return results


if __name__ == "__main__":
    # Quick standalone test — prints available Study3 CSV files
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s — %(message)s")
    import glob
    csvs = glob.glob(os.path.join(config.RESULTS_DIR, "study3_gss_*.csv"))
    print(f"Found {len(csvs)} Study3 CSVs: {[os.path.basename(c) for c in csvs]}")
