"""
figures.py — Publication-quality figure generation for TACL paper.

"Attentional Salience as a Complementary Bias Lens in Multilingual Language Models"

5 figures:
  Figure 1: GSS heatmap (Model × Language × Category)
  Figure 2: VAR grouped bar chart (model × language)
  Figure 3: Multi-turn accumulation lines with 95% CI
  Figure 4: Cross-lingual scatter (English vs Hindi/Bengali per model)
  Figure 5: RAD scatter (representational rank vs attentional rank)

All figures: 300 DPI, saved as PDF + PNG.
Full language names on all axes ("English", "Hindi", "Bengali").
Venue: TACL (not Nature MI).
"""

import logging
import os
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # non-interactive backend for server environments
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

FIGURES_DIR = "./results/figures"
DPI = 300
PALETTE = "colorblind"

# Full language display names (§4.4)
LANG_ORDER = ["English", "Hindi", "Bengali"]
MODEL_ORDER = [
    "Gemini 2.5 Flash",
    "DeepSeek V3",
    "LLaMA 3.3 70B Instruct",
    "Gemma 2 27B IT",
]
CATEGORIES = ["caste", "gender", "religion", "race"]


def _save_figure(fig: plt.Figure, name: str) -> None:
    """Save figure as both PDF and PNG."""
    os.makedirs(FIGURES_DIR, exist_ok=True)
    for ext in ("pdf", "png"):
        path = os.path.join(FIGURES_DIR, f"{name}.{ext}")
        fig.savefig(path, dpi=DPI, bbox_inches="tight", facecolor="white")
        logger.info(f"Saved {path}")
    plt.close(fig)


def _filter_order(series: pd.Series, order: list[str]) -> list[str]:
    """Return only the elements of order that appear in series."""
    present = set(series.dropna().unique())
    return [x for x in order if x in present]


# ─────────────────────────────────────────────────────────────────────────────
# Figure 1: GSS Heatmap
# ─────────────────────────────────────────────────────────────────────────────

def figure1_gss_heatmap(study3_df: pd.DataFrame) -> None:
    """
    GSS heatmap: rows = categories, columns = (model × language).
    Three sub-heatmaps (one per language) side by side, or a single faceted heatmap.
    """
    if study3_df.empty or "gss" not in study3_df.columns:
        logger.warning("Figure 1: no data available.")
        return

    df = study3_df.copy()
    # Average over seeds
    agg = df.groupby(["model_name", "language", "category"])["gss"].mean().reset_index()

    model_order = _filter_order(agg["model_name"], MODEL_ORDER)
    lang_order = _filter_order(agg["language"], LANG_ORDER)
    cat_order = _filter_order(agg["category"], CATEGORIES)

    if not model_order or not lang_order:
        logger.warning("Figure 1: insufficient data after filtering.")
        return

    n_langs = len(lang_order)
    fig, axes = plt.subplots(1, n_langs, figsize=(6 * n_langs, max(4, len(cat_order) * 0.8 + 1)))
    if n_langs == 1:
        axes = [axes]

    for ax, lang in zip(axes, lang_order):
        lang_df = agg[agg["language"] == lang]
        pivot = lang_df.pivot_table(
            index="category", columns="model_name", values="gss", aggfunc="mean"
        )
        # Reorder
        pivot = pivot.reindex(index=[c for c in cat_order if c in pivot.index],
                              columns=[m for m in model_order if m in pivot.columns])
        sns.heatmap(
            pivot, ax=ax, annot=True, fmt=".2f", cmap="YlOrRd",
            linewidths=0.5, linecolor="white", vmin=0,
            cbar_kws={"label": "GSS (%)"},
        )
        ax.set_title(f"{lang}", fontsize=13, fontweight="bold")
        ax.set_xlabel("Model", fontsize=10)
        ax.set_ylabel("Bias Category" if ax == axes[0] else "", fontsize=10)
        ax.tick_params(axis="x", rotation=30, labelsize=8)
        ax.tick_params(axis="y", rotation=0, labelsize=9)

    fig.suptitle(
        "Figure 1: Generative Salience Score (GSS) by Model, Language, and Bias Category\n"
        "(Attentional Salience in Multilingual LLMs — TACL)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    _save_figure(fig, "figure1_gss_heatmap")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 2: VAR grouped bar chart
# ─────────────────────────────────────────────────────────────────────────────

def figure2_var_bars(study3_df: pd.DataFrame) -> None:
    """VAR rate grouped bar chart: x=model, hue=language."""
    if study3_df.empty or "var" not in study3_df.columns:
        logger.warning("Figure 2: no data available.")
        return

    df = study3_df.copy()
    agg = df.groupby(["model_name", "language"])["var"].mean().reset_index()
    agg["var_pct"] = agg["var"] * 100

    model_order = _filter_order(agg["model_name"], MODEL_ORDER)
    lang_order = _filter_order(agg["language"], LANG_ORDER)

    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(model_order))
    width = 0.25
    colors = sns.color_palette(PALETTE, len(lang_order))

    for i, (lang, color) in enumerate(zip(lang_order, colors)):
        lang_vals = [
            agg[(agg["model_name"] == m) & (agg["language"] == lang)]["var_pct"].values
            for m in model_order
        ]
        vals = [v[0] if len(v) > 0 else 0.0 for v in lang_vals]
        bars = ax.bar(x + i * width, vals, width, label=lang, color=color, edgecolor="white")
        for bar, val in zip(bars, vals):
            if val > 1:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.5,
                    f"{val:.1f}",
                    ha="center", va="bottom", fontsize=7,
                )

    ax.set_xlabel("Model", fontsize=11)
    ax.set_ylabel("Volunteered Attribute Rate (%)", fontsize=11)
    ax.set_title(
        "Figure 2: Volunteered Attribute Rate (VAR) by Model and Language\n"
        "(TACL — Attentional Salience Study)",
        fontsize=11,
    )
    ax.set_xticks(x + width * (len(lang_order) - 1) / 2)
    ax.set_xticklabels(model_order, rotation=20, ha="right", fontsize=9)
    ax.legend(title="Language", fontsize=9)
    ax.yaxis.set_major_formatter(ticker.FormatStrFormatter("%.1f%%"))
    plt.tight_layout()
    _save_figure(fig, "figure2_var_bars")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3: Multi-turn accumulation
# ─────────────────────────────────────────────────────────────────────────────

def figure3_accumulation(study4_df: pd.DataFrame) -> None:
    """
    Line plot: GSS over turns (1→3) per model, with 95% CI shading.
    One line per model, x-axis = turn number, y-axis = mean GSS.
    """
    if study4_df.empty or "gss" not in study4_df.columns:
        logger.warning("Figure 3: no multi-turn data available.")
        return

    df = study4_df.copy()
    model_order = _filter_order(df["model_name"], MODEL_ORDER)
    lang_order = _filter_order(df["language"], LANG_ORDER)

    n_langs = len(lang_order)
    fig, axes = plt.subplots(1, n_langs, figsize=(5 * n_langs, 4), sharey=True)
    if n_langs == 1:
        axes = [axes]

    colors = sns.color_palette(PALETTE, len(model_order))

    for ax, lang in zip(axes, lang_order):
        lang_df = df[df["language"] == lang]
        for model_name, color in zip(model_order, colors):
            model_df = lang_df[lang_df["model_name"] == model_name]
            turn_stats = (
                model_df.groupby("turn_number")["gss"]
                .agg(["mean", "std", "count"])
                .reset_index()
            )
            if turn_stats.empty:
                continue
            turns = turn_stats["turn_number"].values
            means = turn_stats["mean"].values
            sems = turn_stats["std"].values / np.sqrt(np.maximum(turn_stats["count"].values, 1))
            # 95% CI ≈ 1.96 × SEM
            ci = 1.96 * sems
            ax.plot(turns, means, marker="o", label=model_name, color=color, linewidth=2)
            ax.fill_between(turns, means - ci, means + ci, alpha=0.15, color=color)

        ax.set_title(f"{lang}", fontsize=11, fontweight="bold")
        ax.set_xlabel("Conversation Turn", fontsize=10)
        ax.set_ylabel("Mean GSS (%)" if ax == axes[0] else "", fontsize=10)
        ax.set_xticks([1, 2, 3])
        ax.set_xticklabels(["Turn 1", "Turn 2", "Turn 3"])
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Model", loc="upper right",
                   bbox_to_anchor=(1.02, 0.95), fontsize=8)

    fig.suptitle(
        "Figure 3: GSS Accumulation Across Conversation Turns (±95% CI)\n"
        "(Multi-Turn Attentional Salience — TACL)",
        fontsize=11,
    )
    plt.tight_layout()
    _save_figure(fig, "figure3_accumulation")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 4: Cross-lingual scatter
# ─────────────────────────────────────────────────────────────────────────────

def figure4_crosslingual_scatter(study3_df: pd.DataFrame) -> None:
    """
    Scatter: English GSS (x-axis) vs Hindi GSS / Bengali GSS (y-axis).
    Each point = one prompt_id; colored by model.
    """
    if study3_df.empty or "gss" not in study3_df.columns:
        logger.warning("Figure 4: no data available.")
        return

    df = study3_df.groupby(["prompt_id", "model_name", "language"])["gss"].mean().reset_index()
    model_order = _filter_order(df["model_name"], MODEL_ORDER)
    colors = dict(zip(model_order, sns.color_palette(PALETTE, len(model_order))))

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    comparisons = [("English", "Hindi"), ("English", "Bengali")]

    for ax, (lang_x, lang_y) in zip(axes, comparisons):
        x_df = df[df["language"] == lang_x][["prompt_id", "model_name", "gss"]].rename(columns={"gss": "gss_x"})
        y_df = df[df["language"] == lang_y][["prompt_id", "model_name", "gss"]].rename(columns={"gss": "gss_y"})
        merged = x_df.merge(y_df, on=["prompt_id", "model_name"])
        if merged.empty:
            ax.set_visible(False)
            continue

        for model_name in model_order:
            sub = merged[merged["model_name"] == model_name]
            if sub.empty:
                continue
            ax.scatter(
                sub["gss_x"], sub["gss_y"],
                alpha=0.5, s=20, color=colors.get(model_name, "gray"), label=model_name,
            )

        max_val = max(merged["gss_x"].max(), merged["gss_y"].max()) * 1.05
        ax.plot([0, max_val], [0, max_val], "k--", alpha=0.3, linewidth=1, label="y=x")
        ax.set_xlabel(f"GSS — {lang_x}", fontsize=11)
        ax.set_ylabel(f"GSS — {lang_y}", fontsize=11)
        ax.set_title(f"{lang_x} vs {lang_y}", fontsize=11, fontweight="bold")
        ax.grid(True, alpha=0.3)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, title="Model", loc="upper right",
                   bbox_to_anchor=(1.01, 0.9), fontsize=8)

    fig.suptitle(
        "Figure 4: Cross-Lingual GSS Comparison (English vs Hindi / Bengali)\n"
        "(Tokenizer-fertility confound controlled — see Section 5 — TACL)",
        fontsize=11,
    )
    plt.tight_layout()
    _save_figure(fig, "figure4_crosslingual_scatter")


# ─────────────────────────────────────────────────────────────────────────────
# Figure 5: RAD scatter
# ─────────────────────────────────────────────────────────────────────────────

def figure5_rad_scatter(
    study3_df: pd.DataFrame,
    study1_df: Optional[pd.DataFrame] = None,
) -> None:
    """
    RAD (Representational-Attentional Divergence) scatter.
    x-axis: model rank by SAR (traditional bias benchmark)
    y-axis: model rank by GSS (attentional salience)
    Each point is a model; distance from diagonal = RAD.
    """
    if study3_df.empty:
        logger.warning("Figure 5: no data.")
        return

    # Compute mean GSS rank
    gss_means = study3_df.groupby("model_name")["gss"].mean().sort_values()
    gss_ranks = {m: i + 1 for i, m in enumerate(gss_means.index)}

    # Compute SAR rank if study1 available
    if study1_df is not None and "is_stereo_preferred" in study1_df.columns:
        sar_means = study1_df.groupby("model_name")["is_stereo_preferred"].mean().sort_values()
        sar_ranks = {m: i + 1 for i, m in enumerate(sar_means.index)}
    else:
        sar_ranks = gss_ranks  # fallback

    models = list(set(gss_ranks.keys()) & set(sar_ranks.keys()))
    if not models:
        logger.warning("Figure 5: no overlapping models between Study 1 and Study 3.")
        return

    x_vals = [sar_ranks[m] for m in models]
    y_vals = [gss_ranks[m] for m in models]
    rads = [abs(gss_ranks[m] - sar_ranks[m]) for m in models]

    fig, ax = plt.subplots(figsize=(6, 6))
    colors = sns.color_palette(PALETTE, len(models))
    n = len(models)
    ax.plot([0.5, n + 0.5], [0.5, n + 0.5], "k--", alpha=0.3, linewidth=1, label="RAD=0 (agreement)")

    for x, y, rad, model_name, color in zip(x_vals, y_vals, rads, models, colors):
        ax.scatter(x, y, s=200, color=color, zorder=5)
        ax.annotate(
            f"{model_name}\n(RAD={rad:.1f})",
            xy=(x, y),
            xytext=(x + 0.1, y + 0.1),
            fontsize=8,
            arrowprops=dict(arrowstyle="-", alpha=0.5),
        )

    ax.set_xlabel("Representational Rank (by SAR — CrowS-Pairs)", fontsize=11)
    ax.set_ylabel("Attentional Rank (by GSS — Generative Salience)", fontsize=11)
    ax.set_title(
        "Figure 5: Representational-Attentional Divergence (RAD)\n"
        "(Model ranking disagreement between traditional and novel metrics — TACL)",
        fontsize=10,
    )
    ax.set_xlim(0.3, n + 0.7)
    ax.set_ylim(0.3, n + 0.7)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=8)
    plt.tight_layout()
    _save_figure(fig, "figure5_rad")


# ─────────────────────────────────────────────────────────────────────────────
# Master generator
# ─────────────────────────────────────────────────────────────────────────────

def generate_all_figures(
    study3_df: pd.DataFrame,
    study4_df: Optional[pd.DataFrame] = None,
    study1_df: Optional[pd.DataFrame] = None,
    stats_dict: Optional[dict] = None,
) -> None:
    """Generate all 5 publication-quality figures."""
    logger.info("Generating all figures …")
    if study4_df is None:
        study4_df = pd.DataFrame()
    if study1_df is None:
        study1_df = pd.DataFrame()

    figure1_gss_heatmap(study3_df)
    figure2_var_bars(study3_df)
    figure3_accumulation(study4_df)
    figure4_crosslingual_scatter(study3_df)
    figure5_rad_scatter(study3_df, study1_df)
    logger.info("All figures generated.")
