"""
regenerate_figures.py
Run this AFTER the main run completes to regenerate all figures with
the correct study1_df and study4_df data from saved CSV files.

Usage (on VM):
    cd /workspace/Attentional_Harms/Codes && python regenerate_figures.py
"""
import os
import sys
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(name)s — %(message)s")
logger = logging.getLogger("regenerate_figures")

RESULTS_DIR = "./results"
MODEL_NAMES = ["deepseek", "gpt4o_mini", "grok", "qwen", "gemma3", "llama"]


def load_combined(prefix: str, models: list, required_col: str | None = None) -> pd.DataFrame:
    dfs = []
    for m in models:
        path = os.path.join(RESULTS_DIR, f"{prefix}_{m}.csv")
        if os.path.exists(path):
            df = pd.read_csv(path, encoding="utf-8-sig")
            if required_col is None or required_col in df.columns:
                dfs.append(df)
            else:
                logger.warning(f"  {path}: missing column '{required_col}' — skipped")
        else:
            logger.warning(f"  {path}: not found — skipped")
    if dfs:
        return pd.concat(dfs, ignore_index=True)
    return pd.DataFrame()


def main():
    logger.info("Loading combined study DataFrames from CSV results …")

    combined_s1 = load_combined("study1_sar",     MODEL_NAMES, required_col="prefers_stereo")
    combined_s3 = load_combined("study3_gss",     MODEL_NAMES, required_col="gss_mean")
    combined_s4 = load_combined("study4_multiturn", MODEL_NAMES, required_col="gss_turn1")

    logger.info(f"  combined_s1: {len(combined_s1)} rows, cols={list(combined_s1.columns) if not combined_s1.empty else 'EMPTY'}")
    logger.info(f"  combined_s3: {len(combined_s3)} rows, cols={list(combined_s3.columns) if not combined_s3.empty else 'EMPTY'}")
    logger.info(f"  combined_s4: {len(combined_s4)} rows, cols={list(combined_s4.columns) if not combined_s4.empty else 'EMPTY'}")

    if combined_s3.empty:
        logger.error("combined_s3 is empty — cannot regenerate figures.")
        sys.exit(1)

    from figures import generate_all_figures
    logger.info("Regenerating all figures with correct study4_df and study1_df …")
    generate_all_figures(combined_s3, study4_df=combined_s4, study1_df=combined_s1)
    logger.info("Done. Check results/figures/")


if __name__ == "__main__":
    main()
