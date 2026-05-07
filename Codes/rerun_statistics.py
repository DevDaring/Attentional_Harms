"""
rerun_statistics.py — Re-run Phase 12 (statistical analysis) only.
Loads Study 3 CSVs already on disk, no GPU or models required.
Run from Codes/ directory:
    python rerun_statistics.py
"""
import glob
import logging
import os
import sys

import pandas as pd

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
)
logger = logging.getLogger("rerun_statistics")

RESULTS_DIR = "./results"
STUDY3_PATTERN = os.path.join(RESULTS_DIR, "study3_gss_*.csv")


def main():
    os.chdir(os.path.dirname(os.path.abspath(__file__)))

    files = sorted(glob.glob(STUDY3_PATTERN))
    if not files:
        logger.error(f"No Study 3 CSV files found at {STUDY3_PATTERN}")
        sys.exit(1)

    logger.info(f"Found {len(files)} Study 3 CSVs: {[os.path.basename(f) for f in files]}")

    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            # Infer model name from filename if column missing
            if "model_name" not in df.columns:
                model = os.path.basename(f).replace("study3_gss_", "").replace(".csv", "")
                df["model_name"] = model
            dfs.append(df)
            logger.info(f"  Loaded {os.path.basename(f)}: {len(df)} rows, columns: {list(df.columns)}")
        except Exception as e:
            logger.warning(f"  Could not load {f}: {e}")

    if not dfs:
        logger.error("No data loaded — aborting.")
        sys.exit(1)

    combined = pd.concat(dfs, ignore_index=True)
    logger.info(f"Combined Study 3: {len(combined)} rows, {combined['model_name'].nunique()} models")
    logger.info(f"  Columns: {list(combined.columns)}")
    logger.info(f"  Models: {sorted(combined['model_name'].unique())}")
    if "category" in combined.columns:
        logger.info(f"  Categories: {sorted(combined['category'].dropna().unique())}")

    # Run all statistics
    from statistical_analysis import run_all_statistics
    results = run_all_statistics(combined)

    # Report what was produced
    logger.info("\n=== Results Summary ===")
    for key, val in results.items():
        if val is None:
            logger.warning(f"  {key}: FAILED / None")
        elif hasattr(val, "__len__"):
            logger.info(f"  {key}: OK ({len(val)} rows/items)")
        else:
            logger.info(f"  {key}: OK")

    stats_dir = os.path.join(RESULTS_DIR, "statistics")
    logger.info(f"\nAll outputs in: {stats_dir}")
    for f in sorted(os.listdir(stats_dir)):
        path = os.path.join(stats_dir, f)
        size = os.path.getsize(path)
        logger.info(f"  {f} ({size:,} bytes)")

    logger.info("\nDone.")


if __name__ == "__main__":
    main()
