"""
analysis/salvage_may2026.py - exploratory reconstruction of the May 2026 Study 3
outputs after deduplication (Future_Plan.md s2). Exploratory archive only:
results are never preregistered or confirmatory.

For each model file Codes/results/study3_gss_<model>.csv:
  * keep one row per unique prompt (the four Indian-BhED labels share one pool),
  * flag Devanagari and literal mask tokens,
  * recompute the old GSS and the v2 non-pronoun density on the stored
    representative response.
"""
from __future__ import annotations

import re

import pandas as pd

import config_v2 as C
from scoring import lexical_features

MODELS = ["qwen", "gemma3", "llama", "deepseek", "gpt4o_mini", "grok", "qwen72b"]
_DEV = re.compile(r"[ऀ-ॿ]")
_MASK = re.compile(r"\bMASK\b|\[M\]", re.I)


def run(out_dir=None) -> pd.DataFrame:
    out_dir = out_dir or (C.OUTPUT_DIR / "salvage_may2026")
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    for m in MODELS:
        path = C.CODES_DIR / "results" / f"study3_gss_{m}.csv"
        if not path.exists():
            continue
        df = pd.read_csv(path, encoding="utf-8-sig", dtype=str, keep_default_na=False)
        n_rows = len(df)
        df["source"] = df["category"].map(lambda c: "honest" if c in ("female", "male") else "indian_bhed_pool")
        dedup = df.drop_duplicates("prompt_text").copy()
        dedup["has_devanagari"] = dedup["prompt_text"].str.contains(_DEV)
        dedup["has_mask"] = dedup["prompt_text"].str.contains(_MASK)
        feats = dedup["rep_response"].map(lambda t: lexical_features(t, []))
        dedup = pd.concat([dedup.reset_index(drop=True), pd.DataFrame(list(feats))], axis=1)
        dedup.to_csv(out_dir / f"study3_dedup_{m}.csv", index=False, encoding="utf-8-sig")
        clean = dedup[~dedup["has_devanagari"] & ~dedup["has_mask"]]
        for src, g in clean.groupby("source"):
            summary.append({"model": m, "source": src, "rows_original": n_rows,
                            "unique_prompts_all": len(dedup), "unique_clean": len(g),
                            "mean_gss_mean_original_col": pd.to_numeric(g["gss_mean"], errors="coerce").mean(),
                            "mean_density_non_pronoun_rep": g["density_non_pronoun"].mean(),
                            "share_any_group_label_rep": g["any_group_label"].mean()})
    out = pd.DataFrame(summary)
    out.to_csv(out_dir / "salvage_summary.csv", index=False, encoding="utf-8-sig")
    return out
