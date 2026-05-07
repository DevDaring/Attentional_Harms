"""
main.py — Orchestrator for the full attentional salience pipeline.

Target: TACL 2025
Language: English only
Models: 6 (3 API + 3 local ≤8B)
Hardware: NVIDIA H100 80GB SXM5
Estimated wall time: ~8–10 hours

Phases
  0. Verify environment + validate API keys
  1. Download / verify dataset cache (ensure_datasets_cached)
  2. Load all datasets
  3. Neutralize prompts (identity terms → "someone")
  4. Build control prompts (50 abstract English sentences)
  5. Load all 3 local models into VRAM simultaneously (~27 GB of 80 GB)
  6. Study 1 — SAR (CrowS-Pairs perplexity, all 6 models)
  7. Study 2 — SS (StereoSet perplexity, all 6 models)
  8. Study 3 — Generative Salience (GSS + FI, all 6 models, 3 seeds)
  9. Study 4 — Multi-Turn Identity Drift (IDrift, all 6 models)
 10. Study 5 — Control study (baseline GSS ≈ 0 check)
 11. LLM judge ensemble (Gemini + DeepSeek + Mistral, zero retry)
 12. Statistical analysis (ANOVA, bootstrap CI, mixed-effects, pairwise)
 13. Generate figures
 14. Cross-category analysis
 15. Build model_snapshots.json
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import Optional

import config
import checkpointing

# Ensure the default results directory exists before the module-level log file handler.
os.makedirs(config.RESULTS_DIR, exist_ok=True)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(
            os.path.join(config.RESULTS_DIR, "pipeline.log"),
            encoding="utf-8",
        ),
    ],
)
logger = logging.getLogger("main")


# ─────────────────────────────────────────────────────────────────────────────
# Phase 0 — Environment verification
# ─────────────────────────────────────────────────────────────────────────────

def verify_environment() -> bool:
    """
    Check that all required packages and API keys are present.
    Aborts on critical failures; warns on non-critical gaps.
    """
    critical_ok = True
    warnings: list[str] = []

    # Python packages
    required_pkgs = [
        ("torch",          True),
        ("transformers",   True),
        ("datasets",       True),
        ("huggingface_hub",True),
        ("pandas",         True),
        ("numpy",          True),
        ("scipy",          True),
        ("requests",       True),
        ("statsmodels",    False),   # non-critical
        ("krippendorff",   False),   # non-critical
        ("matplotlib",     False),   # needed for figures
        ("seaborn",        False),
    ]
    for pkg, critical in required_pkgs:
        try:
            __import__(pkg)
        except ImportError:
            if critical:
                logger.error(f"CRITICAL package missing: {pkg}")
                critical_ok = False
            else:
                warnings.append(f"Optional package missing: {pkg}")

    # API keys
    key_status = config.validate_keys()
    for key_name, present in key_status.items():
        if key_name in ("GEMINI_KEYS",) and not present:
            warnings.append(f"No {key_name} — LLM Judge A (Gemini) will fail")
        elif key_name in ("DEEPSEEK_KEYS",) and not present:
            logger.error(f"CRITICAL: No {key_name} — generation model unavailable")
            critical_ok = False
        elif key_name in ("OPENROUTER_KEYS",) and not present:
            logger.error(f"CRITICAL: No {key_name} — OpenRouter models (GPT-4o Mini, Grok) unavailable")
            critical_ok = False
        elif key_name == "MISTRAL_KEYS" and not present:
            warnings.append("No MISTRAL_KEYS — LLM Judge C (Mistral) will fail")

    if not config.HF_TOKEN:
        warnings.append("HF_TOKEN not set — gated models (Llama) may fail to download")

    # GPU check
    try:
        import torch
        if not torch.cuda.is_available():
            warnings.append("CUDA not available — local models will run on CPU (very slow)")
        else:
            vram_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
            if vram_gb < 20:
                warnings.append(f"GPU has only {vram_gb:.0f} GB VRAM. ≥27 GB needed for 3 local models simultaneously.")
            else:
                logger.info(f"GPU: {torch.cuda.get_device_name(0)}  {vram_gb:.0f} GB VRAM")
    except Exception as exc:
        warnings.append(f"GPU check failed: {exc}")

    for w in warnings:
        logger.warning(w)

    if not critical_ok:
        logger.error("Critical environment issues found. Aborting.")
    return critical_ok


# ─────────────────────────────────────────────────────────────────────────────
# Phase 1 — Dataset cache
# ─────────────────────────────────────────────────────────────────────────────

def phase_dataset_cache():
    from data_pipeline import ensure_datasets_cached
    logger.info("=== Phase 1: Dataset cache ===")
    paths = ensure_datasets_cached()
    missing = [k for k, p in paths.items() if not p.exists()]
    if missing:
        logger.warning(f"Missing datasets after download: {missing}")
    else:
        logger.info("All dataset CSVs present.")
    return paths


# ─────────────────────────────────────────────────────────────────────────────
# Phase 2 — Load datasets
# ─────────────────────────────────────────────────────────────────────────────

def phase_load_datasets():
    from data_pipeline import (
        load_dataset_crows_pairs, load_dataset_stereoset,
        load_dataset_honest, load_dataset_indian_bias,
    )
    logger.info("=== Phase 2: Load datasets ===")
    # Use config sample sizes so that --dry-run overrides propagate here.
    crows    = load_dataset_crows_pairs(sample_n=config.SAMPLE_CROWS_PAIRS)
    stereo   = load_dataset_stereoset(sample_n=config.SAMPLE_STEREOSET)
    honest   = load_dataset_honest(sample_n=config.SAMPLE_HONEST)
    indian   = load_dataset_indian_bias(
        sample_per_category=max(1, config.SAMPLE_INDIAN_BIAS // 4)
    )
    logger.info(
        f"CrowS-Pairs: {len(crows)}  StereoSet: {len(stereo)}  "
        f"HONEST: {len(honest)}  Indian-BhED: {len(indian)}"
    )
    return crows, stereo, honest, indian


# ─────────────────────────────────────────────────────────────────────────────
# Phase 3 — Neutralize prompts
# ─────────────────────────────────────────────────────────────────────────────

def phase_neutralize(crows_df, indian_df):
    from data_pipeline import neutralize_prompts
    logger.info("=== Phase 3: Neutralize prompts ===")
    sent_more_neutralized = neutralize_prompts(crows_df["sent_more"].tolist())
    sent_less_neutralized = neutralize_prompts(crows_df["sent_less"].tolist())
    logger.info(f"  Neutralized {len(sent_more_neutralized) + len(sent_less_neutralized)} CrowS-Pairs sentences")

    # Build Indian-BhED prompt list
    # Use case-insensitive column lookup (dataset has 'Sentence' with capital S)
    _col_lower_map = {c.lower(): c for c in indian_df.columns}
    prompt_col = None
    for search in ("sentence", "prompt", "text", "sent", "stereotype"):
        if search in _col_lower_map:
            prompt_col = _col_lower_map[search]
            break
    indian_prompts = []
    if prompt_col:
        for _, row in indian_df.iterrows():
            indian_prompts.append({
                "item_id":    f"indian_{row.name}",
                "prompt_text": str(row[prompt_col]),
                "category":   str(row.get("category", "unknown")),
            })
    else:
        print(f"  WARNING: No prompt column found in Indian-BhED. Columns: {list(indian_df.columns)}")
    logger.info(f"  Built {len(indian_prompts)} Indian-BhED prompts")
    return sent_more_neutralized, sent_less_neutralized, indian_prompts


# ─────────────────────────────────────────────────────────────────────────────
# Phase 4 — Build control prompts
# ─────────────────────────────────────────────────────────────────────────────

def phase_build_controls():
    from data_pipeline import build_control_prompts
    logger.info("=== Phase 4: Build control prompts ===")
    controls = build_control_prompts(n=config.N_CONTROL_PROMPTS)
    logger.info(f"  {len(controls)} control prompts ready")
    return controls


# ─────────────────────────────────────────────────────────────────────────────
# Phase 5 — Load local models
# ─────────────────────────────────────────────────────────────────────────────

def phase_load_local_models():
    from models import load_all_local_models
    logger.info("=== Phase 5: Load local models (qwen, gemma3, llama) ===")
    model_objects = load_all_local_models()
    logger.info(f"  Loaded: {list(model_objects.keys())}")
    return model_objects


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _get_model_args(model_name: str, model_objects: dict) -> dict:
    """Return model_obj and tokenizer_obj kwargs for call_model / studies."""
    if config.MODELS[model_name]["type"] == "local":
        if model_name not in model_objects:
            raise RuntimeError(
                f"Local model '{model_name}' failed to load — cannot run studies. "
                "Check GPU memory and HuggingFace token."
            )
        model, tok = model_objects[model_name]
        return {"model_obj": model, "tokenizer_obj": tok}
    return {}


def _make_generative_prompts(honest_df, indian_prompts) -> list[dict]:
    """Build the Study 3 prompt list from HONEST + Indian-BhED items."""
    prompts = []
    for idx, row in honest_df.iterrows():
        template = row.get("template_masked", row.get("template", ""))
        if isinstance(template, str) and "[MASK]" in template:
            text = template.replace("[MASK]", "someone")
        elif isinstance(template, str) and "[M]" in template:
            # HONEST dataset uses [M] as mask token
            text = template.replace("[M]", "someone")
        else:
            text = str(template)
        prompts.append({
            "item_id":     f"honest_{idx}",
            "prompt_text": text,
            "category":    str(row.get("category", "gender")),
        })
    prompts.extend(indian_prompts)
    return prompts


def _make_multiturn_prompts(prompts: list[dict], n: int = config.SAMPLE_MULTI_TURN) -> list[dict]:
    """Convert flat prompts into 3-turn conversations."""
    import random
    rng = random.Random(config.RANDOM_SEED)
    selected = rng.sample(prompts, min(n, len(prompts)))
    mt = []
    for item in selected:
        mt.append({
            "item_id":  f"mt_{item['item_id']}",
            "category": item.get("category", "unknown"),
            "turns": [
                {"role": "user",      "content": "Let's talk about everyday life."},
                {"role": "assistant", "content": "Sure! What would you like to discuss?"},
                {"role": "user",      "content": item["prompt_text"]},
            ],
        })
    return mt


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Attentional Salience Pipeline")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with tiny sample sizes (2 per dataset) to verify the pipeline end-to-end.",
    )
    args = parser.parse_args()

    if args.dry_run:
        # Override config sample sizes to minimal values for dry-run validation.
        config.SAMPLE_CROWS_PAIRS  = 2
        config.SAMPLE_STEREOSET    = 2
        config.SAMPLE_HONEST       = 2
        config.SAMPLE_INDIAN_BIAS  = 4   # 1 per category × 4
        config.SAMPLE_MULTI_TURN   = 2
        config.N_CONTROL_PROMPTS   = 2
        # Write dry-run outputs to a separate directory so they don't
        # pollute the full-run checkpoints / results.
        config.RESULTS_DIR = config.DRY_RUN_DIR
        config.FIGURES_DIR = os.path.join(config.DRY_RUN_DIR, "figures")
        config.STATS_DIR   = os.path.join(config.DRY_RUN_DIR, "statistics")
        config.RAW_DIR     = os.path.join(config.DRY_RUN_DIR, "raw_responses")

    # Re-configure the log file now that RESULTS_DIR is finalised.
    results_dir = config.RESULTS_DIR

    t_start = time.time()
    os.makedirs(results_dir,           exist_ok=True)
    os.makedirs(config.FIGURES_DIR,    exist_ok=True)
    os.makedirs(config.STATS_DIR,      exist_ok=True)
    os.makedirs(config.RAW_DIR,        exist_ok=True)
    os.makedirs(config.DATA_CACHE_DIR, exist_ok=True)

    logger.info("=" * 72)
    if args.dry_run:
        logger.info("DRY-RUN MODE — sample_n=2 per dataset, all 6 models, all phases")
    logger.info(f"Pipeline started: {datetime.now().isoformat()}")
    logger.info(f"Models: {list(config.MODELS.keys())}")
    logger.info(f"Language: {config.LANGUAGE_DISPLAY}")
    logger.info("=" * 72)

    # Phase 0
    if not verify_environment():
        sys.exit(1)

    # Phase 1 — ensure dataset CSVs exist
    phase_dataset_cache()

    # Phase 2 — load into memory (uses overridden config.SAMPLE_* if --dry-run)
    crows_df, stereo_df, honest_df, indian_df = phase_load_datasets()

    # Phase 3 — neutralize
    _, _, indian_prompts = phase_neutralize(crows_df, indian_df)

    # Phase 4 — controls
    control_prompts = phase_build_controls()

    # Phase 5 — local models
    model_objects = phase_load_local_models()

    # Build generative + multi-turn prompt lists
    generative_prompts = _make_generative_prompts(honest_df, indian_prompts)
    multiturn_prompts  = _make_multiturn_prompts(generative_prompts)

    model_names = list(config.MODELS.keys())  # 6 models

    all_study1_dfs = []
    all_study3_dfs = []
    all_study4_dfs = []

    for model_name in model_names:
        logger.info(f"--- Processing model: {model_name} ---")
        kwargs = _get_model_args(model_name, model_objects)

        # Phase 6 — Study 1 SAR
        from studies import run_study_crows_pairs
        s1 = run_study_crows_pairs(model_name, checkpoint_dir=results_dir, **kwargs)
        if s1 is not None:
            s1["model_name"] = model_name
            all_study1_dfs.append(s1)

        # Phase 7 — Study 2 SS
        from studies import run_study_stereoset
        run_study_stereoset(model_name, checkpoint_dir=results_dir, **kwargs)

        # Phase 8 — Study 3 GSS
        from studies import run_study_generative_salience
        s3 = run_study_generative_salience(
            model_name, generative_prompts, checkpoint_dir=results_dir, **kwargs
        )
        s3["model_name"] = model_name
        all_study3_dfs.append(s3)

        # Phase 9 — Study 4 multi-turn
        from studies import run_study_multi_turn
        s4 = run_study_multi_turn(
            model_name, multiturn_prompts, checkpoint_dir=results_dir, **kwargs
        )
        if s4 is not None:
            s4["model_name"] = model_name
            all_study4_dfs.append(s4)

        # Phase 10 — Study 5 control
        from studies import run_control_study
        run_control_study(model_name, control_prompts, checkpoint_dir=results_dir, **kwargs)

    # Phase 11 — LLM judge ensemble on Study 3 responses
    import pandas as pd
    combined_s1 = pd.concat(all_study1_dfs, ignore_index=True) if all_study1_dfs else pd.DataFrame()
    combined_s3 = pd.concat(all_study3_dfs, ignore_index=True) if all_study3_dfs else pd.DataFrame()
    combined_s4 = pd.concat(all_study4_dfs, ignore_index=True) if all_study4_dfs else pd.DataFrame()

    if not combined_s3.empty and "rep_response" in combined_s3.columns:
        logger.info("=== Phase 11: LLM judge ensemble ===")
        judge_items = [
            {
                "item_id":        row["item_id"],
                "item_text":      row["rep_response"],
                "dimension":      row.get("category", "unknown"),
                "prompt_template": "hurtful",
            }
            for _, row in combined_s3.iterrows()
            if pd.notna(row.get("rep_response")) and row.get("rep_response", "")
        ]
        from llm_judge import batch_judge_and_report
        judge_df = batch_judge_and_report(
            judge_items,
            output_csv=os.path.join(config.STATS_DIR, "judge_agreement.csv"),
        )
        # Merge judge labels back into combined_s3
        judge_map = dict(zip(judge_df["item_id"], judge_df["final_label"]))
        combined_s3["hurtful_label"] = combined_s3["item_id"].map(judge_map).fillna(-1).astype(int)

    # Phase 12 — Statistical analysis
    if not combined_s3.empty:
        logger.info("=== Phase 12: Statistical analysis ===")
        from statistical_analysis import run_all_statistics
        run_all_statistics(combined_s3)

    # Phase 13 — Figures
    try:
        logger.info("=== Phase 13: Figures ===")
        from figures import generate_all_figures
        generate_all_figures(combined_s3, study4_df=combined_s4, study1_df=combined_s1)
    except Exception as exc:
        logger.warning(f"Figure generation failed (non-fatal): {exc}")

    # Phase 14 — Cross-category analysis
    if not combined_s3.empty:
        logger.info("=== Phase 14: Cross-category analysis ===")
        from studies import compute_cross_category_analysis
        compute_cross_category_analysis(combined_s3)

    # Phase 15 — Model snapshots
    logger.info("=== Phase 15: Model snapshots ===")
    try:
        from models import build_model_snapshots
        snapshots = build_model_snapshots()
        snap_path = os.path.join(config.RESULTS_DIR, "model_snapshots.json")
        with open(snap_path, "w", encoding="utf-8") as f:
            json.dump(snapshots, f, indent=2, ensure_ascii=False)
        logger.info(f"  model_snapshots.json → {snap_path}")
    except Exception as exc:
        logger.warning(f"Snapshot generation failed (non-fatal): {exc}")

    # Final summary
    elapsed = time.time() - t_start
    logger.info("=" * 72)
    logger.info(f"Pipeline complete in {elapsed / 3600:.2f} hours")
    logger.info(f"Completed: {datetime.now().isoformat()}")
    logger.info("=" * 72)


if __name__ == "__main__":
    main()

