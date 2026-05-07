"""
main_rerun.py — Safe parallel orchestrator for the TACL Re_Run pipeline.

Run from the Codes/ directory:
    cd /workspace/Attentional_Harms/Codes
    python Re_Run/main_rerun.py

Phases:
  Phase 0 — Validate environment (API keys, CUDA, data files)
  Phase 1 — Load local models (qwen 1.5B, gemma3 2B, llama 8B) onto H100
  Phase 2 — Study 4 expanded (300 items, 7-turn, 4 depths)
             2A: API models in ThreadPoolExecutor(3) — IO-bound, safe
             2B: Local models sequential — CUDA not thread-safe across model objects
  Phase 3 — GenSAR/GenSS for existing API models (deepseek, gpt4o_mini, grok)
             + Qwen72B Study3 — concurrent in ThreadPoolExecutor(2)
  Phase 4 — Qwen72B Study4 + GenSAR/GenSS concurrent; Study5 sequential after
  Phase 5 — Combine all DataFrames → run_all_stats_rerun()
  Phase 6 — Regenerate figures

Idempotency: each phase writes a marker file results/.phase_{tag}_done.
If a marker exists the phase is skipped, so restarting after a crash is safe.

Estimated runtime on H100 80GB:
  Phase 1:  ~8 min   (model loads)
  Phase 2:  ~2.5 hr  (Study 4 × 7 models × 300 items × 4 depths)
  Phase 3:  ~1.5 hr  (GenSAR/GenSS for 3 API models + Study3-72B)
  Phase 4:  ~2.5 hr  (72B is slow on OR)
  Phase 5:  ~2 min   (stats)
  Phase 6:  ~3 min   (figures)
  Total:    ~9 hours  (estimate)

Logging: results/rerun.log  (also stdout)
"""

import glob
import logging
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed, wait, FIRST_EXCEPTION

# ── Path setup ────────────────────────────────────────────────────────────────
_CODES_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, _CODES_DIR)

# ── Logging (to file + stdout) ────────────────────────────────────────────────
def _setup_logging(results_dir: str) -> None:
    os.makedirs(results_dir, exist_ok=True)
    log_path = os.path.join(results_dir, "rerun.log")
    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(log_path, mode="a", encoding="utf-8"),
    ]
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-8s %(name)s — %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
    )

logger = logging.getLogger(__name__)


# ── Imports after path setup ──────────────────────────────────────────────────
import config
from Re_Run.config_rerun import (
    API_MODELS, LOCAL_MODELS, QWEN72B_KEY, SAMPLE_MULTI_TURN,
    get_ts_openrouter_rotator, get_ts_deepseek_rotator,
)
from Re_Run.study4_expanded import run_study4_expanded
from Re_Run.study1_2_gensar import run_gensar, run_genss
from Re_Run.study_qwen72b import (
    run_qwen72b_study3, run_qwen72b_study4,
    run_qwen72b_study5, run_qwen72b_gensar, run_qwen72b_genss,
)
from Re_Run.statistical_rerun import run_all_stats_rerun

import pandas as pd


# ── Phase marker helpers ──────────────────────────────────────────────────────

def _marker_path(tag: str, results_dir: str) -> str:
    return os.path.join(results_dir, f".phase_{tag}_done")


def _phase_done(tag: str, results_dir: str) -> bool:
    return os.path.exists(_marker_path(tag, results_dir))


def _mark_done(tag: str, results_dir: str) -> None:
    with open(_marker_path(tag, results_dir), "w") as f:
        f.write(f"done\n")


# ── Phase 0: Validate environment ────────────────────────────────────────────

def phase0_validate(results_dir: str) -> None:
    if _phase_done("validate", results_dir):
        logger.info("Phase 0: skipped (already validated)")
        return

    logger.info("Phase 0: Validating environment …")
    errors = []

    # API keys
    try:
        key_status = config.validate_keys()
        failed_keys = [k for k, v in key_status.items() if not v]
        if failed_keys:
            errors.append(f"Missing API keys: {', '.join(failed_keys)}")
        else:
            logger.info("Phase 0: API keys OK")
    except Exception as exc:
        errors.append(f"API key validation: {exc}")

    # CUDA
    try:
        import torch
        if not torch.cuda.is_available():
            errors.append("CUDA not available — local model inference will fail")
        else:
            gpu = torch.cuda.get_device_name(0)
            vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
            logger.info(f"Phase 0: GPU={gpu}  VRAM={vram:.1f} GB")
    except ImportError:
        errors.append("torch not importable")

    # Data files — check the cache files directly (config has no DATASET_PATH attrs)
    import config as _cfg
    data_checks = [
        (os.path.join(_cfg.DATA_CACHE_DIR, "en_honest.csv"), "HONEST dataset"),
        (os.path.join(_cfg.DATA_CACHE_DIR, "en_indian_caste.csv"), "Indian bias dataset"),
    ]
    for path, name in data_checks:
        if not os.path.exists(path):
            errors.append(f"{name} not found at {path}")
        else:
            logger.info(f"Phase 0: {name} found at {path}")

    if errors:
        logger.error("Phase 0: Validation FAILED:")
        for e in errors:
            logger.error(f"  - {e}")
        sys.exit(1)

    _mark_done("validate", results_dir)
    logger.info("Phase 0: PASSED")


# ── Phase 1: Load local models ────────────────────────────────────────────────

def phase1_load_local_models(results_dir: str) -> dict:
    """
    Load qwen 1.5B, gemma3 2B, llama 8B onto GPU.
    Returns dict mapping model_name → (model_obj, tokenizer_obj).
    ~27 GB total; fits in H100 80 GB with ample headroom for inference.
    """
    if _phase_done("load_local", results_dir):
        logger.info("Phase 1: local models already loaded marker found (in new process, will reload)")
    # Always reload (marker just for logging; models live in memory)
    logger.info("Phase 1: Loading local models onto GPU …")
    t0 = time.time()
    from models import load_all_local_models
    model_objects = load_all_local_models()
    elapsed = time.time() - t0
    logger.info(f"Phase 1: {len(model_objects)} local models loaded in {elapsed:.0f}s")
    _mark_done("load_local", results_dir)
    return model_objects


# ── Phase 2: Study 4 expanded ─────────────────────────────────────────────────

def phase2_study4_expanded(
    generative_prompts: list,
    model_objects: dict,
    results_dir: str,
) -> dict:
    """
    Study 4 expanded for all 7 models.
    API models run concurrently (ThreadPoolExecutor(3)).
    Local models run sequentially (CUDA not thread-safe across model objects).
    Returns dict model_name → DataFrame.
    """
    if _phase_done("study4_expanded", results_dir):
        logger.info("Phase 2: Study 4 expanded already done — loading saved CSVs")
        dfs = {}
        for m in API_MODELS + LOCAL_MODELS:
            p = os.path.join(results_dir, f"study4_multiturn_{m}.csv")
            if os.path.exists(p):
                dfs[m] = pd.read_csv(p)
        return dfs

    logger.info("Phase 2A: Study 4 expanded — API models (parallel) …")
    api_to_run = [m for m in API_MODELS if m != QWEN72B_KEY]  # qwen72b done in phase 4

    dfs: dict = {}

    # 2A: API models concurrently
    def _run_api_study4(model_name):
        from Re_Run.config_rerun import MODELS as RERUN_MODELS
        return model_name, run_study4_expanded(
            model_name=model_name,
            generative_prompts=generative_prompts,
            model_obj=None,
            tokenizer_obj=None,
            results_dir=results_dir,
            model_cfg=RERUN_MODELS.get(model_name),
        )

    with ThreadPoolExecutor(max_workers=len(api_to_run)) as executor:
        futures = {
            executor.submit(_run_api_study4, m): m
            for m in api_to_run
        }
        for future in as_completed(futures):
            model_name = futures[future]
            try:
                name, df = future.result()
                dfs[name] = df
                logger.info(f"Phase 2A [{name}]: Study4 complete  n={len(df)}")
            except Exception as exc:
                logger.error(f"Phase 2A [{model_name}] FAILED: {exc}", exc_info=True)

    # 2B: Local models sequentially
    logger.info("Phase 2B: Study 4 expanded — local models (sequential) …")
    for model_name in LOCAL_MODELS:
        model_obj, tokenizer_obj = model_objects.get(model_name, (None, None))
        if model_obj is None:
            logger.warning(f"Phase 2B [{model_name}]: model not loaded — skipping")
            continue
        try:
            df = run_study4_expanded(
                model_name=model_name,
                generative_prompts=generative_prompts,
                model_obj=model_obj,
                tokenizer_obj=tokenizer_obj,
                results_dir=results_dir,
                model_cfg=None,  # local models are in config.MODELS
            )
            dfs[model_name] = df
            logger.info(f"Phase 2B [{model_name}]: Study4 complete  n={len(df)}")
        except Exception as exc:
            logger.error(f"Phase 2B [{model_name}] FAILED: {exc}", exc_info=True)

    _mark_done("study4_expanded", results_dir)
    return dfs


# ── Phase 3: GenSAR/GenSS for 3 API models + Qwen72B Study3 ──────────────────

def phase3_gensar_and_qwen72b_study3(
    generative_prompts: list,
    results_dir: str,
) -> tuple:
    """
    Runs concurrently:
      - GenSAR + GenSS for deepseek, gpt4o_mini, grok (existing API models)
      - Qwen72B Study3 (GSS generative)
    Both are IO-bound so ThreadPoolExecutor(2) is safe.
    Returns (study1_dfs, study2_dfs, study3_72b_df).
    """
    if _phase_done("gensar_and_study3_72b", results_dir):
        logger.info("Phase 3: already done — loading saved CSVs")
        s1, s2, s3_72 = {}, {}, pd.DataFrame()
        for m in ["deepseek", "gpt4o_mini", "grok"]:
            for d, prefix in ((s1, "study1_sar"), (s2, "study2_ss")):
                p = os.path.join(results_dir, f"{prefix}_{m}.csv")
                if os.path.exists(p):
                    d[m] = pd.read_csv(p)
        p72 = os.path.join(results_dir, f"study3_gss_{QWEN72B_KEY}.csv")
        if os.path.exists(p72):
            s3_72 = pd.read_csv(p72)
        return s1, s2, s3_72

    logger.info("Phase 3: GenSAR/GenSS (3 API models) + Qwen72B Study3 …")

    def _run_gensar_grok_etc():
        results_sar, results_ss = {}, {}
        for m in ["deepseek", "gpt4o_mini", "grok"]:
            logger.info(f"Phase 3: GenSAR [{m}] …")
            results_sar[m] = run_gensar(m, results_dir=results_dir)
            logger.info(f"Phase 3: GenSS [{m}] …")
            results_ss[m]  = run_genss(m, results_dir=results_dir)
        return results_sar, results_ss

    def _run_study3_72b():
        logger.info(f"Phase 3: Study3 [{QWEN72B_KEY}] …")
        return run_qwen72b_study3(generative_prompts, results_dir=results_dir)

    study1_dfs, study2_dfs = {}, {}
    study3_72b = pd.DataFrame()

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_gensar = executor.submit(_run_gensar_grok_etc)
        f_s3_72b = executor.submit(_run_study3_72b)

        done, _ = wait([f_gensar, f_s3_72b], return_when=FIRST_EXCEPTION)
        # Check for exceptions before collecting results
        for f in done:
            if f.exception():
                logger.error(f"Phase 3 worker raised: {f.exception()}", exc_info=True)

        try:
            study1_dfs, study2_dfs = f_gensar.result()
        except Exception as exc:
            logger.error(f"Phase 3 GenSAR/GenSS FAILED: {exc}", exc_info=True)
        try:
            study3_72b = f_s3_72b.result()
        except Exception as exc:
            logger.error(f"Phase 3 Study3-72B FAILED: {exc}", exc_info=True)

    _mark_done("gensar_and_study3_72b", results_dir)
    return study1_dfs, study2_dfs, study3_72b


# ── Phase 4: Qwen72B all remaining studies ────────────────────────────────────

def phase4_qwen72b(
    generative_prompts: list,
    control_prompts: list,
    results_dir: str,
) -> tuple:
    """
    Qwen72B Study4 expanded + GenSAR/GenSS concurrent (IO-bound).
    Qwen72B Study5 runs sequentially after (shares OR rate limit headroom).
    Returns (study1_72b_df, study2_72b_df, study4_72b_df, study5_72b_df).
    """
    if _phase_done("qwen72b_all", results_dir):
        logger.info("Phase 4: already done — loading saved CSVs")
        dfs = {}
        for fname in (
            f"study1_sar_{QWEN72B_KEY}", f"study2_ss_{QWEN72B_KEY}",
            f"study4_multiturn_{QWEN72B_KEY}", f"study5_control_{QWEN72B_KEY}",
        ):
            p = os.path.join(results_dir, f"{fname}.csv")
            if os.path.exists(p):
                dfs[fname] = pd.read_csv(p)
        return (
            dfs.get(f"study1_sar_{QWEN72B_KEY}", pd.DataFrame()),
            dfs.get(f"study2_ss_{QWEN72B_KEY}", pd.DataFrame()),
            dfs.get(f"study4_multiturn_{QWEN72B_KEY}", pd.DataFrame()),
            dfs.get(f"study5_control_{QWEN72B_KEY}", pd.DataFrame()),
        )

    logger.info("Phase 4: Qwen72B Study4 + GenSAR/GenSS concurrent …")

    s1_72b = s2_72b = s4_72b = s5_72b = pd.DataFrame()

    def _run_72b_study4():
        logger.info(f"Phase 4: Study4-exp [{QWEN72B_KEY}] …")
        return run_qwen72b_study4(generative_prompts, results_dir=results_dir)

    def _run_72b_gensar_genss():
        logger.info(f"Phase 4: GenSAR [{QWEN72B_KEY}] …")
        s1 = run_qwen72b_gensar(results_dir=results_dir)
        logger.info(f"Phase 4: GenSS [{QWEN72B_KEY}] …")
        s2 = run_qwen72b_genss(results_dir=results_dir)
        return s1, s2

    with ThreadPoolExecutor(max_workers=2) as executor:
        f_s4  = executor.submit(_run_72b_study4)
        f_gen = executor.submit(_run_72b_gensar_genss)

        done, _ = wait([f_s4, f_gen], return_when=FIRST_EXCEPTION)
        for f in done:
            if f.exception():
                logger.error(f"Phase 4 worker raised: {f.exception()}", exc_info=True)

        try:
            s4_72b = f_s4.result()
        except Exception as exc:
            logger.error(f"Phase 4 Study4-72B FAILED: {exc}", exc_info=True)
        try:
            s1_72b, s2_72b = f_gen.result()
        except Exception as exc:
            logger.error(f"Phase 4 GenSAR/GenSS-72B FAILED: {exc}", exc_info=True)

    # Study 5 sequential (avoids rate-limit pile-up with Study4 + GenSAR)
    logger.info(f"Phase 4: Study5 [{QWEN72B_KEY}] (sequential) …")
    try:
        s5_72b = run_qwen72b_study5(control_prompts, results_dir=results_dir)
    except Exception as exc:
        logger.error(f"Phase 4 Study5-72B FAILED: {exc}", exc_info=True)

    _mark_done("qwen72b_all", results_dir)
    return s1_72b, s2_72b, s4_72b, s5_72b


# ── Phase 5: Statistics ───────────────────────────────────────────────────────

def phase5_statistics(results_dir: str) -> None:
    if _phase_done("statistics", results_dir):
        logger.info("Phase 5: statistics already done — skipping")
        return

    logger.info("Phase 5: Running extended statistical analysis …")

    # Load all Study3 CSVs (7 models)
    study3_csvs = glob.glob(os.path.join(results_dir, "study3_gss_*.csv"))
    study3_all  = pd.concat(
        [pd.read_csv(p) for p in study3_csvs], ignore_index=True
    ) if study3_csvs else pd.DataFrame()

    # Load all Study4 expanded CSVs (7 models)
    study4_csvs = glob.glob(os.path.join(results_dir, "study4_multiturn_*.csv"))
    study4_all  = pd.concat(
        [pd.read_csv(p) for p in study4_csvs], ignore_index=True
    ) if study4_csvs else pd.DataFrame()

    # Load Study1 (SAR) CSVs for GenSAR metric
    study1_csvs = glob.glob(os.path.join(results_dir, "study1_sar_*.csv"))
    study1_all  = pd.concat(
        [pd.read_csv(p) for p in study1_csvs], ignore_index=True
    ) if study1_csvs else None

    # Load Study2 (SS) CSVs for GenSS metric
    study2_csvs = glob.glob(os.path.join(results_dir, "study2_ss_*.csv"))
    study2_all  = pd.concat(
        [pd.read_csv(p) for p in study2_csvs], ignore_index=True
    ) if study2_csvs else None

    run_all_stats_rerun(
        study3_all_df=study3_all,
        study4_all_df=study4_all,
        study1_all_df=study1_all,
        study2_all_df=study2_all,
        output_dir=config.STATS_DIR,
    )

    _mark_done("statistics", results_dir)
    logger.info("Phase 5: statistics done")


# ── Phase 6: Figure regeneration ─────────────────────────────────────────────

def phase6_figures(results_dir: str) -> None:
    if _phase_done("figures", results_dir):
        logger.info("Phase 6: figures already done — skipping")
        return

    logger.info("Phase 6: Regenerating figures …")
    try:
        import glob
        from figures import generate_all_figures

        ALL_MODELS = ["deepseek", "gpt4o_mini", "grok", "qwen", "gemma3", "llama", "qwen72b"]

        def _load_combined(prefix, required_col=None):
            dfs = []
            for m in ALL_MODELS:
                p = os.path.join(results_dir, f"{prefix}_{m}.csv")
                if os.path.exists(p):
                    df = pd.read_csv(p, encoding="utf-8-sig")
                    if required_col is None or required_col in df.columns:
                        dfs.append(df)
            return pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()

        study3_df = _load_combined("study3_gss",      required_col="gss_mean")
        study4_df = _load_combined("study4_multiturn", required_col="gss_turn1")
        study1_df = _load_combined("study1_sar",       required_col="prefers_stereo")

        if study3_df.empty:
            logger.warning("Phase 6: study3 CSVs not found — figures skipped")
        else:
            generate_all_figures(study3_df, study4_df=study4_df, study1_df=study1_df)
            logger.info("Phase 6: figures done")
    except Exception as exc:
        logger.warning(f"Phase 6: figure generation failed: {exc}", exc_info=True)

    _mark_done("figures", results_dir)


# ── Main ──────────────────────────────────────────────────────────────────────

def main() -> None:
    import argparse
    parser = argparse.ArgumentParser(description="TACL Re_Run pipeline")
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Override sample sizes to 2 for a quick end-to-end validation run",
    )
    args = parser.parse_args()

    if args.dry_run:
        import Re_Run.config_rerun as _cr
        import config as _cfg
        _cr.SAMPLE_MULTI_TURN = 2
        _cfg.SAMPLE_CROWS_PAIRS  = 2
        _cfg.SAMPLE_STEREOSET    = 2
        _cfg.SAMPLE_HONEST       = 2
        _cfg.SAMPLE_INDIAN_BIAS  = 2  # total; 1 per category but fine
        _cfg.N_CONTROL_PROMPTS   = 2

    results_dir = config.RESULTS_DIR
    _setup_logging(results_dir)

    if args.dry_run:
        logger.info("DRY-RUN mode: all sample sizes overridden to 2")

    logger.info("=" * 60)
    logger.info("TACL Re_Run pipeline starting")
    logger.info("=" * 60)

    t_start = time.time()

    # Phase 0: validate
    phase0_validate(results_dir)

    # Load datasets (needed by all compute phases)
    logger.info("Loading datasets …")
    from data_pipeline import (
        load_dataset_honest,
        load_dataset_indian_bias,
        build_control_prompts,
    )
    from main import _make_generative_prompts

    honest_df  = load_dataset_honest()
    indian_df  = load_dataset_indian_bias()

    # Convert indian_df → indian_prompts list (same logic as phase_neutralize in main.py)
    _col_lower = {c.lower(): c for c in indian_df.columns}
    _prompt_col = None
    for _search in ("sentence", "prompt", "text", "sent", "stereotype"):
        if _search in _col_lower:
            _prompt_col = _col_lower[_search]
            break
    indian_prompts = []
    if _prompt_col:
        for _, _row in indian_df.iterrows():
            indian_prompts.append({
                "item_id":     f"indian_{_row.name}",
                "prompt_text": str(_row[_prompt_col]),
                "category":    str(_row.get("category", "unknown")),
            })
    else:
        logger.warning(f"Indian-BhED: no prompt column found in {list(indian_df.columns)}")

    gen_prompts     = _make_generative_prompts(honest_df, indian_prompts)
    control_prompts = build_control_prompts(n=config.N_CONTROL_PROMPTS)
    logger.info(
        f"Datasets loaded: {len(gen_prompts)} generative prompts, "
        f"{len(control_prompts)} control prompts"
    )

    # Phase 1: local models
    model_objects = phase1_load_local_models(results_dir)

    # Phase 2: Study4 expanded (all 7 models except qwen72b which is in phase 4)
    study4_dfs = phase2_study4_expanded(gen_prompts, model_objects, results_dir)

    # Phase 3: GenSAR/GenSS (3 existing API models) + Qwen72B Study3
    study1_dfs, study2_dfs, study3_72b = phase3_gensar_and_qwen72b_study3(
        gen_prompts, results_dir
    )

    # Phase 4: Qwen72B remaining studies
    s1_72b, s2_72b, s4_72b, s5_72b = phase4_qwen72b(
        gen_prompts, control_prompts, results_dir
    )

    # Phase 5: Statistics
    phase5_statistics(results_dir)

    # Phase 6: Figures
    phase6_figures(results_dir)

    elapsed = time.time() - t_start
    logger.info("=" * 60)
    logger.info(f"Re_Run pipeline COMPLETE  elapsed={elapsed/3600:.2f}h")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
