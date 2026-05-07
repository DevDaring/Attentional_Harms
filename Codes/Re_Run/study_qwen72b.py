"""
study_qwen72b.py — All empirical studies for Qwen2.5-72B-Instruct (7th model).

Addresses reviewer concern: "No models above 8B — scale effects are unknown."

Qwen2.5-72B is called via OpenRouter (qwen/qwen-2.5-72b-instruct).
Uses the shared ThreadSafeKeyRotator from config_rerun — fully compatible with
other concurrent OpenRouter callers (gpt4o_mini, grok) since the rotator is
protected by threading.Lock.

Studies run for this model:
  Study 1: GenSAR only  (no PPL available for black-box APIs)
            → creates results/study1_sar_qwen72b.csv
  Study 2: GenSS only
            → creates results/study2_ss_qwen72b.csv
  Study 3: GSS generative (400 prompts × 3 seeds)
            → creates results/study3_gss_qwen72b.csv
  Study 4: Multi-turn expanded (300 items, 7 turns, depths 0/2/4/6)
            → creates results/study4_multiturn_qwen72b.csv
  Study 5: Control / stability (200 identity-free prompts)
            → creates results/study5_control_qwen72b.csv

Parallelism: Study 3 and Study 5 use ThreadPoolExecutor(N_API_WORKERS) for
row-level concurrency (IO-bound API waits dominate).
Study 4 reuses run_study4_expanded() which is sequential per item but
can itself run concurrently with other API phases.
"""

import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

import config
import checkpointing
from lexicons import compute_gss, count_identity_terms
from models import call_openrouter_api
from studies import SYSTEM_PROMPT
from Re_Run.config_rerun import (
    QWEN72B_KEY,
    QWEN72B_MODEL_ID,
    N_API_WORKERS,
    get_ts_openrouter_rotator,
)
from Re_Run.study4_expanded import run_study4_expanded
from Re_Run.study1_2_gensar import run_gensar, run_genss

logger = logging.getLogger(__name__)


def _call_qwen72b(
    prompt: str,
    system_prompt: str,
    temperature: float = 0.7,
) -> dict:
    """
    Single call to Qwen2.5-72B via OpenRouter.
    Uses the module-level ThreadSafeKeyRotator — safe for concurrent calls.
    """
    rotator = get_ts_openrouter_rotator()
    return call_openrouter_api(
        prompt=prompt,
        system_prompt=system_prompt,
        model_id=QWEN72B_MODEL_ID,
        key_rotator=rotator,
        temperature=temperature,
    )


# ── Study 3: Generative Salience Score ───────────────────────────────────────

def run_qwen72b_study3(
    generative_prompts: list[dict],
    results_dir: str = None,
) -> pd.DataFrame:
    """
    Study 3 (GSS) for Qwen2.5-72B.
    Parallel across prompts with ThreadPoolExecutor(N_API_WORKERS).
    Checkpoint tag: study3_qwen72b
    """
    if results_dir is None:
        results_dir = config.RESULTS_DIR

    model_name = QWEN72B_KEY
    tag        = f"study3_{model_name}"
    done       = checkpointing.load_checkpoint(tag, results_dir)
    done_ids   = {r.get("item_id") for r in done}
    todo       = [p for p in generative_prompts if p["item_id"] not in done_ids]

    logger.info(
        f"Study3 [{model_name}]: {len(todo)} prompts to process "
        f"(checkpoint: {len(done_ids)} done)"
    )

    completed_rows = list(done)  # start from checkpoint data

    def _process_item(item: dict) -> dict:
        item_id     = item["item_id"]
        prompt_text = item["prompt_text"]
        category    = item.get("category", "unknown")

        seed_results = []
        for seed in config.GENERATION_SEEDS:
            try:
                result = _call_qwen72b(prompt_text, SYSTEM_PROMPT)
                response_text = result.get("text", "")
            except Exception as exc:
                logger.warning(f"Study3-72B {item_id} seed={seed}: {exc}")
                response_text = ""

            seed_results.append({
                "seed": seed,
                "gss":  compute_gss(response_text),
                "fi":   count_identity_terms(response_text).get("total", 0),
            })

        avg_gss = float(np.mean([r["gss"] for r in seed_results])) if seed_results else float("nan")
        avg_fi  = float(np.mean([r["fi"]  for r in seed_results])) if seed_results else float("nan")

        return {
            "item_id":      item_id,
            "model_name":   model_name,
            "prompt_text":  prompt_text,
            "category":     category,
            "gss_mean":     avg_gss,
            "fi_mean":      avg_fi,
            "gss_seed42":   seed_results[0]["gss"] if len(seed_results) > 0 else float("nan"),
            "gss_seed123":  seed_results[1]["gss"] if len(seed_results) > 1 else float("nan"),
            "gss_seed7":    seed_results[2]["gss"] if len(seed_results) > 2 else float("nan"),
            "rep_response": "",
        }

    done_count = 0
    with ThreadPoolExecutor(max_workers=N_API_WORKERS) as executor:
        futures = {
            executor.submit(_process_item, item): item["item_id"]
            for item in todo
        }
        for future in as_completed(futures):
            try:
                record = future.result()
                completed_rows.append(record)
                checkpointing.append_checkpoint(tag, record, results_dir)
            except Exception as exc:
                logger.warning(f"Study3 [{model_name}] future error: {exc}")
            done_count += 1
            if done_count % 50 == 0:
                logger.info(f"Study3 [{model_name}]: {done_count}/{len(todo)} done")

    result_df = pd.DataFrame(completed_rows)
    out_path  = os.path.join(results_dir, f"study3_gss_{model_name}.csv")
    os.makedirs(results_dir, exist_ok=True)
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(
        f"Study3 [{model_name}]: mean_gss={result_df['gss_mean'].mean():.4f}  → {out_path}"
    )
    return result_df


# ── Study 4: Multi-Turn Expanded ─────────────────────────────────────────────

def run_qwen72b_study4(
    generative_prompts: list[dict],
    results_dir: str = None,
) -> pd.DataFrame:
    """
    Study 4 expanded (7-turn, 300 items, 4 depth measurements) for Qwen2.5-72B.
    Delegates to run_study4_expanded() which handles checkpointing.
    """
    if results_dir is None:
        results_dir = config.RESULTS_DIR
    from Re_Run.config_rerun import MODELS as RERUN_MODELS
    return run_study4_expanded(
        model_name=QWEN72B_KEY,
        generative_prompts=generative_prompts,
        model_obj=None,
        tokenizer_obj=None,
        results_dir=results_dir,
        model_cfg=RERUN_MODELS.get(QWEN72B_KEY),
    )


# ── Study 5: Control / Stability ─────────────────────────────────────────────

def run_qwen72b_study5(
    control_prompts: list[str],
    results_dir: str = None,
) -> pd.DataFrame:
    """
    Study 5 control for Qwen2.5-72B.
    Parallel across prompts with ThreadPoolExecutor(N_API_WORKERS).
    Checkpoint tag: study5_control_qwen72b
    """
    if results_dir is None:
        results_dir = config.RESULTS_DIR

    model_name = QWEN72B_KEY
    tag        = f"study5_control_{model_name}"
    done       = checkpointing.load_checkpoint(tag, results_dir)
    done_ids   = {r.get("item_id") for r in done}
    todo       = [
        (f"ctrl_{i}", p)
        for i, p in enumerate(control_prompts)
        if f"ctrl_{i}" not in done_ids
    ]

    logger.info(
        f"Study5 [{model_name}]: {len(todo)} prompts to process "
        f"(checkpoint: {len(done_ids)} done)"
    )

    completed_rows = list(done)

    def _process(id_and_prompt):
        item_id, prompt_text = id_and_prompt
        try:
            result = _call_qwen72b(prompt_text, SYSTEM_PROMPT)
            resp   = result.get("text", "")
        except Exception as exc:
            logger.warning(f"Study5 [{model_name}] {item_id}: {exc}")
            resp = ""
        return {
            "item_id":     item_id,
            "model_name":  model_name,
            "prompt_text": prompt_text,
            "response":    resp,
            "gss":         compute_gss(resp),
            "fi":          count_identity_terms(resp).get("total", 0),
        }

    with ThreadPoolExecutor(max_workers=N_API_WORKERS) as executor:
        futures = {
            executor.submit(_process, ip): ip[0]
            for ip in todo
        }
        for future in as_completed(futures):
            try:
                record = future.result()
                completed_rows.append(record)
                checkpointing.append_checkpoint(tag, record, results_dir)
            except Exception as exc:
                logger.warning(f"Study5 [{model_name}] future error: {exc}")

    result_df = pd.DataFrame(completed_rows)
    out_path  = os.path.join(results_dir, f"study5_control_{model_name}.csv")
    os.makedirs(results_dir, exist_ok=True)
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(
        f"Study5 [{model_name}]: mean_gss={result_df['gss'].mean():.4f}  → {out_path}"
    )
    return result_df


# ── Study 1 & 2: GenSAR / GenSS for Qwen72B ─────────────────────────────────

def run_qwen72b_gensar(results_dir: str = None) -> pd.DataFrame:
    """
    Creates study1_sar_qwen72b.csv from scratch (ppl_more/less = -1 as for
    all API models) then runs GenSAR preference scoring.
    """
    if results_dir is None:
        results_dir = config.RESULTS_DIR

    csv_path = os.path.join(results_dir, f"study1_sar_{QWEN72B_KEY}.csv")
    if not os.path.exists(csv_path):
        from data_pipeline import load_dataset_crows_pairs
        df_raw = load_dataset_crows_pairs(sample_n=config.SAMPLE_CROWS_PAIRS)
        rows = []
        for idx, row in df_raw.iterrows():
            rows.append({
                "item_id":        f"crows_{idx}",
                "model_name":     QWEN72B_KEY,
                "sent_more":      row["sent_more"],
                "sent_less":      row["sent_less"],
                "bias_type":      row.get("bias_type", ""),
                "ppl_more":       -1.0,
                "ppl_less":       -1.0,
                "prefers_stereo": -1,
                "gen_sar":        float("nan"),
            })
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(
            f"GenSAR [{QWEN72B_KEY}]: created baseline CSV ({len(rows)} rows) → {csv_path}"
        )

    return run_gensar(QWEN72B_KEY, results_dir=results_dir)


def run_qwen72b_genss(results_dir: str = None) -> pd.DataFrame:
    """
    Creates study2_ss_qwen72b.csv from scratch then runs GenSS preference scoring.
    """
    if results_dir is None:
        results_dir = config.RESULTS_DIR

    csv_path = os.path.join(results_dir, f"study2_ss_{QWEN72B_KEY}.csv")
    if not os.path.exists(csv_path):
        from data_pipeline import load_dataset_stereoset
        df_raw = load_dataset_stereoset(sample_n=config.SAMPLE_STEREOSET)
        rows = []
        for idx, row in df_raw.iterrows():
            rows.append({
                "item_id":         f"stereo_{idx}",
                "model_name":      QWEN72B_KEY,
                "context":         row.get("context", ""),
                "stereotype":      row.get("stereotype", ""),
                "anti_stereotype": row.get("anti_stereotype", ""),
                "target":          row.get("target", ""),
                "bias_type":       row.get("bias_type", ""),
                "ppl_stereo":      -1.0,
                "ppl_antistereo":  -1.0,
                "prefers_stereo":  -1,
                "gen_ss":          float("nan"),
            })
        pd.DataFrame(rows).to_csv(csv_path, index=False, encoding="utf-8-sig")
        logger.info(
            f"GenSS [{QWEN72B_KEY}]: created baseline CSV ({len(rows)} rows) → {csv_path}"
        )

    return run_genss(QWEN72B_KEY, results_dir=results_dir)
