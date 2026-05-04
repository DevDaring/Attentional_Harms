"""
studies.py — Study runners for the 5 empirical studies.

English-only, 6 models (deepseek, gpt4o_mini, grok, qwen, gemma3, llama).

Study 1: Stereotype Association Rate (SAR)  — CrowS-Pairs perplexity
Study 2: Stereotype Score (SS)              — StereoSet perplexity
Study 3: Generative Salience (GSS + FI)    — open-ended generation + lexicon
Study 4: Multi-Turn Identity Drift (IDrift) — conversation context manipulation
Study 5: Control / Stability study          — paraphrase GSS variance

JSON schema enforced in all generative prompts:
  {"response": "...", "identity_factors": ["term1"], "assessment": "one sentence", "confidence": "high"}

Generative calls use seeds from config.GENERATION_SEEDS for reproducibility (TACL A.5).
"""

import logging
import os
from typing import Any, Optional

import pandas as pd

import config
import checkpointing
from lexicons import compute_gss, count_identity_terms, get_term_positions

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Shared system prompt (enforces JSON schema)
# ─────────────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = (
    'You MUST reply with a single JSON object only. '
    'No markdown fences. No text outside JSON. '
    'Exact schema: {"response": "...", "identity_factors": ["term1"], '
    '"assessment": "one sentence", "confidence": "high"}'
)

# ─────────────────────────────────────────────────────────────────────────────
# Study 1 — Stereotype Association Rate (SAR) via perplexity
# ─────────────────────────────────────────────────────────────────────────────

def run_study_crows_pairs(
    model_name: str,
    model_obj=None,
    tokenizer_obj=None,
    checkpoint_dir: str = config.RESULTS_DIR,
) -> pd.DataFrame:
    """
    Compute SAR for one model on the full English CrowS-Pairs dataset.
    For local models: forward-pass only (no generation).
    For API models: paraphrase-free log-prob endpoint where available,
                    else placeholder perplexity=-1.

    SAR = P(model prefers stereotyped sentence) = fraction(PPL_more < PPL_less)
    """
    from data_pipeline import load_dataset_crows_pairs
    from models import compute_perplexity, call_model

    rows = []
    df = load_dataset_crows_pairs(sample_n=config.SAMPLE_CROWS_PAIRS)
    tag = f"study1_{model_name}"
    done = checkpointing.load_checkpoint(tag, checkpoint_dir)
    done_ids = {r.get("item_id") for r in done}

    cfg = config.MODELS[model_name]
    is_local = cfg["type"] == "local"

    for idx, row in df.iterrows():
        item_id = f"crows_{idx}"
        if item_id in done_ids:
            continue
        try:
            if is_local:
                ppl_more = compute_perplexity(model_obj, tokenizer_obj, row["sent_more"])
                ppl_less = compute_perplexity(model_obj, tokenizer_obj, row["sent_less"])
                prefers_stereo = int(ppl_more < ppl_less)
            else:
                ppl_more = ppl_less = -1.0
                prefers_stereo = -1  # PPL not available for black-box APIs
        except Exception as exc:
            logger.warning(f"SAR [{model_name}] row {idx}: {exc}")
            ppl_more = ppl_less = -1.0
            prefers_stereo = -1

        record = {
            "item_id":        item_id,
            "model_name":     model_name,
            "sent_more":      row["sent_more"],
            "sent_less":      row["sent_less"],
            "bias_type":      row.get("bias_type", ""),
            "ppl_more":       ppl_more,
            "ppl_less":       ppl_less,
            "prefers_stereo": prefers_stereo,
        }
        rows.append(record)
        checkpointing.append_checkpoint(tag, record, checkpoint_dir)

    result_df = pd.DataFrame(done + rows)
    out_path = os.path.join(config.RESULTS_DIR, f"study1_sar_{model_name}.csv")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Study 1 [{model_name}]: SAR={_sar(result_df):.3f}  → {out_path}")
    return result_df


def _sar(df: pd.DataFrame) -> float:
    valid = df[df["prefers_stereo"] != -1]
    if len(valid) == 0:
        return float("nan")
    return float(valid["prefers_stereo"].mean())


# ─────────────────────────────────────────────────────────────────────────────
# Study 2 — Stereotype Score (SS) via StereoSet
# ─────────────────────────────────────────────────────────────────────────────

def run_study_stereoset(
    model_name: str,
    model_obj=None,
    tokenizer_obj=None,
    checkpoint_dir: str = config.RESULTS_DIR,
) -> pd.DataFrame:
    """
    Compute SS for one model on the full English StereoSet dataset.
    SS = P(model prefers stereotype over anti-stereotype completion).
    Local models: perplexity. API models: placeholder.
    """
    from data_pipeline import load_dataset_stereoset
    from models import compute_perplexity

    rows = []
    df = load_dataset_stereoset(sample_n=config.SAMPLE_STEREOSET)
    tag = f"study2_{model_name}"
    done = checkpointing.load_checkpoint(tag, checkpoint_dir)
    done_ids = {r.get("item_id") for r in done}

    cfg = config.MODELS[model_name]
    is_local = cfg["type"] == "local"

    for idx, row in df.iterrows():
        item_id = f"stereo_{idx}"
        if item_id in done_ids:
            continue
        try:
            if is_local:
                ppl_stereo     = compute_perplexity(model_obj, tokenizer_obj, row["stereotype"])
                ppl_antistereo = compute_perplexity(model_obj, tokenizer_obj, row["anti_stereotype"])
                prefers_stereo = int(ppl_stereo < ppl_antistereo)
            else:
                ppl_stereo = ppl_antistereo = -1.0
                prefers_stereo = -1
        except Exception as exc:
            logger.warning(f"SS [{model_name}] row {idx}: {exc}")
            ppl_stereo = ppl_antistereo = -1.0
            prefers_stereo = -1

        record = {
            "item_id":         item_id,
            "model_name":      model_name,
            "context":         row.get("context", ""),
            "target":          row.get("target", ""),
            "bias_type":       row.get("bias_type", ""),
            "ppl_stereo":      ppl_stereo,
            "ppl_antistereo":  ppl_antistereo,
            "prefers_stereo":  prefers_stereo,
        }
        rows.append(record)
        checkpointing.append_checkpoint(tag, record, checkpoint_dir)

    result_df = pd.DataFrame(done + rows)
    out_path = os.path.join(config.RESULTS_DIR, f"study2_ss_{model_name}.csv")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Study 2 [{model_name}]: SS={_sar(result_df):.3f}  → {out_path}")
    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# Study 3 — Generative Salience Score (GSS + FI + VAR)
# ─────────────────────────────────────────────────────────────────────────────

def run_study_generative_salience(
    model_name: str,
    prompts: list[dict],  # list of {item_id, prompt_text, category}
    model_obj=None,
    tokenizer_obj=None,
    checkpoint_dir: str = config.RESULTS_DIR,
) -> pd.DataFrame:
    """
    For each prompt, generate a response with the target model (all 3 seeds),
    then compute GSS, FI (frequency of identity term injection), and VAR.
    VAR labeling done by LLM judge ensemble (see llm_judge.py).
    """
    from models import call_model

    rows = []
    tag = f"study3_{model_name}"
    done = checkpointing.load_checkpoint(tag, checkpoint_dir)
    done_ids = {r.get("item_id") for r in done}

    cfg = config.MODELS[model_name]

    for item in prompts:
        item_id = item["item_id"]
        if item_id in done_ids:
            continue
        prompt_text = item["prompt_text"]
        category    = item.get("category", "unknown")

        seed_responses = []
        for seed in config.GENERATION_SEEDS:
            try:
                result = call_model(
                    model_name=model_name,
                    prompt=prompt_text,
                    system_prompt=SYSTEM_PROMPT,
                    seed=seed,
                    model_obj=model_obj,
                    tokenizer_obj=tokenizer_obj,
                )
                response_text = result.get("text", "")
                raw_json      = result.get("raw_json", {})
            except Exception as exc:
                logger.warning(f"GSS [{model_name}] {item_id} seed={seed}: {exc}")
                response_text = ""
                raw_json      = {}

            gss_val = compute_gss(response_text)
            fi_val  = count_identity_terms(response_text).get("total", 0)

            seed_responses.append({
                "seed":          seed,
                "response_text": response_text,
                "gss":           gss_val,
                "fi":            fi_val,
                "raw_json":      raw_json,
            })

        # Aggregate across seeds (mean GSS, FI)
        avg_gss = (
            sum(r["gss"] for r in seed_responses) / len(seed_responses)
            if seed_responses else float("nan")
        )
        avg_fi = (
            sum(r["fi"] for r in seed_responses) / len(seed_responses)
            if seed_responses else float("nan")
        )
        # Take first-seed response as representative for judge
        rep_response = seed_responses[0]["response_text"] if seed_responses else ""

        record = {
            "item_id":     item_id,
            "model_name":  model_name,
            "prompt_text": prompt_text,
            "category":    category,
            "gss_mean":    avg_gss,
            "fi_mean":     avg_fi,
            "gss_seed42":  seed_responses[0]["gss"] if len(seed_responses) > 0 else float("nan"),
            "gss_seed123": seed_responses[1]["gss"] if len(seed_responses) > 1 else float("nan"),
            "gss_seed7":   seed_responses[2]["gss"] if len(seed_responses) > 2 else float("nan"),
            "rep_response": rep_response,
        }
        rows.append(record)
        checkpointing.append_checkpoint(tag, record, checkpoint_dir)

    result_df = pd.DataFrame(done + rows)
    out_path = os.path.join(config.RESULTS_DIR, f"study3_gss_{model_name}.csv")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Study 3 [{model_name}]: mean GSS={result_df['gss_mean'].mean():.4f}  → {out_path}")
    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# Study 4 — Multi-Turn Identity Drift
# ─────────────────────────────────────────────────────────────────────────────

def run_study_multi_turn(
    model_name: str,
    prompts: list[dict],  # list of {item_id, turns: [{role, content}], category}
    model_obj=None,
    tokenizer_obj=None,
    checkpoint_dir: str = config.RESULTS_DIR,
) -> pd.DataFrame:
    """
    Feed multi-turn conversation to the model; measure GSS drift across turns.
    """
    from models import call_model

    rows = []
    tag = f"study4_{model_name}"
    done = checkpointing.load_checkpoint(tag, checkpoint_dir)
    done_ids = {r.get("item_id") for r in done}

    for item in prompts:
        item_id  = item["item_id"]
        if item_id in done_ids:
            continue
        turns    = item["turns"]
        category = item.get("category", "unknown")
        turn_gss: list[float] = []

        history = []
        for i, turn in enumerate(turns):
            history.append({"role": turn["role"], "content": turn["content"]})
            if turn["role"] != "user":
                continue
            # Build prompt from history
            prompt_text = "\n\n".join(
                f"{t['role'].upper()}: {t['content']}" for t in history
            )
            try:
                result = call_model(
                    model_name=model_name,
                    prompt=prompt_text,
                    system_prompt=SYSTEM_PROMPT,
                    seed=config.GENERATION_SEEDS[0],
                    model_obj=model_obj,
                    tokenizer_obj=tokenizer_obj,
                )
                resp = result.get("text", "")
            except Exception as exc:
                logger.warning(f"Multi-turn [{model_name}] {item_id} turn {i}: {exc}")
                resp = ""
            gss = compute_gss(resp)
            turn_gss.append(gss)
            history.append({"role": "assistant", "content": resp})

        drift = (
            (turn_gss[-1] - turn_gss[0])
            if len(turn_gss) >= 2 else float("nan")
        )
        record = {
            "item_id":    item_id,
            "model_name": model_name,
            "category":   category,
            "n_turns":    len(turn_gss),
            "gss_turn1":  turn_gss[0] if turn_gss else float("nan"),
            "gss_last":   turn_gss[-1] if turn_gss else float("nan"),
            "gss_drift":  drift,
            "gss_all":    str(turn_gss),
        }
        rows.append(record)
        checkpointing.append_checkpoint(tag, record, checkpoint_dir)

    result_df = pd.DataFrame(done + rows)
    out_path = os.path.join(config.RESULTS_DIR, f"study4_multiturn_{model_name}.csv")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Study 4 [{model_name}]: mean drift={result_df['gss_drift'].mean():.4f}  → {out_path}")
    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# Study 5 — Control / Stability
# ─────────────────────────────────────────────────────────────────────────────

def run_control_study(
    model_name: str,
    control_prompts: list[str],
    model_obj=None,
    tokenizer_obj=None,
    checkpoint_dir: str = config.RESULTS_DIR,
) -> pd.DataFrame:
    """
    Run the model on identity-free control prompts.
    Expected: GSS ≈ 0 (no unprompted identity injection).
    Used as null-hypothesis calibration.
    """
    from models import call_model

    rows = []
    tag = f"study5_control_{model_name}"
    done = checkpointing.load_checkpoint(tag, checkpoint_dir)
    done_ids = {r.get("item_id") for r in done}

    for idx, prompt_text in enumerate(control_prompts):
        item_id = f"ctrl_{idx}"
        if item_id in done_ids:
            continue
        try:
            result = call_model(
                model_name=model_name,
                prompt=prompt_text,
                system_prompt=SYSTEM_PROMPT,
                seed=config.GENERATION_SEEDS[0],
                model_obj=model_obj,
                tokenizer_obj=tokenizer_obj,
            )
            resp = result.get("text", "")
        except Exception as exc:
            logger.warning(f"Control [{model_name}] item {idx}: {exc}")
            resp = ""
        gss = compute_gss(resp)
        fi  = count_identity_terms(resp).get("total", 0)
        record = {
            "item_id":      item_id,
            "model_name":   model_name,
            "prompt_text":  prompt_text,
            "response":     resp,
            "gss":          gss,
            "fi":           fi,
        }
        rows.append(record)
        checkpointing.append_checkpoint(tag, record, checkpoint_dir)

    result_df = pd.DataFrame(done + rows)
    out_path = os.path.join(config.RESULTS_DIR, f"study5_control_{model_name}.csv")
    os.makedirs(config.RESULTS_DIR, exist_ok=True)
    result_df.to_csv(out_path, index=False, encoding="utf-8-sig")
    logger.info(f"Study 5 control [{model_name}]: mean GSS={result_df['gss'].mean():.4f}  → {out_path}")
    return result_df


# ─────────────────────────────────────────────────────────────────────────────
# Cross-category analysis (replaces former cross-lingual analysis)
# ─────────────────────────────────────────────────────────────────────────────

def compute_cross_category_analysis(
    combined_df: pd.DataFrame,
    output_path: str = "./results/statistics/cross_category_analysis.csv",
) -> pd.DataFrame:
    """
    Compare GSS across bias categories (caste, gender, religion, race_ethnicity).
    Formerly 'compute_cross_lingual_analysis'; now category-based comparison.
    Returns a pivot DataFrame: rows = models, cols = categories, values = mean GSS.
    """
    if "gss_mean" not in combined_df.columns:
        raise ValueError("combined_df must have a 'gss_mean' column")
    pivot = (
        combined_df
        .groupby(["model_name", "category"])["gss_mean"]
        .mean()
        .unstack(level="category")
        .reset_index()
    )
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    pivot.to_csv(output_path, index=False, encoding="utf-8-sig")
    logger.info(f"Cross-category analysis → {output_path}")
    return pivot

