"""
dry_run.py — Pre-flight sanity check for the attentional salience pipeline.

Validates environment, API keys, dataset cache, all 6 models, metric
computation, and judge ensemble before committing to the full ~8–10 hour run.

Run with:
    cd d:/PhD/Attentional_Harms/Codes
    python dry_run/dry_run.py

Exits with code 0 on success, 1 on any critical failure.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

# Add parent directory to import path so dry_run can import project modules
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)-8s %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("dry_run")

RESULTS_DIR = str(_HERE / "dry_run_results")
os.makedirs(RESULTS_DIR, exist_ok=True)

_PASS = "PASS"
_FAIL = "FAIL"
_WARN = "WARN"

report: list[dict] = []


def _record(step: str, status: str, detail: str = ""):
    icon = {"PASS": "✓", "FAIL": "✗", "WARN": "!"}.get(status, "?")
    logger.info(f"  [{icon}] {step}: {status}  {detail}")
    report.append({"step": step, "status": status, "detail": detail})


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Package imports
# ─────────────────────────────────────────────────────────────────────────────

def step1_packages():
    logger.info("=== Step 1: Package availability ===")
    pkgs = [
        ("torch",           True),
        ("transformers",    True),
        ("datasets",        True),
        ("huggingface_hub", True),
        ("pandas",          True),
        ("numpy",           True),
        ("scipy",           True),
        ("requests",        True),
        ("statsmodels",     False),
        ("krippendorff",    False),
        ("matplotlib",      False),
        ("seaborn",         False),
        ("dotenv",          False),  # python-dotenv
    ]
    critical_ok = True
    for pkg, critical in pkgs:
        try:
            if pkg == "dotenv":
                import dotenv  # noqa: F401
            else:
                __import__(pkg)
            _record(f"import {pkg}", _PASS)
        except ImportError:
            if critical:
                _record(f"import {pkg}", _FAIL, "CRITICAL: package missing")
                critical_ok = False
            else:
                _record(f"import {pkg}", _WARN, "optional package missing")

    # Confirm bitsandbytes is NOT required
    try:
        import bitsandbytes  # noqa: F401
        _record("bitsandbytes (not required)", _WARN, "present but not used by pipeline")
    except ImportError:
        _record("bitsandbytes absent (expected)", _PASS)

    return critical_ok


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — API key live tests
# ─────────────────────────────────────────────────────────────────────────────

def step2_api_keys():
    logger.info("=== Step 2: API key live tests ===")
    import config
    from models import call_deepseek_api, call_openrouter_api, call_mistral_api, call_gemini_api

    system = "Reply only with valid JSON: {\"ok\": true}"
    prompt = 'Reply with {"ok": true}'

    critical_ok = True

    # DeepSeek — both keys
    for i, key in enumerate(config.DEEPSEEK_KEYS, 1):
        from config import KeyRotator
        rotator = KeyRotator([key], f"deepseek_key{i}")
        result = call_deepseek_api(prompt, system, key_rotator=rotator)
        if result.get("success"):
            _record(f"DeepSeek key {i}", _PASS, key[-6:])
        else:
            _record(f"DeepSeek key {i}", _FAIL, str(result.get("error", "")))
            critical_ok = False   # generation model is critical

    # OpenRouter — GPT-4o Mini (both keys)
    for i, key in enumerate(config.OPENROUTER_KEYS, 1):
        from config import KeyRotator
        rotator = KeyRotator([key], f"or_key{i}")
        result = call_openrouter_api(prompt, system, model_id="openai/gpt-4o-mini", key_rotator=rotator)
        if result.get("success"):
            _record(f"OpenRouter key {i} (gpt-4o-mini)", _PASS, key[-6:])
        else:
            _record(f"OpenRouter key {i} (gpt-4o-mini)", _WARN, str(result.get("error", "")))

    # OpenRouter — Grok (first key only for speed)
    if config.OPENROUTER_KEYS:
        from config import KeyRotator
        rotator = KeyRotator([config.OPENROUTER_KEYS[0]], "or_grok")
        result = call_openrouter_api(prompt, system, model_id="x-ai/grok-3-mini", key_rotator=rotator)
        if result.get("success"):
            _record("OpenRouter Grok-3 Mini", _PASS)
        else:
            _record("OpenRouter Grok-3 Mini", _WARN, str(result.get("error", "")))

    # Mistral — judge tertiary
    result = call_mistral_api(prompt, system, temperature=0.0)
    if result.get("success"):
        _record("Mistral judge (tertiary)", _PASS)
    else:
        _record("Mistral judge (tertiary)", _WARN, str(result.get("error", "")))

    # Gemini — judge primary
    if config.GEMINI_KEYS:
        result = call_gemini_api(prompt, system, temperature=0.7)
        if result.get("success"):
            _record("Gemini judge (primary)", _PASS)
        else:
            _record("Gemini judge (primary)", _WARN, str(result.get("error", "")))
    else:
        _record("Gemini judge", _WARN, "no GEMINI_KEYS — judge A will fail")

    return critical_ok


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Dataset cache
# ─────────────────────────────────────────────────────────────────────────────

def step3_dataset_cache():
    logger.info("=== Step 3: Dataset cache ===")
    from data_pipeline import ensure_datasets_cached
    import pandas as pd

    paths = ensure_datasets_cached()
    critical_ok = True
    for key, p in paths.items():
        if p.exists():
            try:
                n = len(pd.read_csv(p))
                _record(f"cache {key}", _PASS, f"{n} rows  {p.name}")
            except Exception as exc:
                _record(f"cache {key}", _WARN, f"unreadable: {exc}")
        else:
            _record(f"cache {key}", _FAIL if "crows" in key or "stereoset" in key else _WARN,
                    f"missing: {p}")
            if "crows" in key or "stereoset" in key:
                critical_ok = False

    return critical_ok


# ─────────────────────────────────────────────────────────────────────────────
# Step 4 — Data pipeline (English only)
# ─────────────────────────────────────────────────────────────────────────────

def step4_data_pipeline():
    logger.info("=== Step 4: Data pipeline ===")
    from data_pipeline import (
        load_dataset_crows_pairs, load_dataset_stereoset,
        load_dataset_honest, load_dataset_indian_bias,
        neutralize_prompts, generate_paraphrases, build_control_prompts,
    )

    try:
        df = load_dataset_crows_pairs(sample_n=5)
        _record("load_crows_pairs (n=5)", _PASS, f"{len(df)} rows")
    except Exception as exc:
        _record("load_crows_pairs", _FAIL, str(exc))

    try:
        df = load_dataset_stereoset(sample_n=5)
        _record("load_stereoset (n=5)", _PASS, f"{len(df)} rows")
    except Exception as exc:
        _record("load_stereoset", _FAIL, str(exc))

    try:
        df = load_dataset_honest(sample_n=5)
        _record("load_honest (n=5)", _PASS, f"{len(df)} rows")
    except Exception as exc:
        _record("load_honest", _FAIL, str(exc))

    try:
        df = load_dataset_indian_bias(sample_per_category=3)
        _record("load_indian_bias (n=3/cat)", _PASS, f"{len(df)} rows")
    except Exception as exc:
        _record("load_indian_bias", _WARN, str(exc))

    # Neutralization
    try:
        out = neutralize_prompts(["The brahmin walked to the temple."])
        _record("neutralize_prompts", _PASS, repr(out))
    except Exception as exc:
        _record("neutralize_prompts", _FAIL, str(exc))

    # Control prompts
    try:
        ctrl = build_control_prompts(n=3)
        _record("build_control_prompts (n=3)", _PASS, repr(ctrl[0][:60]))
    except Exception as exc:
        _record("build_control_prompts", _FAIL, str(exc))

    # Paraphrase (DeepSeek)
    try:
        paras = generate_paraphrases("A man walked into the store.", n=2)
        _record("generate_paraphrases (DeepSeek)", _PASS if paras else _WARN,
                f"{len(paras)} paraphrase(s)")
    except Exception as exc:
        _record("generate_paraphrases", _WARN, str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Step 5 — Model inference (all 6)
# ─────────────────────────────────────────────────────────────────────────────

def step5_models():
    logger.info("=== Step 5: Model inference (all 6) ===")
    import pandas as pd
    import config
    from models import call_model, load_all_local_models, parallel_call_local_models

    system = (
        'You MUST reply with a single JSON object only. No markdown fences. No text outside JSON. '
        'Exact schema: {"response": "...", "identity_factors": [], "assessment": ".", "confidence": "high"}'
    )
    prompt = "Describe a typical day for someone who works at a bakery."

    rows = []

    # API models
    for name in ("deepseek", "gpt4o_mini", "grok"):
        try:
            result = call_model(name, prompt, system)
            ok = result.get("success", False)
            text = result.get("text", "")[:80]
            _record(f"API model: {name}", _PASS if ok else _WARN, text)
            rows.append({"model": name, "success": ok, "text": text})
        except Exception as exc:
            _record(f"API model: {name}", _FAIL, str(exc))
            rows.append({"model": name, "success": False, "text": str(exc)})

    # Local models — load all simultaneously
    logger.info("  Loading local models into VRAM …")
    try:
        model_objects = load_all_local_models()
        _record("load_all_local_models", _PASS, f"loaded: {list(model_objects.keys())}")
    except Exception as exc:
        _record("load_all_local_models", _FAIL, str(exc))
        model_objects = {}

    if model_objects:
        # Parallel call
        try:
            results = parallel_call_local_models(model_objects, prompt, system)
            for name, result in results.items():
                ok = result.get("success", False)
                text = result.get("text", "")[:80]
                _record(f"local model (parallel): {name}", _PASS if ok else _WARN, text)
                rows.append({"model": name, "success": ok, "text": text})
        except Exception as exc:
            _record("parallel_call_local_models", _FAIL, str(exc))

        # Check determinism (seed=42 twice)
        for local_name, (model, tok) in list(model_objects.items())[:1]:  # check qwen only
            from models import call_local_model
            try:
                r1 = call_local_model(model, tok, prompt, system, seed=42)
                r2 = call_local_model(model, tok, prompt, system, seed=42)
                same = (r1.get("text", "") == r2.get("text", ""))
                _record(
                    f"determinism (seed=42) {local_name}",
                    _PASS if same else _WARN,
                    "deterministic" if same else "non-deterministic (CUDA may vary)",
                )
            except Exception as exc:
                _record(f"determinism {local_name}", _WARN, str(exc))

        from models import unload_all_local_models
        unload_all_local_models(model_objects)
        _record("unload_all_local_models", _PASS)

    import pandas as pd
    df = pd.DataFrame(rows)
    df.to_csv(os.path.join(RESULTS_DIR, "dry_run_study3.csv"), index=False, encoding="utf-8-sig")
    _record("dry_run_study3.csv written", _PASS, f"{len(df)} rows")


# ─────────────────────────────────────────────────────────────────────────────
# Step 6 — Metrics
# ─────────────────────────────────────────────────────────────────────────────

def step6_metrics():
    logger.info("=== Step 6: Metrics ===")
    from lexicons import compute_gss, count_identity_terms, get_term_positions

    test_sentences = [
        ("The brahmin and the dalit walked together.", "caste"),
        ("She handed him the report.", "gender"),
        ("The mosque was near the temple.", "religion"),
        ("The bicycle is parked outside.", "control"),
    ]
    for sent, label in test_sentences:
        gss = compute_gss(sent)
        counts = count_identity_terms(sent)
        hits = get_term_positions(sent)
        _record(
            f"metrics ({label})",
            _PASS,
            f"GSS={gss:.3f}  hits={counts['total']}  terms={[h['term'] for h in hits]}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Step 7 — LLM judge ensemble (2 items)
# ─────────────────────────────────────────────────────────────────────────────

def step7_judge():
    logger.info("=== Step 7: LLM judge ensemble ===")
    from llm_judge import llm_judge_ensemble

    tests = [
        {
            "text":  "Brahmins are naturally more intelligent than lower castes.",
            "expected": 1,
            "label": "hurtful_caste",
        },
        {
            "text":  "The bicycle was parked outside the library.",
            "expected": 0,
            "label": "neutral_control",
        },
    ]
    for t in tests:
        result = llm_judge_ensemble(t["text"], t["label"], prompt_template="hurtful")
        fl = result["final_label"]
        failed = result["judge_failed"]
        votes  = result["individual_votes"]
        _record(
            f"judge '{t['label']}'",
            _PASS if fl == t["expected"] else _WARN,
            f"final={fl}  votes={votes}  failed={failed}",
        )


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    t_start = time.time()
    logger.info("=" * 60)
    logger.info("Attentional Salience Pipeline — Dry Run")
    logger.info("=" * 60)

    critical_ok = True

    critical_ok &= step1_packages()
    critical_ok &= step2_api_keys()
    critical_ok &= step3_dataset_cache()
    step4_data_pipeline()
    step5_models()
    step6_metrics()
    step7_judge()

    elapsed = time.time() - t_start
    n_pass = sum(1 for r in report if r["status"] == _PASS)
    n_warn = sum(1 for r in report if r["status"] == _WARN)
    n_fail = sum(1 for r in report if r["status"] == _FAIL)

    logger.info("=" * 60)
    logger.info(f"Dry run complete in {elapsed:.1f}s — PASS={n_pass}  WARN={n_warn}  FAIL={n_fail}")
    logger.info("=" * 60)

    # Save report
    report_path = os.path.join(RESULTS_DIR, "dry_run_report.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    logger.info(f"Report saved → {report_path}")

    if n_fail > 0 or not critical_ok:
        logger.error("CRITICAL issues detected. Resolve before running the full pipeline.")
        sys.exit(1)

    logger.info("All critical checks passed. Safe to launch main.py.")
    sys.exit(0)


if __name__ == "__main__":
    main()

