"""
config_rerun.py — Configuration overrides for the TACL Re_Run.

Inherits everything from the parent config.py, then patches specific values
to close three reviewer-blocking gaps:
  1. Study 4 expanded: 60 → 300 items, 3 → 7 turns, GSS at 4 context depths
  2. 7th model: Qwen2.5-72B-Instruct via OpenRouter (already in .env keys)
  3. GenSAR/GenSS: preference-question coverage for all API models

Run from Codes/ directory:
  python Re_Run/main_rerun.py
"""

import os
import sys
import threading

# Make parent Codes/ importable from this sub-package
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import config as _cfg

# Re-export everything from parent so downstream code can just
# "from Re_Run.config_rerun import *" without breaking.
from config import (  # noqa: F401
    HF_TOKEN,
    GEMINI_KEYS, GEMINI_MODEL_ID, GEMINI_ENDPOINT,
    DEEPSEEK_KEYS, OPENROUTER_KEYS, MISTRAL_KEYS,
    MISTRAL_API_KEY, MISTRAL_MODEL_NAME,
    KeyRotator,
    get_gemini_rotator, get_deepseek_rotator,
    get_openrouter_rotator, get_mistral_rotator,
    LANGUAGE, LANGUAGE_DISPLAY,
    GENERATION_SEEDS, GENERATION_MODEL,
    PARALLEL_LOCAL_MODELS,
    DATASET_CROWS_PAIRS, DATASET_STEREOSET,
    DATASET_HONEST, DATASET_INDIAN_BIAS,
    SAMPLE_CROWS_PAIRS, SAMPLE_STEREOSET,
    SAMPLE_HONEST, SAMPLE_INDIAN_BIAS,
    N_PARAPHRASE_VARIANTS, PARAPHRASE_TEMPERATURE,
    N_CONTROL_PROMPTS,
    RESULTS_DIR, FIGURES_DIR, STATS_DIR, RAW_DIR,
    DATA_CACHE_DIR, PROGRESS_FILE,
    validate_keys,
)

# ── Study 4 overrides ─────────────────────────────────────────────────────────
RANDOM_SEED       = _cfg.RANDOM_SEED
SAMPLE_MULTI_TURN = 300           # was 60 — 75 per category × 4 categories
MULTI_TURN_DEPTHS = [0, 2, 4, 6]  # prior-turn context depths to measure GSS at

# ── Parallelism controls ──────────────────────────────────────────────────────
# H100 80 GB: local models loaded simultaneously (~27 GB total), inference sequential.
# API models are IO-bound — safe to run N_API_WORKERS threads concurrently.
N_API_WORKERS    = 4   # concurrent API model calls (outer: one thread per model)
N_GENSAR_WORKERS = 6   # concurrent row-level threads within one model's GenSAR run

# ── 7th model: Qwen2.5-72B via OpenRouter ────────────────────────────────────
QWEN72B_KEY      = "qwen72b"
QWEN72B_MODEL_ID = "qwen/qwen-2.5-72b-instruct"

# Build 7-model MODELS dict (copy to avoid mutating parent module)
MODELS = dict(_cfg.MODELS)
MODELS[QWEN72B_KEY] = {
    "display_name": "Qwen2.5-72B",
    "model_id":     QWEN72B_MODEL_ID,
    "type":         "api",
    "provider":     "openrouter",
    "temperature":  0.7,
    "max_tokens":   512,
}

# ── Model groupings ───────────────────────────────────────────────────────────
ALL_MODELS   = list(MODELS.keys())           # 7 models
API_MODELS   = ["deepseek", "gpt4o_mini", "grok", QWEN72B_KEY]
LOCAL_MODELS = ["qwen", "gemma3", "llama"]


# ── Thread-safe key rotator ───────────────────────────────────────────────────
# The standard KeyRotator uses a plain int counter which is NOT thread-safe
# across attribute read-modify-write bytecodes. We wrap it with a Lock.

class ThreadSafeKeyRotator:
    """
    Drop-in replacement for config.KeyRotator that is safe to share across
    concurrent ThreadPoolExecutor workers.

    Uses threading.Lock to make get_next_key() and mark_exhausted() atomic.
    """

    def __init__(self, keys: list, provider: str = ""):
        if not keys:
            raise ValueError(f"No API keys supplied for provider '{provider}'")
        self._keys     = list(keys)
        self._provider = provider
        self._index    = 0
        self._lock     = threading.Lock()

    def get_next_key(self) -> str:
        with self._lock:
            if not self._keys:
                raise RuntimeError(
                    f"All keys for '{self._provider}' have been exhausted."
                )
            key = self._keys[self._index % len(self._keys)]
            self._index += 1
            return key

    def mark_exhausted(self, key: str) -> None:
        with self._lock:
            if key in self._keys:
                self._keys.remove(key)

    @property
    def provider(self) -> str:
        return self._provider

    def __len__(self) -> int:
        with self._lock:
            return len(self._keys)


# ── Module-level thread-safe singleton rotators ───────────────────────────────
# These are shared across ALL concurrent threads in the Re_Run.
# Initialised lazily but protected by a lock so only one instance is created.

_ts_openrouter_rotator: "ThreadSafeKeyRotator | None" = None
_ts_deepseek_rotator:   "ThreadSafeKeyRotator | None" = None
_ts_init_lock = threading.Lock()


def get_ts_openrouter_rotator() -> ThreadSafeKeyRotator:
    global _ts_openrouter_rotator
    with _ts_init_lock:
        if _ts_openrouter_rotator is None:
            _ts_openrouter_rotator = ThreadSafeKeyRotator(
                _cfg.OPENROUTER_KEYS, "openrouter"
            )
    return _ts_openrouter_rotator


def get_ts_deepseek_rotator() -> ThreadSafeKeyRotator:
    global _ts_deepseek_rotator
    with _ts_init_lock:
        if _ts_deepseek_rotator is None:
            _ts_deepseek_rotator = ThreadSafeKeyRotator(
                _cfg.DEEPSEEK_KEYS, "deepseek"
            )
    return _ts_deepseek_rotator
