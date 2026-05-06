"""
config.py — Configuration for:
"Attentional Salience as a Complementary Bias Lens in Language Models"
Target venue: ACL

English-only experiment.
6 models: DeepSeek V3 (API), GPT-4o Mini (OpenRouter), Grok-3 Mini (OpenRouter),
          Qwen2.5-1.5B (local), Gemma-3-4B (local), Llama-3.1-8B (local).
LLM-judge: Gemini (primary) + DeepSeek (secondary) + Mistral (tertiary), zero retry.
All keys loaded from .env — none hardcoded.
"""

import os
import logging
from typing import Optional
from dotenv import load_dotenv

logger = logging.getLogger(__name__)

load_dotenv(override=True)


def _strip(v: Optional[str]) -> Optional[str]:
    """Strip whitespace/quotes from env values (common copy-paste artefact)."""
    if not v:
        return None
    return v.strip().strip('"').strip("'") or None


# ── HuggingFace ───────────────────────────────────────────────────────────────
HF_TOKEN: Optional[str] = (
    _strip(os.environ.get("HUGGINGFACE_TOKEN"))
    or _strip(os.environ.get("HF_Classic_Token"))
    or _strip(os.environ.get("HF_KEY"))
    or _strip(os.environ.get("HF_TOKEN"))
)

# ── Gemini / GCP (judge-only — 4 keys, round-robin) ─────────────────────────
GEMINI_KEYS: list[str] = [
    k for k in [
        _strip(os.environ.get("GEMINI_API_KEY_1")),
        _strip(os.environ.get("GEMINI_API_KEY_2")),
        _strip(os.environ.get("GEMINI_API_KEY_3")),
        _strip(os.environ.get("GEMINI_API_KEY_4")),
        # legacy key names
        _strip(os.environ.get("GCP_Key1")),
        _strip(os.environ.get("GCP_Key2")),
        _strip(os.environ.get("GCP_Key3")),
        _strip(os.environ.get("GCP_Key4")),
    ] if k
]
# gemini-2.5-flash is the current stable model (not deprecated).
# gemini-2.0-flash and gemini-1.5-flash are deprecated and shut down soon.
# Next upgrade: gemini-3-flash-preview (no shutdown date, frontier-class).
GEMINI_MODEL_ID: str = (
    _strip(os.environ.get("GEMINI_MODEL_NAME"))
    or _strip(os.environ.get("GEMINI_MODEL_ID"))
    or "gemini-2.5-flash"
)
GEMINI_ENDPOINT = (
    "https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent"
)

# ── DeepSeek (up to 2 keys, round-robin) ─────────────────────────────────────
DEEPSEEK_KEYS: list[str] = [
    k for k in [
        _strip(os.environ.get("DEEPSEEK_API_KEY_1")),
        _strip(os.environ.get("DEEPSEEK_API_KEY_2")),
        # legacy key names
        _strip(os.environ.get("Deepseek_API_key")),
        _strip(os.environ.get("Deepseek_API_key2")),
        _strip(os.environ.get("DEEPSEEK_API_KEY")),
    ] if k
]

# ── OpenRouter (up to 2 keys, round-robin) ────────────────────────────────────
OPENROUTER_KEYS: list[str] = [
    k for k in [
        _strip(os.environ.get("OPENROUTER_API_KEY_1")),
        _strip(os.environ.get("OPENROUTER_API_KEY_2")),
        # legacy key names
        _strip(os.environ.get("OR_Key1")),
        _strip(os.environ.get("OR_Key2")),
        _strip(os.environ.get("OPENROUTER_API_KEY")),
    ] if k
]

# ── Mistral (up to 2 keys, round-robin, judge tertiary) ──────────────────────
MISTRAL_KEYS: list[str] = [
    k for k in [
        _strip(os.environ.get("MISTRAL_API_KEY1")),
        _strip(os.environ.get("Mistral_API_KEY2")),
        # legacy single-key name
        _strip(os.environ.get("MISTRAL_API_KEY")),
    ] if k
]
# Backward-compat alias used by models.py
MISTRAL_API_KEY: Optional[str] = MISTRAL_KEYS[0] if MISTRAL_KEYS else None
MISTRAL_MODEL_NAME: str = (
    _strip(os.environ.get("MISTRAL_MODEL_NAME")) or "mistral-small-latest"
)

# ── Translation GCP keys (paraphrase generation) ─────────────────────────────
TRANSLATION_KEYS: list[str] = [
    k for k in [
        _strip(os.environ.get("GCP_Tanslation1")),   # typo in .env preserved
        _strip(os.environ.get("GCP_Tanslation2")),
        _strip(os.environ.get("GCP_Tanslation3")),
    ] if k
]


# ─────────────────────────────────────────────────────────────────────────────
# Generic round-robin key rotator (no retry, no sleeping, no reserved keys)
# ─────────────────────────────────────────────────────────────────────────────

class KeyRotator:
    """
    Round-robin across a list of API keys.
    Callers do ONE attempt per key. If a call fails, they move on — no retry.
    If a key is permanently dead, call mark_exhausted() to remove it.
    """

    def __init__(self, keys: list[str], provider: str = ""):
        if not keys:
            raise ValueError(f"No API keys supplied for provider '{provider}'")
        self._keys = list(keys)
        self._provider = provider
        self._index = 0

    def get_next_key(self) -> str:
        """Return the next key in round-robin order."""
        if not self._keys:
            raise RuntimeError(
                f"All keys for '{self._provider}' have been exhausted."
            )
        key = self._keys[self._index % len(self._keys)]
        self._index += 1
        return key

    def mark_exhausted(self, key: str) -> None:
        """Permanently remove a key from the pool (e.g., quota exceeded)."""
        if key in self._keys:
            self._keys.remove(key)
            logger.warning(
                f"Key ending '…{key[-6:]}' removed from {self._provider} pool. "
                f"{len(self._keys)} key(s) remaining."
            )

    @property
    def provider(self) -> str:
        return self._provider

    def __len__(self) -> int:
        return len(self._keys)


# ── Module-level rotator singletons (lazy init) ───────────────────────────────
_gemini_rotator: Optional[KeyRotator] = None
_deepseek_rotator: Optional[KeyRotator] = None
_openrouter_rotator: Optional[KeyRotator] = None


def get_gemini_rotator() -> KeyRotator:
    global _gemini_rotator
    if _gemini_rotator is None:
        _gemini_rotator = KeyRotator(GEMINI_KEYS, "gemini")
    return _gemini_rotator


def get_deepseek_rotator() -> KeyRotator:
    global _deepseek_rotator
    if _deepseek_rotator is None:
        _deepseek_rotator = KeyRotator(DEEPSEEK_KEYS, "deepseek")
    return _deepseek_rotator


def get_openrouter_rotator() -> KeyRotator:
    global _openrouter_rotator
    if _openrouter_rotator is None:
        _openrouter_rotator = KeyRotator(OPENROUTER_KEYS, "openrouter")
    return _openrouter_rotator


_mistral_rotator: Optional[KeyRotator] = None


def get_mistral_rotator() -> KeyRotator:
    global _mistral_rotator
    if _mistral_rotator is None:
        _mistral_rotator = KeyRotator(MISTRAL_KEYS, "mistral")
    return _mistral_rotator


# ── MODELS dict — 6 models ───────────────────────────────────────────────────
MODELS: dict[str, dict] = {
    # ── API models ──────────────────────────────────────────────────────────
    "deepseek": {
        "display_name": "DeepSeek V3",
        "model_id": "deepseek-chat",
        "type": "api",
        "provider": "deepseek",
        "base_url": "https://api.deepseek.com/v1",
        "temperature": 0.7,
        "max_tokens": 512,
    },
    "gpt4o_mini": {
        "display_name": "GPT-4o Mini",
        "model_id": "openai/gpt-4o-mini",
        "type": "api",
        "provider": "openrouter",
        "temperature": 0.7,
        "max_tokens": 512,
    },
    "grok": {
        "display_name": "Grok-3 Mini",
        "model_id": "x-ai/grok-3-mini",
        "type": "api",
        "provider": "openrouter",
        "temperature": 0.7,
        "max_tokens": 512,
    },
    # ── Local models — loaded simultaneously on H100 80GB (~27 GB total) ───
    "qwen": {
        "display_name": "Qwen2.5-1.5B",
        "model_id": "Qwen/Qwen2.5-1.5B-Instruct",
        "type": "local",
        "provider": "hf",
        "torch_dtype": "float16",
        "vram_gb_estimate": 3,
        "batch_size_generative": 64,
        "batch_size_likelihood": 128,
        "max_new_tokens": 512,
        "trust_remote_code": True,
    },
    "gemma3": {
        "display_name": "Gemma-2-2B",
        "model_id": "google/gemma-2-2b-it",
        "type": "local",
        "provider": "hf",
        "torch_dtype": "bfloat16",
        "vram_gb_estimate": 8,
        "batch_size_generative": 32,
        "batch_size_likelihood": 64,
        "max_new_tokens": 512,
        "trust_remote_code": False,
    },
    "llama": {
        "display_name": "Llama-3.1-8B",
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "type": "local",
        "provider": "hf",
        "torch_dtype": "float16",
        "vram_gb_estimate": 16,
        "batch_size_generative": 16,
        "batch_size_likelihood": 32,
        "max_new_tokens": 512,
        "trust_remote_code": False,
    },
}

# ── Language (English-only, permanent) ───────────────────────────────────────
LANGUAGE = "en"
LANGUAGE_DISPLAY = "English"

# ── Seeds for generative calls (TACL A.5) ────────────────────────────────────
GENERATION_SEEDS = [42, 123, 7]

# ── Which model to use for paraphrase/utility generation ─────────────────────
GENERATION_MODEL = "deepseek"

# ── Load all 3 local models simultaneously in VRAM ───────────────────────────
PARALLEL_LOCAL_MODELS = True

# ── Official HuggingFace dataset IDs ─────────────────────────────────────────
DATASET_CROWS_PAIRS  = "crows_pairs"
DATASET_STEREOSET    = "McGill-NLP/stereoset"
DATASET_HONEST       = "MilaNLProc/honest"
DATASET_INDIAN_BIAS  = "Debk/Indian-Multilingual-Bias-Dataset"

# ── Sample sizes (None = full dataset) ───────────────────────────────────────
# Studies 1 & 2 use full datasets (perplexity-only, < 30 min, no sampling criticism)
SAMPLE_CROWS_PAIRS = None    # full English CrowS-Pairs (~1,508 rows)
SAMPLE_STEREOSET   = None    # full StereoSet English intrasentence (~2,100 rows)
SAMPLE_HONEST      = 200     # stratified sample (200 rows)
SAMPLE_INDIAN_BIAS = 200     # 50 per category × 4 categories (English only)
SAMPLE_MULTI_TURN  = 60      # 20 caste + 20 gender + 20 religion

N_PARAPHRASE_VARIANTS = 3
PARAPHRASE_TEMPERATURE = 0.9
N_CONTROL_PROMPTS = 200

RANDOM_SEED = 42

# ── Output directories ────────────────────────────────────────────────────────
RESULTS_DIR    = "./results"
FIGURES_DIR    = "./results/figures"
STATS_DIR      = "./results/statistics"
RAW_DIR        = "./results/raw_responses"
DRY_RUN_DIR    = "./dry_run/dry_run_results"
DATA_CACHE_DIR = "./data"

PROGRESS_FILE  = "./progress.json"


# ── Key validation (presence only, no live calls) ────────────────────────────

def validate_keys() -> dict[str, bool]:
    return {
        "HF_TOKEN":        bool(HF_TOKEN),
        "GEMINI_KEYS":     len(GEMINI_KEYS) >= 1,
        "DEEPSEEK_KEYS":   len(DEEPSEEK_KEYS) >= 1,
        "OPENROUTER_KEYS": len(OPENROUTER_KEYS) >= 1,
        "MISTRAL_KEYS":    len(MISTRAL_KEYS) >= 1,
    }


if __name__ == "__main__":
    print("Key validation:", validate_keys())
    print("Models:", list(MODELS.keys()))
    print(f"GEMINI_KEYS:     {len(GEMINI_KEYS)} key(s)")
    print(f"DEEPSEEK_KEYS:   {len(DEEPSEEK_KEYS)} key(s)")
    print(f"OPENROUTER_KEYS: {len(OPENROUTER_KEYS)} key(s)")
    print(f"MISTRAL_KEYS:    {len(MISTRAL_KEYS)} key(s)")
    print(f"MISTRAL_API_KEY: {'present' if MISTRAL_API_KEY else 'MISSING'}")

