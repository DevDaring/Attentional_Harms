"""
models.py — API and local model wrappers.

6 models:
  API  — DeepSeek V3 (deepseek platform)
       — GPT-4o Mini (OpenRouter)
       — Grok-3 Mini  (OpenRouter)
  Local— Qwen2.5-1.5B-Instruct  (float16, ~3 GB VRAM)
       — Gemma-3-4B-IT           (bfloat16, ~8 GB VRAM)
       — Llama-3.1-8B-Instruct   (float16, ~16 GB VRAM)

All 3 local models are loaded simultaneously into H100 80GB (~27 GB total).
Parallel inference uses ThreadPoolExecutor — safe because PyTorch releases
the GIL during CUDA kernel execution.

Gemini API is judge-only (used in llm_judge.py, not here for experiments).
No bitsandbytes — models are small enough to run in native dtype.
All API wrappers retry up to 3 times with exponential backoff (2 s, 4 s, 8 s).
On 429 the key is marked exhausted and the rotator advances to the next key before
each retry. If all 4 attempts fail the wrapper returns {success: False}.
"""

import gc
import json
import logging
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Optional

import requests

import config
from config import KeyRotator

logger = logging.getLogger(__name__)

# Maximum number of retry attempts after the initial call (total attempts = _MAX_RETRIES + 1)
_MAX_RETRIES = 3


# ─────────────────────────────────────────────────────────────────────────────
# JSON parsing helpers
# ─────────────────────────────────────────────────────────────────────────────

def _parse_json_response(text: str) -> dict[str, Any]:
    """
    4-step robust JSON parser.
    1. Direct json.loads
    2. Strip markdown fences then parse
    3. Extract first { … last } then parse
    4. Return error dict
    """
    if not text:
        return {"error": "EMPTY_RESPONSE", "raw": ""}
    try:
        return json.loads(text, strict=False)
    except json.JSONDecodeError:
        pass
    stripped = re.sub(r"^```(?:json)?\s*", "", text.strip(), flags=re.MULTILINE)
    stripped = re.sub(r"```\s*$", "", stripped.strip(), flags=re.MULTILINE).strip()
    try:
        return json.loads(stripped, strict=False)
    except json.JSONDecodeError:
        pass
    start = text.find("{")
    end = text.rfind("}") + 1
    if start != -1 and end > start:
        try:
            return json.loads(text[start:end], strict=False)
        except json.JSONDecodeError:
            pass
    return {"error": "JSON_PARSE_FAILED", "raw": text[:500]}


def _extract_response_text(parsed: dict[str, Any]) -> str:
    """Extract 'response' field (with fallbacks) from a parsed JSON dict."""
    if "error" in parsed:
        return ""
    return str(parsed.get("response", parsed.get("text", parsed.get("content", ""))))


# ─────────────────────────────────────────────────────────────────────────────
# Gemini API wrapper (judge-only — called from llm_judge.py)
# ─────────────────────────────────────────────────────────────────────────────

def call_gemini_api(
    prompt: str,
    system_prompt: str,
    key_rotator: Optional[KeyRotator] = None,
    temperature: float = 0.7,
    model_id: Optional[str] = None,
) -> dict[str, Any]:
    """
    Up to _MAX_RETRIES+1 attempts with exponential backoff (2 s, 4 s, 8 s).
    On 429, marks the current key exhausted and rotates to the next before retry.
    Returns {success, text, raw_json, error}.
    """
    if key_rotator is None:
        key_rotator = config.get_gemini_rotator()
    if model_id is None:
        model_id = config.GEMINI_MODEL_ID

    endpoint = config.GEMINI_ENDPOINT.format(model_id=model_id)
    payload = {
        "systemInstruction": {"parts": [{"text": system_prompt}]},
        "contents": [{"role": "user", "parts": [{"text": prompt}]}],
        "generationConfig": {
            "temperature": temperature,
            "topP": 0.9,
            "maxOutputTokens": 512,
            "responseMimeType": "application/json",
        },
        # thinkingConfig removed: gemini-2.5-flash rejects thinkingBudget:0 with 400
    }

    for attempt in range(_MAX_RETRIES + 1):
        api_key = key_rotator.get_next_key()
        url = f"{endpoint}?key={api_key}"
        try:
            resp = requests.post(url, json=payload, timeout=30)
            if resp.status_code == 429:
                key_rotator.mark_exhausted(api_key)
                logger.warning(f"Gemini 429 (attempt {attempt + 1}/{_MAX_RETRIES + 1})")
                if attempt < _MAX_RETRIES:
                    time.sleep(2 ** (attempt + 1))
                    continue
                return {"success": False, "error": "RATE_LIMIT", "text": "", "raw_json": {}}
            if not resp.ok:
                body_snippet = resp.text[:300] if resp.text else "(empty)"
                logger.warning(f"Gemini HTTP {resp.status_code} body: {body_snippet}")
            resp.raise_for_status()
            data = resp.json()
            raw_text = (
                data.get("candidates", [{}])[0]
                .get("content", {})
                .get("parts", [{}])[0]
                .get("text", "")
            )
            parsed = _parse_json_response(raw_text)
            return {
                "success": True,
                "text": _extract_response_text(parsed),
                "raw_json": parsed,
                "raw_body": data,
            }
        except Exception as exc:
            logger.warning(f"Gemini API error attempt {attempt + 1}/{_MAX_RETRIES + 1}: {exc}")
            if attempt < _MAX_RETRIES:
                time.sleep(2 ** (attempt + 1))
                continue
            return {"success": False, "error": str(exc), "text": "", "raw_json": {}}
    return {"success": False, "error": "MAX_RETRIES_EXCEEDED", "text": "", "raw_json": {}}


# ─────────────────────────────────────────────────────────────────────────────
# DeepSeek API wrapper (experiment + judge secondary)
# ─────────────────────────────────────────────────────────────────────────────

def call_deepseek_api(
    prompt: str,
    system_prompt: str,
    key_rotator: Optional[KeyRotator] = None,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """
    Up to _MAX_RETRIES+1 attempts with exponential backoff (2 s, 4 s, 8 s).
    Rotates across DEEPSEEK_KEYS round-robin; on 429 marks key exhausted then rotates.
    JSON mode enforced via response_format.
    """
    if key_rotator is None:
        key_rotator = config.get_deepseek_rotator()

    url = "https://api.deepseek.com/v1/chat/completions"
    payload = {
        "model": "deepseek-chat",
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "top_p": 0.9,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(_MAX_RETRIES + 1):
        api_key = key_rotator.get_next_key()
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                key_rotator.mark_exhausted(api_key)
                logger.warning(f"DeepSeek 429 (attempt {attempt + 1}/{_MAX_RETRIES + 1})")
                if attempt < _MAX_RETRIES:
                    time.sleep(2 ** (attempt + 1))
                    continue
                return {"success": False, "error": "RATE_LIMIT", "text": "", "raw_json": {}}
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["choices"][0]["message"]["content"]
            parsed = _parse_json_response(raw_text)
            return {
                "success": True,
                "text": _extract_response_text(parsed),
                "raw_json": parsed,
                "raw_body": data,
            }
        except Exception as exc:
            logger.warning(f"DeepSeek API error attempt {attempt + 1}/{_MAX_RETRIES + 1}: {exc}")
            if attempt < _MAX_RETRIES:
                time.sleep(2 ** (attempt + 1))
                continue
            return {"success": False, "error": str(exc), "text": "", "raw_json": {}}
    return {"success": False, "error": "MAX_RETRIES_EXCEEDED", "text": "", "raw_json": {}}


# ─────────────────────────────────────────────────────────────────────────────
# OpenRouter wrapper — serves both GPT-4o Mini and Grok-3 Mini
# ─────────────────────────────────────────────────────────────────────────────

def call_openrouter_api(
    prompt: str,
    system_prompt: str,
    model_id: str = "openai/gpt-4o-mini",
    key_rotator: Optional[KeyRotator] = None,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """
    Up to _MAX_RETRIES+1 attempts with exponential backoff (2 s, 4 s, 8 s).
    model_id selects the backend model (gpt-4o-mini or grok-3-mini).
    Rotates across OPENROUTER_KEYS round-robin; on 429 marks key exhausted then rotates.
    """
    if key_rotator is None:
        key_rotator = config.get_openrouter_rotator()

    url = "https://openrouter.ai/api/v1/chat/completions"
    payload = {
        "model": model_id,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(_MAX_RETRIES + 1):
        api_key = key_rotator.get_next_key()
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json",
        }
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=60)
            if resp.status_code == 429:
                key_rotator.mark_exhausted(api_key)
                logger.warning(f"OpenRouter 429 ({model_id}) attempt {attempt + 1}/{_MAX_RETRIES + 1}")
                if attempt < _MAX_RETRIES:
                    time.sleep(2 ** (attempt + 1))
                    continue
                return {"success": False, "error": "RATE_LIMIT", "text": "", "raw_json": {}}
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["choices"][0]["message"]["content"]
            parsed = _parse_json_response(raw_text)
            return {
                "success": True,
                "text": _extract_response_text(parsed),
                "raw_json": parsed,
                "raw_body": data,
            }
        except Exception as exc:
            logger.warning(f"OpenRouter API error ({model_id}) attempt {attempt + 1}/{_MAX_RETRIES + 1}: {exc}")
            if attempt < _MAX_RETRIES:
                time.sleep(2 ** (attempt + 1))
                continue
            return {"success": False, "error": str(exc), "text": "", "raw_json": {}}
    return {"success": False, "error": "MAX_RETRIES_EXCEEDED", "text": "", "raw_json": {}}


# ─────────────────────────────────────────────────────────────────────────────
# Mistral API wrapper (judge tertiary only)
# ─────────────────────────────────────────────────────────────────────────────

def call_mistral_api(
    prompt: str,
    system_prompt: str,
    temperature: float = 0.0,
) -> dict[str, Any]:
    """
    Up to _MAX_RETRIES+1 attempts with exponential backoff (2 s, 4 s, 8 s).
    Rotates across MISTRAL_KEYS (up to 2 keys) round-robin. Used exclusively as Judge C in llm_judge.py.
    """
    api_key = config.get_mistral_rotator().get_next_key()
    if not api_key:
        return {"success": False, "error": "NO_MISTRAL_KEY", "text": "", "raw_json": {}}

    url = "https://api.mistral.ai/v1/chat/completions"
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    payload = {
        "model": config.MISTRAL_MODEL_NAME,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": prompt},
        ],
        "temperature": temperature,
        "max_tokens": 512,
        "response_format": {"type": "json_object"},
    }

    for attempt in range(_MAX_RETRIES + 1):
        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=30)
            if resp.status_code == 429:
                logger.warning(f"Mistral 429 (attempt {attempt + 1}/{_MAX_RETRIES + 1})")
                if attempt < _MAX_RETRIES:
                    time.sleep(2 ** (attempt + 1))
                    continue
                return {"success": False, "error": "RATE_LIMIT", "text": "", "raw_json": {}}
            resp.raise_for_status()
            data = resp.json()
            raw_text = data["choices"][0]["message"]["content"]
            parsed = _parse_json_response(raw_text)
            return {
                "success": True,
                "text": _extract_response_text(parsed),
                "raw_json": parsed,
                "raw_body": data,
            }
        except Exception as exc:
            logger.warning(f"Mistral API error attempt {attempt + 1}/{_MAX_RETRIES + 1}: {exc}")
            if attempt < _MAX_RETRIES:
                time.sleep(2 ** (attempt + 1))
                continue
            return {"success": False, "error": str(exc), "text": "", "raw_json": {}}
    return {"success": False, "error": "MAX_RETRIES_EXCEEDED", "text": "", "raw_json": {}}


# ─────────────────────────────────────────────────────────────────────────────
# Local model loading — native dtype, no bitsandbytes
# ─────────────────────────────────────────────────────────────────────────────

def _log_gpu_memory(label: str = "") -> None:
    try:
        import torch
        alloc = torch.cuda.memory_allocated() / 1e9
        reserved = torch.cuda.memory_reserved() / 1e9
        logger.info(f"GPU [{label}]: allocated={alloc:.2f} GB  reserved={reserved:.2f} GB")
    except Exception:
        pass


def load_local_model(model_name: str):
    """
    Load a local model in native float16 or bfloat16 (no bitsandbytes).
    All 3 small models fit simultaneously in H100 80GB (~27 GB total).
    Returns (model, tokenizer).
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    cfg = config.MODELS[model_name]
    model_id = cfg["model_id"]
    trust_rc = cfg.get("trust_remote_code", False)
    dtype_str = cfg.get("torch_dtype", "float16")
    dtype = torch.float16 if dtype_str == "float16" else torch.bfloat16

    logger.info(f"Loading {cfg['display_name']} ({model_id}) dtype={dtype_str} …")

    # Try Flash Attention 2, fall back to sdpa (PyTorch optimized), then eager
    for attn_impl in ("flash_attention_2", "sdpa", "eager"):
        try:
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                torch_dtype=dtype,
                attn_implementation=attn_impl,
                device_map="auto",
                trust_remote_code=trust_rc,
                token=config.HF_TOKEN,
            )
            logger.info(f"  attn_implementation={attn_impl!r}")
            break
        except Exception as exc:
            if attn_impl == "flash_attention_2":
                logger.warning(f"  Flash Attention 2 unavailable: {exc}. Falling back to sdpa.")
            elif attn_impl == "sdpa":
                logger.warning(f"  sdpa unavailable: {exc}. Falling back to eager.")
            else:
                raise

    tokenizer = AutoTokenizer.from_pretrained(
        model_id,
        trust_remote_code=trust_rc,
        token=config.HF_TOKEN,
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    _log_gpu_memory(f"after loading {cfg['display_name']}")
    return model, tokenizer


def load_all_local_models() -> dict[str, tuple]:
    """
    Load all 3 local models sequentially into VRAM.
    Returns {model_name: (model, tokenizer)}.
    Total VRAM: ~3 + 8 + 16 = ~27 GB out of 80 GB H100.
    """
    objects: dict[str, tuple] = {}
    for name in ("qwen", "gemma3", "llama"):
        try:
            model, tok = load_local_model(name)
            objects[name] = (model, tok)
        except Exception as exc:
            logger.error(f"Failed to load {name}: {exc}")
    _log_gpu_memory("after loading all local models")
    return objects


def unload_local_model(model, tokenizer=None) -> None:
    """Free GPU memory for a single model."""
    import torch
    _log_gpu_memory("before unload")
    del model
    if tokenizer is not None:
        del tokenizer
    gc.collect()
    torch.cuda.empty_cache()
    _log_gpu_memory("after unload")


def unload_all_local_models(objects: dict) -> None:
    """Unload all local models from VRAM."""
    import torch
    for name in list(objects.keys()):
        model, tok = objects.pop(name)
        try:
            del model
            del tok
        except Exception:
            pass
    gc.collect()
    torch.cuda.empty_cache()
    _log_gpu_memory("after unloading all models")


# ─────────────────────────────────────────────────────────────────────────────
# Local model inference
# ─────────────────────────────────────────────────────────────────────────────

def call_local_model(
    model,
    tokenizer,
    prompt: str,
    system_prompt: str,
    seed: int = 42,
    max_new_tokens: int = 512,
) -> dict[str, Any]:
    """
    Run a single generation with a fixed seed (reproducibility per TACL A.5).
    Returns {success, text, raw_json, raw_body}.
    """
    import torch

    torch.manual_seed(seed)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    try:
        input_ids = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_tensors="pt",
        ).to(model.device)
    except Exception:
        full_text = f"{system_prompt}\n\nUser: {prompt}\nAssistant:"
        input_ids = tokenizer(full_text, return_tensors="pt").input_ids.to(model.device)

    with torch.no_grad():
        output_ids = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    new_tokens = output_ids[0][input_ids.shape[1]:]
    raw_text = tokenizer.decode(new_tokens, skip_special_tokens=True)
    parsed = _parse_json_response(raw_text)
    return {
        "success": "error" not in parsed,
        "text": _extract_response_text(parsed),
        "raw_json": parsed,
        "raw_body": raw_text,
    }


def parallel_call_local_models(
    model_objects: dict[str, tuple],
    prompt: str,
    system_prompt: str,
    seed: int = 42,
) -> dict[str, dict[str, Any]]:
    """
    Run inference on all loaded local models concurrently.
    ThreadPoolExecutor is safe: PyTorch releases GIL during CUDA kernel execution.
    Returns {model_name: result_dict}.
    """
    results: dict[str, dict[str, Any]] = {}

    def _call(name: str) -> tuple[str, dict]:
        model, tokenizer = model_objects[name]
        cfg = config.MODELS[name]
        return name, call_local_model(
            model, tokenizer, prompt, system_prompt, seed=seed,
            max_new_tokens=cfg.get("max_new_tokens", 512),
        )

    with ThreadPoolExecutor(max_workers=len(model_objects)) as ex:
        futures = {ex.submit(_call, name): name for name in model_objects}
        for future in as_completed(futures):
            try:
                name, result = future.result()
                results[name] = result
            except Exception as exc:
                name = futures[future]
                logger.error(f"Parallel local model {name} error: {exc}")
                results[name] = {"success": False, "error": str(exc), "text": "", "raw_json": {}}

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Perplexity (used in Studies 1 & 2 for local models)
# ─────────────────────────────────────────────────────────────────────────────

def compute_perplexity(model, tokenizer, sentence: str) -> float:
    """
    Cross-entropy loss ≈ log-perplexity for a sentence under a causal LM.
    Lower = more likely. Used for SAR and SS computation (Studies 1 & 2).
    """
    import torch

    inputs = tokenizer(sentence, return_tensors="pt").to(model.device)
    with torch.no_grad():
        outputs = model(**inputs, labels=inputs["input_ids"])
    return float(outputs.loss.item())


# ─────────────────────────────────────────────────────────────────────────────
# Unified call dispatcher
# ─────────────────────────────────────────────────────────────────────────────

def call_model(
    model_name: str,
    prompt: str,
    system_prompt: str,
    seed: int = 42,
    model_obj=None,
    tokenizer_obj=None,
    key_rotator: Optional[KeyRotator] = None,
    temperature: float = 0.7,
) -> dict[str, Any]:
    """
    Route a call to the correct provider based on model config.
    For local models, model_obj and tokenizer_obj must be provided.
    """
    cfg = config.MODELS.get(model_name, {})
    provider = cfg.get("provider", "")
    model_type = cfg.get("type", "")

    if model_type == "local":
        if model_obj is None or tokenizer_obj is None:
            raise ValueError(f"model_obj and tokenizer_obj required for local model '{model_name}'")
        return call_local_model(
            model_obj, tokenizer_obj, prompt, system_prompt, seed=seed,
            max_new_tokens=cfg.get("max_new_tokens", 512),
        )

    if provider == "deepseek":
        return call_deepseek_api(
            prompt, system_prompt,
            key_rotator=key_rotator or config.get_deepseek_rotator(),
            temperature=temperature,
        )

    if provider == "openrouter":
        return call_openrouter_api(
            prompt, system_prompt,
            model_id=cfg["model_id"],
            key_rotator=key_rotator or config.get_openrouter_rotator(),
            temperature=temperature,
        )

    raise ValueError(f"Unknown model '{model_name}' or provider '{provider}'")


# ─────────────────────────────────────────────────────────────────────────────
# Model snapshot info (TACL A.10)
# ─────────────────────────────────────────────────────────────────────────────

def fetch_hf_model_sha(model_id: str) -> Optional[str]:
    """Fetch the latest commit SHA of a HuggingFace model for provenance."""
    try:
        from huggingface_hub import HfApi
        api = HfApi()
        info = api.model_info(model_id, token=config.HF_TOKEN)
        return info.sha
    except Exception as exc:
        logger.warning(f"Could not fetch HF SHA for {model_id}: {exc}")
        return None


def build_model_snapshots() -> dict:
    """Build model_snapshots.json content (TACL A.10.1)."""
    import datetime
    today = datetime.date.today().isoformat()
    snapshots = {}

    # API models
    snapshots["deepseek-chat"] = {
        "accessed": today,
        "endpoint": "https://api.deepseek.com/v1/chat/completions",
        "version_string": "deepseek-chat (v3)",
    }
    snapshots["openai/gpt-4o-mini"] = {
        "accessed": today,
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
    }
    snapshots["x-ai/grok-3-mini"] = {
        "accessed": today,
        "endpoint": "https://openrouter.ai/api/v1/chat/completions",
    }

    # Local models — record HF commit SHA
    for local_name in ("qwen", "gemma3", "llama"):
        model_id = config.MODELS[local_name]["model_id"]
        sha = fetch_hf_model_sha(model_id)
        snapshots[model_id] = {"hf_revision": sha, "accessed": today}

    return snapshots

