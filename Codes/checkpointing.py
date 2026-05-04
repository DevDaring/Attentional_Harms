"""
checkpointing.py — Progress tracking and crash-safe CSV resume logic.

Maintains progress.json at project root. Provides helpers for:
  - Reading / writing progress
  - Appending rows to CSV in crash-safe mode
  - Generating composite row keys for deduplication
  - Loading already-processed rows so studies can skip them
"""

import json
import os
import time
import logging
from pathlib import Path
from typing import Any, Optional

import pandas as pd

logger = logging.getLogger(__name__)

PROGRESS_FILE = "./progress.json"

# ─────────────────────────────────────────────────────────────────────────────
# Default progress template
# ─────────────────────────────────────────────────────────────────────────────

DEFAULT_PROGRESS: dict[str, Any] = {
    "environment_verified": False,
    "dry_run_completed": False,
    "data_loaded": False,
    "paraphrases_generated": False,
    "control_prompts_built": False,
    "fertility_computed": False,
    "studies": {
        "crows_pairs": {
            "completed": False,
            "rows_processed": 0,
            "output_file": "./results/study1_crows_pairs.csv",
        },
        "stereoset": {
            "completed": False,
            "rows_processed": 0,
            "output_file": "./results/study2_stereoset.csv",
        },
        "generative_salience": {
            "completed": False,
            "rows_processed": 0,
            "output_file": "./results/study3_generative_salience.csv",
        },
        "multi_turn": {
            "completed": False,
            "rows_processed": 0,
            "output_file": "./results/study4_multi_turn.csv",
        },
    },
    "paraphrase_robustness": {"completed": False, "output_file": "./results/paraphrase_robustness.csv"},
    "control_results": {"completed": False, "output_file": "./results/control_results.csv"},
    "metrics_computed": False,
    "statistics_done": False,
    "figures_generated": False,
    "report_generated": False,
    "timestamp_started": "",
    "timestamp_last_update": "",
}


# ─────────────────────────────────────────────────────────────────────────────
# Read / write
# ─────────────────────────────────────────────────────────────────────────────

def load_progress(path: str = PROGRESS_FILE) -> dict[str, Any]:
    """Load progress.json, creating a fresh one if absent."""
    if not os.path.exists(path):
        progress = DEFAULT_PROGRESS.copy()
        progress["timestamp_started"] = _ts()
        save_progress(progress, path)
        return progress
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_progress(progress: dict[str, Any], path: str = PROGRESS_FILE) -> None:
    """Write progress.json atomically (write to tmp, rename)."""
    progress["timestamp_last_update"] = _ts()
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(progress, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def mark_phase_complete(phase_key: str, path: str = PROGRESS_FILE) -> None:
    """Set a top-level boolean phase to True."""
    p = load_progress(path)
    p[phase_key] = True
    save_progress(p, path)


def mark_study_complete(study_key: str, path: str = PROGRESS_FILE) -> None:
    """Mark a study as completed in progress.json."""
    p = load_progress(path)
    p["studies"][study_key]["completed"] = True
    save_progress(p, path)


def update_study_row_count(study_key: str, count: int, path: str = PROGRESS_FILE) -> None:
    p = load_progress(path)
    p["studies"][study_key]["rows_processed"] = count
    save_progress(p, path)


def is_phase_complete(phase_key: str, path: str = PROGRESS_FILE) -> bool:
    p = load_progress(path)
    return bool(p.get(phase_key, False))


def is_study_complete(study_key: str, path: str = PROGRESS_FILE) -> bool:
    p = load_progress(path)
    return bool(p.get("studies", {}).get(study_key, {}).get("completed", False))


# ─────────────────────────────────────────────────────────────────────────────
# CSV helpers
# ─────────────────────────────────────────────────────────────────────────────

def make_row_key(prompt_id: Any, model_name: str, language: str, turn_number: int = 1) -> str:
    """Create a composite key for deduplication across study CSVs."""
    return f"{prompt_id}_{model_name}_{language}_{turn_number}"


def append_rows_to_csv(df: pd.DataFrame, path: str, encoding: str = "utf-8-sig") -> None:
    """
    Append DataFrame rows to a CSV file. Creates file (with header) on first write.
    Uses UTF-8 BOM (utf-8-sig) for proper Indic script display in Excel.
    """
    file_exists = os.path.exists(path) and os.path.getsize(path) > 0
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    df.to_csv(
        path,
        mode="a",
        header=not file_exists,
        index=False,
        encoding=encoding,
    )


def load_processed_keys(csv_path: str) -> set[str]:
    """
    Load already-processed row_keys from an existing CSV output file.
    Returns empty set if file does not exist.
    """
    if not os.path.exists(csv_path):
        return set()
    try:
        df = pd.read_csv(csv_path, encoding="utf-8-sig", usecols=["row_key"])
        return set(df["row_key"].astype(str).tolist())
    except Exception as exc:
        logger.warning(f"Could not load processed keys from {csv_path}: {exc}")
        return set()


def should_skip_row(row_key: str, processed_keys: set[str]) -> bool:
    return row_key in processed_keys


# ─────────────────────────────────────────────────────────────────────────────
# Provenance helpers (TACL A.10)
# ─────────────────────────────────────────────────────────────────────────────

def compute_sha256(file_path: str) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def record_provenance(dataset_name: str, file_path: Optional[str], sha256: Optional[str] = None) -> None:
    provenance_path = "./results/provenance.json"
    os.makedirs("./results", exist_ok=True)
    if os.path.exists(provenance_path):
        with open(provenance_path, "r", encoding="utf-8") as f:
            prov = json.load(f)
    else:
        prov = {}
    prov[dataset_name] = {
        "file_path": file_path,
        "sha256": sha256 or (compute_sha256(file_path) if file_path and os.path.exists(file_path) else None),
        "timestamp": _ts(),
    }
    with open(provenance_path, "w", encoding="utf-8") as f:
        json.dump(prov, f, indent=2, ensure_ascii=False)


# ─────────────────────────────────────────────────────────────────────────────
# Raw response persistence (TACL A.10)
# ─────────────────────────────────────────────────────────────────────────────

def save_raw_response(model_name: str, prompt_id: str, seed: int, response_body: Any) -> None:
    """Persist the full API response body to disk (TACL A.10.2)."""
    dir_path = f"./results/raw_responses/{model_name}"
    os.makedirs(dir_path, exist_ok=True)
    file_path = f"{dir_path}/{prompt_id}_seed{seed}.json"
    with open(file_path, "w", encoding="utf-8") as f:
        json.dump(response_body, f, ensure_ascii=False, default=str)


# ─────────────────────────────────────────────────────────────────────────────
# Utilities
# ─────────────────────────────────────────────────────────────────────────────

def _ts() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def ensure_dirs() -> None:
    """Create all required output directories."""
    dirs = [
        "./results",
        "./results/figures",
        "./results/statistics",
        "./results/raw_responses",
    ]
    for d in dirs:
        os.makedirs(d, exist_ok=True)
