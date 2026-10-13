"""
store.py - append-only response store with resume by request key.

Each request is identified by a SHA-256 over model, system prompt, full
messages, settings and the design coordinates (experiment, family, condition,
arm, draw, trajectory, turn). The draw index is part of the key, so repeated
draws are distinct requests (Future_Plan.md s8, Provider controls).
"""
from __future__ import annotations

import threading
from pathlib import Path
from typing import Iterator

import pandas as pd

from common import append_jsonl, read_jsonl, sha256_text, stable_json

FINAL_STATUSES = {"ok"}


def request_key(model_id: str, system: str | None, messages: list[dict],
                settings: dict, coords: dict) -> str:
    payload = {"model_id": model_id, "system": system, "messages": messages,
               "settings": settings, "coords": coords}
    return sha256_text(stable_json(payload))


class ResponseStore:
    """Latest record per request_key wins. Records are never rewritten."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self._lock = threading.Lock()
        self._latest: dict[str, dict] = {}
        for rec in read_jsonl(self.path):
            self._latest[rec["request_key"]] = rec

    def __len__(self) -> int:
        return len(self._latest)

    def get(self, key: str) -> dict | None:
        return self._latest.get(key)

    def is_done(self, key: str, retry_ambiguous: bool = False) -> bool:
        rec = self._latest.get(key)
        if rec is None:
            return False
        if rec["status"] in FINAL_STATUSES:
            return True
        if rec["status"] == "ambiguous" and not retry_ambiguous:
            # An ambiguous timeout may have been billed: never re-dispatch blindly.
            return True
        return False

    def append(self, record: dict) -> None:
        append_jsonl(self.path, record)
        with self._lock:
            self._latest[record["request_key"]] = record

    def records(self, status: str | None = "ok") -> Iterator[dict]:
        for rec in list(self._latest.values()):
            if status is None or rec["status"] == status:
                yield rec

    def to_frame(self, status: str | None = "ok") -> pd.DataFrame:
        rows = []
        for r in self.records(status):
            flat = {k: v for k, v in r.items()
                    if k not in ("messages", "raw_blocks", "attempts", "coords")}
            flat.update({f"c_{k}": v for k, v in (r.get("coords") or {}).items()})
            rows.append(flat)
        return pd.DataFrame(rows)

    def export_csv(self, path: Path, status: str | None = "ok") -> int:
        df = self.to_frame(status)
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(path, index=False, encoding="utf-8-sig")
        return len(df)
