"""
common.py - small shared helpers: hashing, JSONL I/O, robust JSON extraction,
reasoning stripping, logging setup.
"""
from __future__ import annotations

import hashlib
import json
import logging
import re
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Iterator

_LOCKS: dict[str, threading.Lock] = {}
_LOCKS_GUARD = threading.Lock()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 16), b""):
            h.update(chunk)
    return h.hexdigest()


def stable_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def _lock_for(path: Path) -> threading.Lock:
    key = str(Path(path).resolve())
    with _LOCKS_GUARD:
        if key not in _LOCKS:
            _LOCKS[key] = threading.Lock()
        return _LOCKS[key]


def append_jsonl(path: Path, record: dict) -> None:
    """Thread-safe append with flush: progress survives a crash."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(record, ensure_ascii=False) + "\n"
    with _lock_for(path):
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(line)
            fh.flush()


def read_jsonl(path: Path) -> Iterator[dict]:
    path = Path(path)
    if not path.exists():
        return
    with open(path, "r", encoding="utf-8") as fh:
        for n, line in enumerate(fh, 1):
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                # A torn final line after a crash is skipped, never fatal.
                logging.getLogger(__name__).warning("Skipping malformed line %d in %s", n, path)


_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL | re.IGNORECASE)
_OPEN_THINK_RE = re.compile(r"^\s*<think>.*", re.DOTALL | re.IGNORECASE)


def strip_reasoning(text: str) -> tuple[str, bool]:
    """Remove <think>...</think> blocks some open models emit inline.
    Returns (final_text, had_reasoning). An unterminated <think> means the
    answer was truncated inside reasoning; final text is then empty."""
    if not text:
        return "", False
    had = bool(re.search(r"<think>", text, re.IGNORECASE))
    out = _THINK_RE.sub("", text)
    if _OPEN_THINK_RE.match(out):
        out = ""
    return out.strip(), had


def extract_json_object(text: str) -> dict | None:
    """Robust JSON extraction from model output: direct parse, fenced block,
    then first balanced {...} scan. Returns None when nothing parses."""
    if not text:
        return None
    text = text.strip()
    for candidate in (text, re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.MULTILINE).strip()):
        try:
            obj = json.loads(candidate, strict=False)
            if isinstance(obj, dict):
                return obj
        except (json.JSONDecodeError, ValueError):
            pass
    depth, start, in_str, esc = 0, None, False, False
    for i, ch in enumerate(text):
        if in_str:
            if esc:
                esc = False
            elif ch == "\\":
                esc = True
            elif ch == '"':
                in_str = False
            continue
        if ch == '"':
            in_str = True
        elif ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}" and depth:
            depth -= 1
            if depth == 0 and start is not None:
                try:
                    obj = json.loads(text[start:i + 1], strict=False)
                    if isinstance(obj, dict):
                        return obj
                except (json.JSONDecodeError, ValueError):
                    start = None
    return None


def setup_logging(log_path: Path | None = None, level: int = logging.INFO) -> None:
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_path:
        Path(log_path).parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path, encoding="utf-8"))
    logging.basicConfig(level=level, handlers=handlers, force=True,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def chunked(items: list, size: int) -> Iterable[list]:
    for i in range(0, len(items), size):
        yield items[i:i + size]
