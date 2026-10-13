"""
executor.py - resumable, budget-controlled execution of request units.

* Every call is keyed (store.request_key) and skipped when already done.
* Worst-case cost is reserved in the ledger before dispatch; a BudgetExceeded
  stops all workers cleanly.
* Dialogue trajectories run turn by turn, feeding the stored final answer of
  each assistant turn back as history; a resumed trajectory rebuilds its
  history from the store, so no turn is generated twice.
"""
from __future__ import annotations

import logging
import math
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed

import config_v2 as C
from common import utc_now
from design import Unit
from ledger import BudgetExceeded, Ledger
from store import ResponseStore, request_key

log = logging.getLogger(__name__)


def estimate_input_tokens(system: str, messages: list[dict]) -> int:
    chars = len(system or "") + sum(len(m["content"]) for m in messages)
    return int(math.ceil(chars / C.CHARS_PER_TOKEN_ESTIMATE)) + \
        C.PER_MESSAGE_OVERHEAD_TOKENS * (len(messages) + 1)


class Runner:
    def __init__(self, store: ResponseStore, ledger: Ledger, providers: dict, workers: int,
                 retry_ambiguous: bool = False):
        self.store = store
        self.ledger = ledger
        self.providers = providers          # account alias -> provider
        self.workers = max(1, workers)
        self.retry_ambiguous = retry_ambiguous
        self.stop = threading.Event()
        self.stop_reason = ""
        self.stats = {"called": 0, "skipped": 0, "ok": 0, "client_error": 0, "exhausted": 0,
                      "ambiguous": 0, "trajectories_stopped": 0}
        self._lock = threading.Lock()
        # Concurrency caps apply per (model, account): each AWS account has its own quota.
        self._model_sem = {(m, a): threading.Semaphore(n) for m, n in C.MODEL_CONCURRENCY.items()
                           for a in C.ACCOUNTS}

    def _bump(self, key: str, n: int = 1) -> None:
        with self._lock:
            self.stats[key] = self.stats.get(key, 0) + n

    # ------------------------------------------------------------ one call
    def _call(self, unit: Unit, messages: list[dict], coords: dict, turn: int | None) -> dict | None:
        spec = C.MODELS[unit.model_key]
        settings = unit.settings
        key = request_key(spec["model_id"], unit.system, messages, settings, coords)
        if self.store.is_done(key, self.retry_ambiguous):
            self._bump("skipped")
            return self.store.get(key)
        if self.stop.is_set():
            return None
        provider = self.providers[unit.account]
        try:
            rid = self.ledger.reserve(unit.model_key, unit.account,
                                      estimate_input_tokens(unit.system, messages),
                                      int(settings["max_tokens"]), unit.phase)
        except BudgetExceeded as exc:
            self.stop_reason = str(exc)
            self.stop.set()
            log.error("Budget stop: %s", exc)
            return None
        sem = self._model_sem.get((unit.model_key, unit.account))
        if sem:
            with sem:
                res = provider.converse(unit.model_key, unit.system, messages, settings)
        else:
            res = provider.converse(unit.model_key, unit.system, messages, settings)
        if res.status == "ok":
            self.ledger.settle(rid, res.input_tokens, res.output_tokens, key)
        elif res.status == "ambiguous":
            self.ledger.mark_ambiguous(rid, key)
        else:
            self.ledger.release(rid, res.error or res.status)
        record = {
            "request_key": key, "status": res.status, "ts": utc_now(),
            "role": unit.role, "phase": unit.phase, "experiment": unit.experiment,
            "model_key": unit.model_key, "model_id": spec["model_id"], "model_family": spec["family"],
            "account": unit.account, "region": getattr(provider, "region", C.REGION),
            "provider": getattr(provider, "name", "?"), "system_id": unit.system_id,
            "system": unit.system, "system_mode": res.system_mode, "messages": messages,
            "settings": settings, "coords": coords, "turn": turn, **{f"m_{k}": v for k, v in unit.meta.items()},
            "text": res.text, "raw_text": res.raw_text, "had_reasoning": res.had_reasoning,
            "stop_reason": res.stop_reason, "input_tokens": res.input_tokens,
            "output_tokens": res.output_tokens, "request_id": res.request_id,
            "latency_ms": res.latency_ms, "error": res.error, "attempts": res.attempts,
        }
        self.store.append(record)
        self._bump("called")
        self._bump(res.status)
        n = self.stats["called"]
        if n % 100 == 0:
            log.info("%d calls; ledger %s", n, self.ledger.summary()["committed_budget_rate_usd"])
        return record

    # ------------------------------------------------------------ units
    def _run_method(self, unit: Unit) -> None:
        """Remedy pipeline: reuse the B0 draft, run method steps, store one 'final' record."""
        from remedies import run_method
        spec = C.MODELS[unit.model_key]
        final_coords = dict(unit.coords, model=unit.model_key, gen_settings=C.GEN_SETTINGS)
        final_key = request_key("FINAL", unit.system_id, [], {}, final_coords)
        if self.store.is_done(final_key):
            self._bump("skipped")
            return
        conv = [{"role": "user", "content": unit.user_turns[0]}]
        not_run = []

        def draft_fn():
            coords = {"experiment": "single", "family_id": unit.coords["family_id"],
                      "condition": unit.coords["condition"], "arm": "B0",
                      "draw": unit.coords["draw"], "prompt_variant": "original"}
            du = Unit("single", "single", "single", unit.model_key, unit.account, "B0",
                      unit.user_turns, coords, unit.meta)
            rec = self._call(du, conv, dict(coords), None)
            if rec is None:
                not_run.append("draft")
            return rec

        def call_fn(step, system, messages):
            settings = C.GATE_SETTINGS if step == "gate" else C.GEN_SETTINGS
            su = Unit("single", "comparators", "comparators_step", unit.model_key, unit.account,
                      f"{unit.system_id}:{step}", [messages[-1]["content"]], {}, unit.meta,
                      role="method_step", system_override=system, settings_override=settings)
            rec = self._call(su, messages, dict(unit.coords, step=step), None)
            if rec is None:
                not_run.append(step)
            return rec

        out = run_method(unit.system_id, conv, draft_fn, call_fn)
        if not_run:
            return          # stopped by budget or interruption: resume later, no final yet
        final = {
            "request_key": final_key, "status": "ok" if out["status"] == "ok" else out["status"],
            "ts": utc_now(), "role": "final", "phase": "comparators", "experiment": "comparators",
            "model_key": unit.model_key, "model_id": spec["model_id"], "model_family": spec["family"],
            "account": unit.account, "region": C.REGION, "provider": "pipeline",
            "system_id": unit.system_id, "system": None, "messages": conv, "settings": C.GEN_SETTINGS,
            "coords": dict(unit.coords), "turn": None, **{f"m_{k}": v for k, v in unit.meta.items()},
            "text": out["text"], "raw_text": out["text"], "trace": out["trace"],
            "input_tokens": 0, "output_tokens": 0, "attempts": [],
        }
        self.store.append(final)
        self._bump("finals")

    def _run_unit(self, unit: Unit) -> None:
        if unit.kind == "method":
            self._run_method(unit)
            return
        if unit.kind == "single":
            msgs = [{"role": "user", "content": unit.user_turns[0]}]
            self._call(unit, msgs, dict(unit.coords), None)
            return
        history: list[dict] = []
        for turn, user_text in enumerate(unit.user_turns):
            if self.stop.is_set():
                return
            msgs = history + [{"role": "user", "content": user_text}]
            rec = self._call(unit, msgs, dict(unit.coords, turn=turn), turn)
            if rec is None or rec["status"] != "ok":
                self._bump("trajectories_stopped")
                return
            reply = rec["text"].strip()
            if not reply:
                # Converse rejects empty assistant turns; a continuation would
                # change the condition. Record and stop this trajectory.
                self._bump("trajectories_stopped")
                log.warning("Empty reply stops trajectory %s turn %d", unit.coords, turn)
                return
            history = msgs + [{"role": "assistant", "content": reply}]

    def run(self, units: list[Unit], limit: int | None = None) -> dict:
        if limit is not None:
            units = units[:limit]
        log.info("Running %d units (%d calls max) with %d workers",
                 len(units), sum(u.n_calls for u in units), self.workers)
        with ThreadPoolExecutor(max_workers=self.workers) as pool:
            futures = [pool.submit(self._run_unit, u) for u in units]
            try:
                for f in as_completed(futures):
                    f.result()
            except KeyboardInterrupt:
                self.stop.set()
                self.stop_reason = "interrupted by user"
                raise
        out = dict(self.stats)
        out["stopped"] = self.stop.is_set()
        out["stop_reason"] = self.stop_reason
        out["ledger"] = self.ledger.summary()
        return out
