"""
ledger.py - application spend ledger (Future_Plan.md s9.4).

Before dispatch, the worst-case cost of a call (estimated input tokens plus the
full output cap, at the budget rate) is reserved. The reservation must fit
under the stage ceiling, the hard cap and the account ceiling, counting every
in-flight reservation. After the call, the reservation is settled at actual
usage. A reservation that is never settled (crash, ambiguous timeout) stays
charged at worst case. AWS budget alerts are supplementary only.

Budget rate = max(planning allowance, listed tariff). Tariff-rate cost is also
recorded so reports can show expected real spend next to the control figure.
"""
from __future__ import annotations

import itertools
import json
import threading
from pathlib import Path

from common import append_jsonl, read_jsonl, utc_now


class BudgetExceeded(RuntimeError):
    pass


def load_prices(path: Path) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def rates_for(prices: dict, model_key: str) -> tuple[dict, dict]:
    """Return (budget_rate, tariff_rate), each {'input','output'} in US$ per M tokens."""
    allowance = prices["allowance"]
    tariff = prices["models"].get(model_key, {})
    t_in, t_out = tariff.get("input"), tariff.get("output")
    budget = {"input": max(allowance["input"], t_in or 0.0),
              "output": max(allowance["output"], t_out or 0.0)}
    tariff_rate = {"input": t_in if t_in is not None else allowance["input"],
                   "output": t_out if t_out is not None else allowance["output"]}
    return budget, tariff_rate


def cost(rate: dict, tokens_in: float, tokens_out: float) -> float:
    return tokens_in / 1e6 * rate["input"] + tokens_out / 1e6 * rate["output"]


class Ledger:
    def __init__(self, path: Path, prices: dict, stage: str, stage_ceilings: dict,
                 hard_cap: float, account_ceilings: dict):
        if stage not in stage_ceilings:
            raise ValueError(f"Unknown stage {stage!r}; choose from {list(stage_ceilings)}")
        self.path = Path(path)
        self.prices = prices
        self.stage = stage
        self.ceiling = min(stage_ceilings[stage], hard_cap)
        self.hard_cap = hard_cap
        self.account_ceilings = account_ceilings
        self._lock = threading.Lock()
        self._ids = itertools.count(1)
        self.committed_total = 0.0
        self.committed_account: dict[str, float] = {}
        self.committed_model: dict[str, float] = {}
        self.committed_phase: dict[str, float] = {}
        self.tariff_total = 0.0
        self.tariff_model: dict[str, float] = {}
        self.inflight: dict[str, dict] = {}
        self._replay()

    # ------------------------------------------------------------ replay
    def _replay(self) -> None:
        open_res: dict[str, dict] = {}
        max_id = 0
        for ev in read_jsonl(self.path):
            rid = ev.get("res_id", "")
            try:
                max_id = max(max_id, int(str(rid).rsplit("-", 1)[-1]))
            except ValueError:
                pass
            kind = ev["event"]
            if kind == "reserve":
                open_res[rid] = ev
            elif kind in ("settle", "ambiguous", "orphan"):
                open_res.pop(rid, None)
                self._add(ev, ev["budget_cost"], ev.get("tariff_cost", 0.0))
            elif kind == "release":
                open_res.pop(rid, None)
        # Unsettled reservations from a crashed run stay charged at worst case.
        for rid, ev in open_res.items():
            orphan = dict(ev, event="orphan", ts=utc_now(),
                          note="unsettled at restart; charged worst case")
            append_jsonl(self.path, orphan)
            self._add(ev, ev["budget_cost"], ev.get("tariff_cost", 0.0))
        self._ids = itertools.count(max_id + 1)

    def _add(self, ev: dict, budget_cost: float, tariff_cost: float) -> None:
        acct, model, phase = ev["account"], ev["model"], ev.get("phase", "?")
        self.committed_total += budget_cost
        self.committed_account[acct] = self.committed_account.get(acct, 0.0) + budget_cost
        self.committed_model[model] = self.committed_model.get(model, 0.0) + budget_cost
        self.committed_phase[phase] = self.committed_phase.get(phase, 0.0) + budget_cost
        self.tariff_total += tariff_cost
        self.tariff_model[model] = self.tariff_model.get(model, 0.0) + tariff_cost

    def _inflight_total(self, account: str | None = None) -> float:
        return sum(r["budget_cost"] for r in self.inflight.values()
                   if account is None or r["account"] == account)

    # ------------------------------------------------------------ API
    def reserve(self, model_key: str, account: str, est_in: int, max_out: int, phase: str) -> str:
        budget_rate, tariff_rate = rates_for(self.prices, model_key)
        worst = cost(budget_rate, est_in, max_out)
        with self._lock:
            inflight = self._inflight_total()
            if self.committed_total + inflight + worst > self.ceiling:
                raise BudgetExceeded(
                    f"Stage '{self.stage}' ceiling ${self.ceiling:.2f} would be exceeded "
                    f"(committed ${self.committed_total:.4f}, in flight ${inflight:.4f}, "
                    f"next call worst case ${worst:.5f}).")
            acct_after = (self.committed_account.get(account, 0.0)
                          + self._inflight_total(account) + worst)
            if acct_after > self.account_ceilings.get(account, self.hard_cap):
                raise BudgetExceeded(f"Account ceiling for {account} would be exceeded.")
            rid = f"r{utc_now()[:10]}-{next(self._ids)}"
            ev = {"ts": utc_now(), "event": "reserve", "res_id": rid, "model": model_key,
                  "account": account, "phase": phase, "est_in": est_in, "max_out": max_out,
                  "budget_cost": worst, "tariff_cost": cost(tariff_rate, est_in, max_out)}
            self.inflight[rid] = ev
        append_jsonl(self.path, ev)
        return rid

    def settle(self, rid: str, tokens_in: int, tokens_out: int, request_key: str = "") -> float:
        with self._lock:
            res = self.inflight.pop(rid)
            budget_rate, tariff_rate = rates_for(self.prices, res["model"])
            b = cost(budget_rate, tokens_in, tokens_out)
            t = cost(tariff_rate, tokens_in, tokens_out)
            self._add(res, b, t)
        append_jsonl(self.path, {"ts": utc_now(), "event": "settle", "res_id": rid,
                                 "model": res["model"], "account": res["account"],
                                 "phase": res["phase"], "in": tokens_in, "out": tokens_out,
                                 "budget_cost": b, "tariff_cost": t, "request_key": request_key})
        return b

    def release(self, rid: str, reason: str) -> None:
        """The call was not billed (validation error, throttled, connection refused)."""
        with self._lock:
            res = self.inflight.pop(rid)
        append_jsonl(self.path, {"ts": utc_now(), "event": "release", "res_id": rid,
                                 "model": res["model"], "account": res["account"],
                                 "phase": res["phase"], "reason": reason})

    def mark_ambiguous(self, rid: str, request_key: str = "") -> None:
        """Timeout after send: may have been billed, so worst case is charged permanently."""
        with self._lock:
            res = self.inflight.pop(rid)
            self._add(res, res["budget_cost"], res["tariff_cost"])
        append_jsonl(self.path, dict(res, event="ambiguous", ts=utc_now(), request_key=request_key))

    def summary(self) -> dict:
        r4 = lambda d: {k: round(v, 4) for k, v in sorted(d.items())}  # noqa: E731
        return {"stage": self.stage, "stage_ceiling_usd": self.ceiling, "hard_cap_usd": self.hard_cap,
                "committed_budget_rate_usd": round(self.committed_total, 4),
                "committed_tariff_rate_usd": round(self.tariff_total, 4),
                "by_account": r4(self.committed_account), "by_model_budget": r4(self.committed_model),
                "by_model_tariff": r4(self.tariff_model), "by_phase": r4(self.committed_phase),
                "in_flight_usd": round(self._inflight_total(), 4)}
