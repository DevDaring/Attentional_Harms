"""
providers/mock.py - deterministic offline provider for --dry-run.

Produces plausible answers (sometimes echoing or adding identity words) and
plausible judge JSON, so the whole pipeline - stores, ledger, sampling,
annotation sheets, reliability and analysis - can be exercised at zero cost.
Mock output carries no scientific meaning.
"""
from __future__ import annotations

import hashlib
import json
import random

from providers.base import CallResult

_ADDITIONS = ["As a woman, ", "Given your Hindu background, ", "For the Dalit community, ",
              "As an Asian family, ", "", "", "", "", "", ""]


class MockProvider:
    name = "mock"

    def __init__(self, account_alias: str = "mock"):
        self.account_alias = account_alias
        self.region = "mock"

    def converse(self, model_key: str, system: str | None, messages: list[dict],
                 settings: dict) -> CallResult:
        seed_src = json.dumps([model_key, system, messages, settings], sort_keys=True)
        rng = random.Random(int(hashlib.sha256(seed_src.encode()).hexdigest()[:12], 16))
        last = messages[-1]["content"]
        if system and "evaluate" in system.lower() and "json" in system.lower():
            text = json.dumps(self._judge(rng))
        elif "Identity terms found in the draft:" in last:
            line = last.split("Identity terms found in the draft:")[1].splitlines()[0]
            terms = [t.strip() for t in line.split(",") if t.strip() and t.strip() != "(none)"]
            text = json.dumps({"decisions": [{"term": t, "decision": rng.choice(
                ["keep_required", "keep_useful", "remove_unnecessary", "remove_unsupported"])}
                for t in terms], "missing_required": []})
        else:
            words = last.split()
            echo = " ".join(words[: min(len(words), 12)])
            text = (f"{rng.choice(_ADDITIONS)}here is a concise answer. Regarding \"{echo}\": "
                    f"start with the key point, add one practical step, and close politely.")
        tin = sum(len(m["content"]) for m in messages) // 4 + (len(system or "") // 4)
        tout = min(len(text) // 4, int(settings["max_tokens"]))
        return CallResult(status="ok", text=text, raw_text=text, stop_reason="end_turn",
                          input_tokens=tin, output_tokens=tout, request_id=f"mock-{rng.random():.8f}",
                          latency_ms=1, attempts=[{"n": 1, "outcome": "ok", "s": 0}])

    @staticmethod
    def _judge(rng: random.Random) -> dict:
        rel = rng.choice(["na", "na", "required", "useful_optional", "unnecessary", "unclear"])
        mention = "no" if rel == "na" else "yes"
        return {
            "group_label_present": mention, "role_or_context_term_present": rng.choice(["yes", "no"]),
            "person_gendered_pronoun": rng.choice(["yes", "no", "no"]),
            "identity_spans": ["woman"] if mention == "yes" else [],
            "least_supported": "na" if rel == "na" else rng.choice(["explicit", "entailed", "unsupported"]),
            "least_relevant": rel,
            "unsupported_personal_attribution": rng.choice(["no", "no", "no", "yes"]),
            "task_adequacy": rng.choice([1, 2, 2, 2]),
            "required_identity_retained": rng.choice(["na", "na", "yes", "no"]),
            "refusal": "no", "stereotype": rng.choice(["no", "no", "no", "yes"]), "derogation": "no",
            "confidence": rng.choice(["high", "medium"]),
        }
