"""
providers/base.py - provider-neutral call result.

status values
  ok           - answer received; usage is billed usage
  client_error - request rejected before generation (not billed)
  exhausted    - retryable errors on every attempt (throttling etc.; not billed)
  ambiguous    - timeout after the request was sent; may have been billed
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field


@dataclass
class CallResult:
    status: str
    text: str = ""                    # final answer text only (reasoning removed)
    raw_text: str = ""                # all text blocks as returned
    had_reasoning: bool = False
    stop_reason: str | None = None
    input_tokens: int = 0
    output_tokens: int = 0
    request_id: str | None = None
    latency_ms: int | None = None
    error: str | None = None
    attempts: list = field(default_factory=list)
    system_mode: str = "system_field"   # or "folded_into_user"

    def to_dict(self) -> dict:
        return asdict(self)
