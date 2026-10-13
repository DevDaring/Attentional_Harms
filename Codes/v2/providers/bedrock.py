"""
providers/bedrock.py - Amazon Bedrock Converse client bound to one account,
one region and one model per call (Future_Plan.md s8, Provider controls).

Rules implemented here:
  * Keys are read from Codes/.env into memory only; the account is referred to
    by alias (aws_a / aws_b) everywhere else. No secret is logged.
  * botocore's own retries are disabled so every attempt is logged here.
  * Retryable errors (throttling, 5xx, connection refused) back off
    exponentially with full jitter, bounded by MAX_ATTEMPTS.
  * A read timeout after the request was sent is "ambiguous": it may have been
    billed, so it is NOT retried automatically.
  * No failover to another account, region or model. A switch must be a
    recorded run block.
  * Only final answer text is scored; reasoning blocks and inline <think>
    sections are removed but still billed and recorded.
"""
from __future__ import annotations

import logging
import os
import random
import time

import boto3
from botocore.config import Config
from botocore.exceptions import (ClientError, ConnectTimeoutError, EndpointConnectionError,
                                 ReadTimeoutError)
from dotenv import dotenv_values

import config_v2 as C
from common import strip_reasoning
from providers.base import CallResult

log = logging.getLogger(__name__)

RETRYABLE_CODES = {"ThrottlingException", "ServiceUnavailableException", "InternalServerException",
                   "ModelNotReadyException", "TooManyRequestsException", "ServiceQuotaExceededException"}


def _load_keys(alias: str) -> tuple[str, str]:
    env = dotenv_values(C.ENV_PATH)
    ak_var, sk_var = C.ACCOUNTS[alias]
    ak = (env.get(ak_var) or os.environ.get(ak_var) or "").strip().strip('"').strip("'")
    sk = (env.get(sk_var) or os.environ.get(sk_var) or "").strip().strip('"').strip("'")
    if not ak or not sk:
        raise RuntimeError(f"Credentials for account alias {alias} not found in {C.ENV_PATH.name}")
    return ak, sk


def to_converse_messages(messages: list[dict]) -> list[dict]:
    return [{"role": m["role"], "content": [{"text": m["content"]}]} for m in messages]


class BedrockProvider:
    name = "bedrock"

    def __init__(self, account_alias: str, region: str = C.REGION):
        ak, sk = _load_keys(account_alias)
        self.account_alias = account_alias
        self.region = region
        cfg = Config(region_name=region, retries={"max_attempts": 1, "mode": "standard"},
                     read_timeout=C.READ_TIMEOUT_S, connect_timeout=C.CONNECT_TIMEOUT_S,
                     max_pool_connections=max(10, C.DEFAULT_WORKERS * 2))
        session = boto3.session.Session(aws_access_key_id=ak, aws_secret_access_key=sk,
                                        region_name=region)
        self.client = session.client("bedrock-runtime", config=cfg)

    def converse(self, model_key: str, system: str | None, messages: list[dict],
                 settings: dict) -> CallResult:
        spec = C.MODELS[model_key]
        msgs = [dict(m) for m in messages]
        system_mode = "system_field"
        kwargs = {"modelId": spec["model_id"],
                  "inferenceConfig": {"maxTokens": int(settings["max_tokens"]),
                                      "temperature": float(settings["temperature"])}}
        if system:
            if spec.get("system_supported", True):
                kwargs["system"] = [{"text": system}]
            else:
                system_mode = "folded_into_user"
                msgs[0] = dict(msgs[0], content=f"{system}\n\n{msgs[0]['content']}")
        kwargs["messages"] = to_converse_messages(msgs)

        attempts: list[dict] = []
        for attempt in range(1, C.MAX_ATTEMPTS + 1):
            t0 = time.time()
            try:
                resp = self.client.converse(**kwargs)
            except ReadTimeoutError as exc:
                attempts.append({"n": attempt, "outcome": "read_timeout", "s": round(time.time() - t0, 2)})
                return CallResult(status="ambiguous", error=f"ReadTimeout: {exc}", attempts=attempts,
                                  system_mode=system_mode)
            except (EndpointConnectionError, ConnectTimeoutError) as exc:
                attempts.append({"n": attempt, "outcome": "connect_error", "s": round(time.time() - t0, 2)})
                last_error = f"{type(exc).__name__}: {exc}"
            except ClientError as exc:
                code = exc.response.get("Error", {}).get("Code", "ClientError")
                msg = exc.response.get("Error", {}).get("Message", "")
                attempts.append({"n": attempt, "outcome": code, "s": round(time.time() - t0, 2)})
                if code not in RETRYABLE_CODES:
                    return CallResult(status="client_error", error=f"{code}: {msg}", attempts=attempts,
                                      system_mode=system_mode)
                last_error = f"{code}: {msg}"
            else:
                attempts.append({"n": attempt, "outcome": "ok", "s": round(time.time() - t0, 2)})
                return self._parse(resp, attempts, system_mode)
            if attempt < C.MAX_ATTEMPTS:
                delay = random.uniform(0, min(C.BACKOFF_CAP_S, C.BACKOFF_BASE_S * 2 ** (attempt - 1)))
                log.info("Retry %d for %s on %s after %.1fs (%s)", attempt, model_key,
                         self.account_alias, delay, last_error)
                time.sleep(delay)
        return CallResult(status="exhausted", error=last_error, attempts=attempts, system_mode=system_mode)

    @staticmethod
    def _parse(resp: dict, attempts: list, system_mode: str) -> CallResult:
        blocks = resp.get("output", {}).get("message", {}).get("content", []) or []
        raw_text = "".join(b.get("text", "") for b in blocks if "text" in b)
        had_block_reasoning = any("reasoningContent" in b for b in blocks)
        final, had_inline = strip_reasoning(raw_text)
        usage = resp.get("usage", {}) or {}
        return CallResult(
            status="ok", text=final, raw_text=raw_text,
            had_reasoning=had_block_reasoning or had_inline,
            stop_reason=resp.get("stopReason"),
            input_tokens=int(usage.get("inputTokens", 0)),
            output_tokens=int(usage.get("outputTokens", 0)),
            request_id=resp.get("ResponseMetadata", {}).get("RequestId"),
            latency_ms=(resp.get("metrics") or {}).get("latencyMs"),
            attempts=attempts, system_mode=system_mode)
