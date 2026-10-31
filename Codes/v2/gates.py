"""
gates.py - decision gates recorded in manifests/gates.json (Future_Plan.md s6, s7).

protocol_frozen      set after the pilot, before any locked-split generation.
                     Stores SHA-256 hashes of protocol.md, both banks, config_v2.py
                     and prices.json. Locked runs refuse to start if any hash
                     changed (no tuning on locked families).
evaluator_validated  set after the evaluator audit meets the frozen targets
                     (or with an explicit, recorded override). Required before
                     production judging.
"""
from __future__ import annotations

import json
from pathlib import Path

import config_v2 as C
from common import sha256_file, utc_now

FROZEN_FILES = {"protocol": C.PROTOCOL_FILE, "prompt_bank": C.PROMPT_BANK,
                "dialogue_bank": C.DIALOGUE_BANK, "config": C.V2_DIR / "config_v2.py",
                "prices": C.PRICES_FILE, "remedies": C.V2_DIR / "remedies.py",
                "judges": C.V2_DIR / "judges.py", "lexicon": C.V2_DIR / "lexicon_v2.py",
                "label_schema": C.ANNOTATION_DIR / "schema.py"}


class GateError(RuntimeError):
    pass


def load_gates(path: Path = C.GATES_FILE) -> dict:
    if not Path(path).exists():
        return {}
    with open(path, "r", encoding="utf-8") as fh:
        return json.load(fh)


def save_gates(gates: dict, path: Path = C.GATES_FILE) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(gates, fh, indent=2)


def current_hashes() -> dict:
    return {k: (sha256_file(p) if Path(p).exists() else None) for k, p in FROZEN_FILES.items()}


def set_gate(name: str, note: str, evidence: dict | None = None, path: Path = C.GATES_FILE) -> dict:
    gates = load_gates(path)
    entry = {"set_at": utc_now(), "note": note, "evidence": evidence or {}}
    if name in ("protocol_frozen", "followup_exploratory"):
        if name == "protocol_frozen" and name in gates:
            raise GateError("protocol_frozen is already set; it is never overwritten. Use followup_exploratory.")
        entry["hashes"] = current_hashes()
    gates[name] = entry
    save_gates(gates, path)
    return entry


def require_protocol_frozen(path: Path = C.GATES_FILE) -> None:
    gates = load_gates(path)
    if "protocol_frozen" not in gates:
        raise GateError("Locked-split runs need the protocol_frozen gate. Finish the pilot, "
                        "freeze protocol.md and the banks, then run: python run_all.py gate --set "
                        "protocol_frozen --note \"...\"")
    frozen = gates["protocol_frozen"]["hashes"]
    changed = [k for k, h in current_hashes().items() if frozen.get(k) != h]
    if changed:
        raise GateError(f"Frozen files changed since the protocol was frozen: {changed}. "
                        "Record a deviation in protocol.md and re-freeze, or restore the files.")


def require_followup_exploratory(path: Path = C.GATES_FILE) -> None:
    """Post-hoc exploratory follow-up runs (review 24 Oct 2026). Never re-freezes the original
    protocol: requires the original protocol_frozen record AND a separate followup_exploratory
    record whose hashes match the current files. Results are exploratory, not confirmatory."""
    gates = load_gates(path)
    if "protocol_frozen" not in gates:
        raise GateError("protocol_frozen record missing.")
    if "followup_exploratory" not in gates:
        raise GateError("Set the followup_exploratory gate (records current hashes and the reason).")
    snap = gates["followup_exploratory"]["hashes"]
    changed = [k for k, h in current_hashes().items() if snap.get(k) != h]
    if changed:
        raise GateError(f"Files changed since the followup_exploratory record: {changed}.")


def require_evaluator_validated(path: Path = C.GATES_FILE) -> None:
    if "evaluator_validated" not in load_gates(path):
        raise GateError("Production judging needs the evaluator_validated gate. Run "
                        "'judge-calibrate' and 'audit' first, then set the gate.")
