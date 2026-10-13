"""
monitor_secondary.py - hourly correctness audit for the secondary judging run (started 10 Oct 2026,
18:48 IST; 72,000 judge calls expected). Prints a short report; exit code 1 when a check fails.

Checks: liveness (process, log advance), errors since launch, progress and ETA (IST), semantic
correctness of the judgments (parse validity, no self-family judging, two judges per target, label
rates in a plausible range, no duplicate target-judge pairs), and spend against the budget.
"""
from __future__ import annotations

import json
import re
import subprocess
import sys
from collections import Counter
from datetime import datetime, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.stdout.reconfigure(encoding="utf-8")
import config_v2 as C  # noqa: E402
import judges as J  # noqa: E402
from common import read_jsonl  # noqa: E402

EXPECTED = 72000
START = datetime(2026, 10, 10, 18, 48)
V2 = Path(__file__).resolve().parent
STATE = V2 / "outputs" / "monitor_state.json"
problems = []

# liveness
procs = subprocess.run(["tasklist", "/FI", "IMAGENAME eq python.exe"], capture_output=True, text=True).stdout
alive = procs.count("python.exe") >= 2           # this monitor + the judging run
log = (V2 / "outputs" / "run.log").read_text(encoding="utf-8", errors="replace").splitlines()
last_line = log[-1] if log else ""
last_ts = last_line[:19]
prev = json.loads(STATE.read_text()) if STATE.exists() else {}

# errors since launch
since = [l for l in log if l[:16] >= START.strftime("%Y-%m-%d %H:%M")]
errs = [l for l in since if re.search(r"Traceback|ERROR|Budget stop|client_error|GateError", l)]
retries = sum(1 for l in since if "Retry" in l)

# judgments
latest = {}
for r in read_jsonl(V2 / "outputs" / "judgments.jsonl"):
    latest[r["request_key"]] = r
sec = [r for r in latest.values() if (r.get("coords") or {}).get("purpose") == "secondary"]
ok = [r for r in sec if r["status"] == "ok"]
bad_status = Counter(r["status"] for r in sec if r["status"] != "ok")
gen = {}
for r in read_jsonl(V2 / "outputs" / "generations.jsonl"):
    gen[r["request_key"]] = r
jf = J.judgments_frame(ok) if ok else None
valid = float(jf["valid"].mean()) if jf is not None else float("nan")
self_family = sum(1 for r in ok if gen.get(r["coords"]["target"]) and
                  C.MODELS[r["model_key"]]["family"] == gen[r["coords"]["target"]]["model_family"])
pairs = Counter((r["coords"]["target"], r["model_key"]) for r in ok)
dups = sum(1 for v in pairs.values() if v > 1)
per_target = Counter(r["coords"]["target"] for r in ok)
over2 = sum(1 for v in per_target.values() if v > 2)
yes_rate = float((jf["unnecessary"] == "yes").mean()) if jf is not None else float("nan")
by_judge = Counter(r["model_key"] for r in ok)

# progress and ETA
now = datetime.now()
done = len(ok)
rate = done / max(1.0, (now - START).total_seconds() / 60)
eta = now + timedelta(minutes=(EXPECTED - done) / rate) if rate > 0 else None
advanced = done > prev.get("done", -1)

# spend
ledger = [json.loads(l) for l in open(C.MANIFEST_DIR / "ledger.jsonl", encoding="utf-8") if l.strip()]
tariff = sum(e.get("tariff_cost", 0) for e in ledger if e["event"] in ("settle", "ambiguous", "orphan"))

if not alive and done < EXPECTED:
    problems.append("judging process not running before completion")
if prev and not advanced and done < EXPECTED:
    problems.append("no new judgments since the last check")
if errs:
    problems.append(f"{len(errs)} error lines since launch")
if valid == valid and valid < 0.98:
    problems.append(f"parse validity {valid:.3f} < 0.98")
if self_family:
    problems.append(f"{self_family} self-family judgments")
if dups or over2:
    problems.append(f"duplicates: {dups} target-judge pairs, {over2} targets with >2 judgments")
if yes_rate == yes_rate and not (0.01 <= yes_rate <= 0.40):
    problems.append(f"'unnecessary' rate {yes_rate:.3f} outside plausible 1-40%")
if tariff > 120:
    problems.append(f"spend ${tariff:.2f} above the $120 alert level")

print(f"[{now:%Y-%m-%d %H:%M} IST] secondary judging: {done:,}/{EXPECTED:,} ok ({100 * done / EXPECTED:.1f}%), "
      f"{rate:.0f}/min, ETA {eta:%a %d %b %H:%M IST}" if eta else "ETA n/a")
print(f"  alive={alive} log_last={last_ts} non-ok={dict(bad_status)} retries_since_launch={retries}")
print(f"  valid={valid:.3f} self_family={self_family} dup_pairs={dups} unnecessary_rate={yes_rate:.3f} by_judge={dict(by_judge)}")
print(f"  spend so far (listed prices) ${tariff:.2f}")
print("  STATUS:", "OK" if not problems else "PROBLEM -> " + "; ".join(problems))
STATE.write_text(json.dumps({"done": done, "checked": now.isoformat()}))
sys.exit(1 if problems else 0)
