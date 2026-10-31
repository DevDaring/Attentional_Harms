#!/usr/bin/env bash
# Locked-split execution after the protocol freeze (9 Oct 2026).
# Order: single first (its B0 answers are the drafts reused by the remedies), then the rest.
# Two passes: the second only retries calls that failed (completed calls are skipped).
set -u
cd "$(dirname "$0")"
for pass in 1 2; do
  for exp in single intervention comparators paraphrase schema dialogue; do
    echo "=== pass$pass $exp $(date '+%Y-%m-%d %H:%M:%S IST')"
    python run_all.py generate --experiment "$exp" --split locked --stage core --confirm-spend --workers 8 2>&1 \
      | grep -E '"(called|skipped|ok|exhausted|ambiguous|client_error|finals|stopped|stop_reason|committed_tariff_rate_usd)"|Traceback|GateError|Budget'
  done
done
echo "=== score $(date '+%H:%M:%S IST')"
python run_all.py score 2>&1 | tail -1
echo "=== sample main batches $(date '+%H:%M:%S IST')"
python run_all.py sample --sample representative,challenge,intervention --batch main --max-per-batch 220 2>&1 | grep -v " INFO "
echo "=== status $(date '+%H:%M:%S IST')"
python run_all.py status 2>&1 | grep -A9 "Generation calls"
echo "=== LOCKED DONE $(date '+%Y-%m-%d %H:%M:%S IST')"
