#!/usr/bin/env bash
# Secondary automated judging (evaluator gate NOT met; see deviations_post_freeze.md).
set -u
cd "$(dirname "$0")"
echo "=== judge-secondary $(date '+%Y-%m-%d %H:%M:%S IST')"
python run_all.py judge-secondary --stage core --confirm-spend --workers 32 2>&1 | grep -E 'secondary judging|"(called|skipped|ok|exhausted|client_error|ambiguous|stopped|stop_reason|committed_tariff_rate_usd)"|Traceback|Budget'
echo "=== retry pass $(date '+%H:%M:%S IST')"
python run_all.py judge-secondary --stage core --confirm-spend --workers 6 2>&1 | grep -E '"(called|ok|exhausted)"|Traceback'
echo "=== third votes $(date '+%H:%M:%S IST')"
python run_all.py judge-third --secondary --stage core --confirm-spend --workers 8 2>&1 | grep -E 'needing_third|cap|scheduled|over_cap|"(called|ok|exhausted)"|Traceback'
echo "=== consensus $(date '+%H:%M:%S IST')"
python run_all.py consensus 2>&1 | tail -6
echo "=== SECONDARY DONE $(date '+%Y-%m-%d %H:%M:%S IST')"
