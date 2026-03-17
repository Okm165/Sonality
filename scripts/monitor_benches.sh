#!/usr/bin/env bash
# monitor_benches.sh — live bench progress monitor
# Usage:  bash scripts/monitor_benches.sh [log_file]
#
# If no log file given, auto-detects the most recent bench log in data/bench_runs/.
# Displays: phase, current pack/step, pass/fail counts, ESS defaults, elapsed time.
# Press Ctrl-C to exit; re-run at any time to see the latest snapshot.

set -euo pipefail
cd "$(dirname "$0")/.."

LOG="${1:-}"
if [[ -z "$LOG" ]]; then
    # Prefer the symlink pointing to the current full run
    if [[ -L data/bench_runs/all_benches_latest.log ]]; then
        LOG=$(readlink -f data/bench_runs/all_benches_latest.log)
    else
        LOG=$(ls -t data/bench_runs/*.log 2>/dev/null | head -1)
    fi
fi
if [[ -z "$LOG" || ! -f "$LOG" ]]; then
    echo "No bench log found. Start benches first: make bench-teaching-rapid"
    exit 1
fi

echo "Monitoring: $LOG"
echo "Updated every 5s. Press Ctrl-C to exit."
echo ""

while true; do
    clear
    echo "═══════════════════════════════════════════════════════════════"
    echo "  BENCH MONITOR  —  $(date '+%Y-%m-%d %H:%M:%S')"
    echo "  Log: $LOG  ($(wc -l < "$LOG") lines)"
    echo "═══════════════════════════════════════════════════════════════"

    # Current phase
    PHASE=$(grep -E '^\[20[0-9-].*\] === PHASE' "$LOG" 2>/dev/null | tail -1 || echo "  starting...")
    echo "  $PHASE"

    # Teaching bench pack progress
    PACK_DONE=$(grep -oE 'pack [0-9]+/70 done' "$LOG" 2>/dev/null | tail -1 || echo "")
    PACK_PASS=$(grep -c 'pass_rate=100%' "$LOG" 2>/dev/null || echo 0)
    PACK_FAIL=$(grep -c 'hard_failures=[1-9]' "$LOG" 2>/dev/null || echo 0)
    [[ -n "$PACK_DONE" ]] && echo "  Teaching: last $PACK_DONE | 100%-packs=$PACK_PASS hard-fail-packs=$PACK_FAIL"

    # Step-level pass/fail counts
    STEP_PASS=$(grep -c 'status=pass' "$LOG" 2>/dev/null || echo 0)
    STEP_FAIL=$(grep -c 'status=fail' "$LOG" 2>/dev/null || echo 0)
    echo "  Step results: passed=$STEP_PASS  failed=$STEP_FAIL"

    # Knowledge/psych test results
    PYTEST_PASS=$(grep -c ' PASSED' "$LOG" 2>/dev/null || echo 0)
    PYTEST_FAIL=$(grep -c ' FAILED' "$LOG" 2>/dev/null || echo 0)
    echo "  Pytest tests: PASSED=$PYTEST_PASS  FAILED=$PYTEST_FAIL"

    # ESS defaults
    ESS_DEF=$(grep -c 'ess.*default\|ESS.*default\|defaulted_fields' "$LOG" 2>/dev/null || echo 0)
    echo "  ESS default events: $ESS_DEF"

    # Latest teaching pack line
    LAST_PACK=$(grep '\[teaching-bench\].*pack' "$LOG" 2>/dev/null | tail -3 || echo "(none yet)")
    echo ""
    echo "  Last teaching activity:"
    echo "$LAST_PACK" | while IFS= read -r line; do echo "    $line"; done

    # Current knowledge test
    LAST_KN=$(grep 'PASSED\|FAILED\|test_k[0-9]' "$LOG" 2>/dev/null | tail -5 || echo "(none yet)")
    if [[ -n "$LAST_KN" ]]; then
        echo ""
        echo "  Last test results:"
        echo "$LAST_KN" | while IFS= read -r line; do echo "    $line"; done
    fi

    # Errors
    ERRORS=$(grep -c 'ERROR\|FAILED\|AssertionError\|RuntimeError' "$LOG" 2>/dev/null || echo 0)
    echo ""
    echo "  Total error/failure lines: $ERRORS"

    # Last 5 lines of log
    echo ""
    echo "  ── Last log lines ──"
    tail -6 "$LOG" | while IFS= read -r line; do echo "    $line"; done

    # Running PID check
    if pgrep -f "bench_runs/all_benches" > /dev/null 2>&1 || pgrep -f "test_teaching_suite_live\|test_knowledge_acquisition" > /dev/null 2>&1; then
        echo ""
        echo "  Status: ⠿ RUNNING"
    elif grep -q "ALL BENCHES COMPLETE" "$LOG" 2>/dev/null; then
        echo ""
        echo "  Status: ✓ COMPLETE"
        break
    else
        echo ""
        echo "  Status: ? UNKNOWN (process may have finished or crashed)"
    fi

    sleep 5
done
