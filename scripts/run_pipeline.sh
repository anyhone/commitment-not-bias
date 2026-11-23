#!/usr/bin/env bash
# Full pipeline for one model, meant to run in tmux (survives logout).
# usage: scripts/run_pipeline.sh <model> [step...]
#   available steps (default: all, in order): baseline vectors probes tests sweep report
#   extra step: test (final evaluation on the test split, to run after reviewing the sweep)
# Every step is idempotent or resumes where it stopped (the sweep skips conditions already done).
set -euo pipefail
cd "$(dirname "$0")/.."
M=${1:?model}; shift || true
STEPS=${*:-baseline vectors probes tests sweep report}
PY=.venv/bin/python
OUT=logs/${M}_pipeline.out
echo "=== $(date '+%F %T') start $M: $STEPS" >> "$OUT"
for s in $STEPS; do
  echo "=== $(date '+%F %T') step $s" >> "$OUT"
  case $s in
    baseline) $PY scripts/run_baseline.py --model "$M" ;;
    vectors)  $PY scripts/extract_vectors.py --model "$M" ;;
    probes)   $PY scripts/train_probe.py --model "$M" && $PY scripts/train_probe.py --model "$M" --labels text ;;
    tests)    STEER_TEST_MODEL="$M" $PY -m pytest tests -q -p no:cacheprovider ;;
    sweep)    $PY scripts/sweep.py --model "$M" ;;
    report)   $PY scripts/report.py --model "$M" ;;
    test)     scripts/run_test.sh "$M" ;;
    *) echo "unknown step: $s"; exit 2 ;;
  esac >> "$OUT" 2>&1
done
echo "=== $(date '+%F %T') END $M" >> "$OUT"
