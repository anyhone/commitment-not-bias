#!/usr/bin/env bash
# Last-token diagnosis (scripts/diag_last_token.py) for the three models, after their pipelines have finished.
cd "$(dirname "$0")/.."
for m in qwen llama mistral_instruct; do
  echo "=== $(date '+%F %T') diagnosis $m"
  .venv/bin/python scripts/diag_last_token.py --model "$m" || echo "=== ERROR diagnosis $m"
done
echo "=== $(date '+%F %T') DIAGNOSIS DONE"
