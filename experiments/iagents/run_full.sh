#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"; DATASET="${DATASET:?Set DATASET to offline evaluation JSONL}"
METHOD="${METHOD:-saliency}"; OUT="${REPORT_DIR:-outputs/experiments/iagents}"
mkdir -p "$OUT"
common=(--dataset "$DATASET" --backend "${BACKEND:-deepseek}" \
  --max-round "${MAX_ROUND:-2}" --limit "${LIMIT:-0}")
"$PYTHON" -m experiments.iagents.programs.run_offline_eval "${common[@]}" \
  --condition identity --out "$OUT/identity.json"
# shellcheck disable=SC2206
method_params=(${METHOD_PARAMS:-})
"$PYTHON" -m experiments.iagents.programs.run_offline_eval "${common[@]}" \
  --condition "$METHOD" "${method_params[@]}" --out "$OUT/$METHOD.json"
"$PYTHON" -m experiments.iagents.programs.regrade \
  "$OUT/identity.json" "$OUT/$METHOD.json" --out "$OUT/regrade.json"
"$PYTHON" -m experiments.iagents.programs.aggregate_sweep \
  "$OUT/identity.json" "$OUT/$METHOD.json" --out "$OUT/summary.md"
