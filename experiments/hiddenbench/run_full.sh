#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"; METHOD="${METHOD:-identity}"
CONFIG="${BENCH_CONFIG:-experiments/hiddenbench/configs/config.example.yaml}"
OUT="${REPORT_DIR:-outputs/experiments/hiddenbench}"
mkdir -p "$OUT"
common=(--config "$CONFIG" --num-tasks "${TASKS:-1}" --rounds "${ROUNDS:-1}" \
  --seed "${SEED:-42}" --no-full-profile --output "$OUT")
"$PYTHON" -m multimodal_comms.benchmarks.hiddenbench.runtime.cli run \
  "${common[@]}" --compressor identity --report-name sweep_full_identity
if [[ "$METHOD" != identity ]]; then
  # shellcheck disable=SC2206
  method_args=(${METHOD_ARGS:-})
  "$PYTHON" -m multimodal_comms.benchmarks.hiddenbench.runtime.cli run \
    "${common[@]}" --compressor "$METHOD" "${method_args[@]}" \
    --report-name "sweep_full_${METHOD}"
fi
"$PYTHON" -m experiments.hiddenbench.programs.aggregate_sweep "$OUT"
