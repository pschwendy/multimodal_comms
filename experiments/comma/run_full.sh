#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"; METHOD="${METHOD:-saliency}"
PUZZLES="${PUZZLES:-src/multimodal_comms/apps/comma/config/puzzles_small.json}"
MODELS="${MODELS:-src/multimodal_comms/apps/comma/config/deepseek_model.json}"
OUT="${REPORT_DIR:-outputs/experiments/comma}"
mkdir -p "$OUT/identity" "$OUT/$METHOD"
run_comma() {
  local method="$1" destination="$2"
  COMMA_COMPRESSOR="$method" "$PYTHON" -m multimodal_comms.apps.comma.run_comma \
    --puzzle_config "$PUZZLES" --model_config "$MODELS" --save_folder "$destination"
}
if [[ "${HEADLESS:-0}" == 1 ]]; then
  COMMA_COMPRESSOR=identity xvfb-run -a "$PYTHON" -m multimodal_comms.apps.comma.run_comma \
    --puzzle_config "$PUZZLES" --model_config "$MODELS" --save_folder "$OUT/identity"
  COMMA_COMPRESSOR="$METHOD" xvfb-run -a "$PYTHON" -m multimodal_comms.apps.comma.run_comma \
    --puzzle_config "$PUZZLES" --model_config "$MODELS" --save_folder "$OUT/$METHOD"
else
  run_comma identity "$OUT/identity"
  run_comma "$METHOD" "$OUT/$METHOD"
fi
"$PYTHON" -m experiments.comma.programs.aggregate_sweep \
  "identity:$OUT/identity" "$METHOD:$OUT/$METHOD" | tee "$OUT/summary.txt"
