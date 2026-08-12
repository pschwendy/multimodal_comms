#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"; STAGE="${STAGE:-all}"
OUT="${OUT:-outputs/experiments/compressed_sensing}"
DEVICE="${DEVICE:-cuda:0}"; MODEL="${MODEL:-Qwen/Qwen3-4B}"
run_stage() { [[ "$STAGE" == all || "$STAGE" == "$1" ]]; }
mkdir -p "$OUT"
if run_stage assumptions; then
  "$PYTHON" -m experiments.validation.sensing_behavior --require-natural-image \
    --output "$OUT/assumptions.json"
fi
if run_stage images; then
  "$PYTHON" -m experiments.compressed_sensing.programs.demo_cs \
    --out-dir "$OUT/fourier_tv" --size "${IMAGE_SIZE:-256}" \
    --iterations "${TV_ITERATIONS:-300}"
  "$PYTHON" -m experiments.compressed_sensing.programs.eval_image_cs \
    --block "${BLOCK:-8}" --ratios ${RATIOS:-0.5 0.625 0.75} \
    --out-dir "$OUT/block_dct"
fi
if run_stage embeddings; then
  "$PYTHON" -m experiments.compressed_sensing.programs.demo_cs_embedding \
    --out-dir "$OUT/embeddings" --model "$MODEL" --device "$DEVICE"
fi
