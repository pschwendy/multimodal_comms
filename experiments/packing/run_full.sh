#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"; STAGE="${STAGE:-all}"; DEVICE="${DEVICE:-cuda:0}"
DATA_DIR="${DATA_DIR:-outputs/data/fineweb_ae}"
BASE_CHECKPOINT="${BASE_CHECKPOINT:-outputs/models/autoencoder/final}"
MODEL_DIR="${MODEL_DIR:-outputs/models/packed_matryoshka}"
REPORT_DIR="${REPORT_DIR:-outputs/experiments/packing}"
run_stage() { [[ "$STAGE" == all || "$STAGE" == "$1" ]]; }
mkdir -p "$MODEL_DIR" "$REPORT_DIR"
if run_stage train; then
  "$PYTHON" -m training.programs.pretrain_packed --init-from "$BASE_CHECKPOINT" \
    --train-data "$DATA_DIR/train.jsonl" --dev-data "$DATA_DIR/dev.jsonl" \
    --out-dir "$MODEL_DIR" --steps "${STEPS:-6000}" --device "$DEVICE"
fi
if run_stage validate; then
  "$PYTHON" -m experiments.packing.programs.eval_packing \
    --model-path "$MODEL_DIR/final" --dev-data "$DATA_DIR/dev.jsonl" \
    --fusion rotor frame block --device "$DEVICE" --out "$REPORT_DIR/packing.json"
  "$PYTHON" -m experiments.packing.programs.eval_crosstalk_sweep \
    --model-path "$MODEL_DIR/final" --dev-data "$DATA_DIR/dev.jsonl" \
    --device "$DEVICE" --out "$REPORT_DIR/crosstalk.json"
fi
if run_stage multimodal; then
  : "${IMAGE_DIR:?Set IMAGE_DIR to a directory of training images}"
  "$PYTHON" -m training.programs.pretrain_image_packed --image-dir "$IMAGE_DIR" \
    --out "$MODEL_DIR/image_bottleneck.pt" --device "$DEVICE"
  "$PYTHON" -m experiments.packing.programs.eval_multimodal \
    --text-model "$MODEL_DIR/final" --image-bottleneck "$MODEL_DIR/image_bottleneck.pt" \
    --image-dir "$IMAGE_DIR" --dev-data "$DATA_DIR/dev.jsonl" --device "$DEVICE" \
    --save-images "$REPORT_DIR/multimodal_recon.npz" \
    --out "$REPORT_DIR/multimodal.json"
fi
