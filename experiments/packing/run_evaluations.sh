#!/usr/bin/env bash
# Full evaluation sweep for a trained packed-matryoshka checkpoint.
#   experiments/packing/run_evaluations.sh <checkpoint-dir> [device]
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
CKPT="${1:-data/packed_matryoshka/final}"
DEV="${2:-cuda:0}"
PY="${PYTHON:-python}"

echo "=== packing sweep: fp32 wire, all fusions, full ladder + 2x overload ==="
"$PY" -m experiments.packing.programs.eval_packing --model-path "$CKPT" --device "$DEV" \
    --fusion rotor frame block --n-packets 4 --slots-scored 3 \
    --short-chars 200 --overload 2.0 \
    --out outputs/experiments/packing/packing_sweep.json

echo "=== packing sweep: 8-bit wire format ==="
"$PY" -m experiments.packing.programs.eval_packing --model-path "$CKPT" --device "$DEV" \
    --fusion rotor frame --n-packets 3 --slots-scored 3 \
    --short-chars 200 --bits 8 --overload \
    --out outputs/experiments/packing/packing_sweep_int8.json

echo "=== mixed text+image packets ==="
"$PY" -m experiments.packing.programs.eval_multimodal --text-model "$CKPT" --device "$DEV" \
    --widths 640 160 80 40 --n-packets 2 --n-images 4 --n-texts 3 --bits 8 \
    --out outputs/experiments/packing/multimodal_sweep.json --save-images outputs/experiments/packing/multimodal_mixed.npz

echo "=== artifact ==="
"$PY" experiments/packing/programs/build_packing_artifact.py
echo "done"
