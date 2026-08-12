#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"
STAGE="${STAGE:-all}"
DATA_DIR="${DATA_DIR:-outputs/data/fineweb_ae}"
BASE_DIR="${BASE_DIR:-outputs/models/autoencoder}"
MODEL_DIR="${MODEL_DIR:-outputs/models/crypt_ae}"
REPORT_DIR="${REPORT_DIR:-outputs/experiments/crypt_ae}"
DEVICE="${DEVICE:-cuda:0}"
SEED="${SEED:-42}"
BENCH_CONFIG="${BENCH_CONFIG:-}"
run_stage() { [[ "$STAGE" == all || "$STAGE" == "$1" ]]; }
mkdir -p "$DATA_DIR" "$BASE_DIR" "$MODEL_DIR" "$REPORT_DIR"

if run_stage data; then
  "$PYTHON" -m training.data.harvest_fineweb_data \
    --train-examples "${TRAIN_EXAMPLES:-60000}" --dev-examples "${DEV_EXAMPLES:-2000}" \
    --seed "$SEED" --out-dir "$DATA_DIR"
fi
if run_stage base; then
  "$PYTHON" -m training.programs.pretrain_autoencoder \
    --train-data "$DATA_DIR/train.jsonl" --dev-data "$DATA_DIR/dev.jsonl" \
    --out-dir "$BASE_DIR" --steps "${BASE_STEPS:-3000}" --device "$DEVICE" --seed "$SEED"
fi
if run_stage train; then
  "$PYTHON" -m training.programs.pretrain_superpose \
    --init-from "$BASE_DIR/final" --train-data "$DATA_DIR/train.jsonl" \
    --dev-data "$DATA_DIR/dev.jsonl" --out-dir "$MODEL_DIR" \
    --steps "${STEPS:-3000}" --max-slots "${MAX_SLOTS:-8}" \
    --key-mode "${KEY_MODE:-qr}" --device "$DEVICE" --seed "$SEED"
fi
if run_stage validate; then
  "$PYTHON" -m experiments.crypt_ae.programs.eval_cryptae \
    --model-path "$MODEL_DIR/final" --dev-data "$DATA_DIR/dev.jsonl" \
    --loads ${LOADS:-1 2 4 8} --key-mode "${KEY_MODE:-qr}" --device "$DEVICE" \
    --seed "$SEED" --out "$REPORT_DIR/fidelity_security.json"
fi
if run_stage benchmark; then
  : "${BENCH_CONFIG:?Set BENCH_CONFIG to a working HiddenBench YAML configuration}"
  common=(--config "$BENCH_CONFIG" --num-tasks "${TASKS:-16}" --rounds "${ROUNDS:-15}" \
    --seed "$SEED" --no-full-profile --output "$REPORT_DIR")
  "$PYTHON" -m multimodal_comms.benchmarks.hiddenbench.runtime.cli run \
    "${common[@]}" --compressor identity --report-name sweep_full_identity
  "$PYTHON" -m multimodal_comms.benchmarks.hiddenbench.runtime.cli run \
    "${common[@]}" --compressor superpose --superpose-model "$MODEL_DIR/final" \
    --superpose-device "$DEVICE" --report-name sweep_full_superpose
  "$PYTHON" -m experiments.hiddenbench.programs.aggregate_sweep "$REPORT_DIR"
fi
