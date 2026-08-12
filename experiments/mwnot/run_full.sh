#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"; STAGE="${STAGE:-all}"; SEED="${SEED:-42}"
DATA_DIR="${DATA_DIR:-outputs/data/fineweb_ae}"
MODEL_DIR="${MODEL_DIR:-outputs/models/mwnot_autoencoder}"
REPORT_DIR="${REPORT_DIR:-outputs/experiments/mwnot}"
DEVICE="${DEVICE:-cuda:0}"; NUM_LATENTS="${NUM_LATENTS:-4}"
run_stage() { [[ "$STAGE" == all || "$STAGE" == "$1" ]]; }
mkdir -p "$DATA_DIR" "$MODEL_DIR" "$REPORT_DIR"
if run_stage data; then
  "$PYTHON" -m training.data.harvest_fineweb_data \
    --train-examples "${TRAIN_EXAMPLES:-60000}" --dev-examples "${DEV_EXAMPLES:-2000}" \
    --seed "$SEED" --out-dir "$DATA_DIR"
fi
if run_stage train; then
  "$PYTHON" -m training.programs.pretrain_mwnot_autoencoder \
    --train-data "$DATA_DIR/train.jsonl" --dev-data "$DATA_DIR/dev.jsonl" \
    --out-dir "$MODEL_DIR" --steps "${STEPS:-3000}" --num-latents "$NUM_LATENTS" \
    --device "$DEVICE" --seed "$SEED"
fi
if run_stage validate; then
  "$PYTHON" -m experiments.mwnot.smoke_test
  "$PYTHON" -m training.validation.eval_autoencoder_qualitative \
    --checkpoint "$MODEL_DIR/final" --dev-data "$DATA_DIR/dev.jsonl" \
    --device "$DEVICE" --seed "$SEED"
fi
if run_stage benchmark; then
  : "${BENCH_CONFIG:?Set BENCH_CONFIG to a working HiddenBench YAML configuration}"
  common=(--config "$BENCH_CONFIG" --num-tasks "${TASKS:-16}" --rounds "${ROUNDS:-15}" \
    --seed "$SEED" --no-full-profile --output "$REPORT_DIR")
  "$PYTHON" -m multimodal_comms.benchmarks.hiddenbench.runtime.cli run \
    "${common[@]}" --compressor identity --report-name sweep_full_identity
  "$PYTHON" -m multimodal_comms.benchmarks.hiddenbench.runtime.cli run \
    "${common[@]}" --compressor mwnot_autoencoder \
    --mwnot-autoencoder-model "$MODEL_DIR/final" \
    --mwnot-autoencoder-num-latents "$NUM_LATENTS" \
    --mwnot-autoencoder-device "$DEVICE" --report-name sweep_full_mwnot_autoencoder
  "$PYTHON" -m experiments.hiddenbench.programs.aggregate_sweep "$REPORT_DIR"
fi
