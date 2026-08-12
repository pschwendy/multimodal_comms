#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
PYTHON="${PYTHON:-python}"; STAGE="${STAGE:-all}"; SEED="${SEED:-42}"
DATA="${DATA:-outputs/data/selectors/train.jsonl}"
MODEL="${MODEL:-outputs/models/selectors/selector.joblib}"
REPORT_DIR="${REPORT_DIR:-outputs/experiments/selectors}"
BENCHMARK_DATA="${BENCHMARK_DATA:-src/multimodal_comms/benchmarks/hiddenbench/data/hiddenbench_official/benchmark.json}"
SPLIT_DIR="${SPLIT_DIR:-outputs/data/selectors/tasks}"
HARVEST_DIR="${HARVEST_DIR:-outputs/data/selectors/discussions}"
EVAL_REPORT="${EVAL_REPORT:-$HARVEST_DIR/eval_identity.json}"
TRAIN_REPORTS="${TRAIN_REPORTS:-$HARVEST_DIR/train_identity.json}"
run_stage() { [[ "$STAGE" == all || "$STAGE" == "$1" ]]; }
mkdir -p "$(dirname "$DATA")" "$(dirname "$MODEL")" "$REPORT_DIR" "$HARVEST_DIR"
if run_stage harvest; then
  : "${BENCH_CONFIG:?Set BENCH_CONFIG to a working HiddenBench YAML configuration}"
  "$PYTHON" -m experiments.selectors.split_tasks --source "$BENCHMARK_DATA" \
    --out-dir "$SPLIT_DIR" --eval-tasks "${EVAL_TASKS:-16}" \
    --train-tasks "${TRAIN_TASKS:-0}" --seed "$SEED"
  common=(--config "$BENCH_CONFIG" --all --rounds "${HARVEST_ROUNDS:-6}" \
    --seed "$SEED" --no-full-profile --official-only --compressor identity \
    --output "$HARVEST_DIR")
  "$PYTHON" -m multimodal_comms.benchmarks.hiddenbench.runtime.cli run \
    "${common[@]}" --data "$SPLIT_DIR/train" --report-name train_identity
  "$PYTHON" -m multimodal_comms.benchmarks.hiddenbench.runtime.cli run \
    "${common[@]}" --data "$SPLIT_DIR/eval" --report-name eval_identity
fi
if run_stage data; then
  # TRAIN_REPORTS is intentionally expanded into multiple --reports arguments.
  # shellcheck disable=SC2086
  "$PYTHON" -m training.data.build_selector_dataset --eval-report "$EVAL_REPORT" \
    --reports $TRAIN_REPORTS --benchmark "$BENCHMARK_DATA" --out "$DATA"
fi
if run_stage train; then
  "$PYTHON" -m training.programs.train_selector --data "$DATA" --out "$MODEL" \
    --validation-tasks "${VALIDATION_TASKS:-4}"
fi
if run_stage validate; then
  "$PYTHON" -m experiments.validation.training_dynamics --base-only
fi
if run_stage benchmark; then
  : "${BENCH_CONFIG:?Set BENCH_CONFIG to a working HiddenBench YAML configuration}"
  common=(--config "$BENCH_CONFIG" --data "$SPLIT_DIR/eval" --official-only --all \
    --rounds "${ROUNDS:-15}" --seed "$SEED" --no-full-profile --output "$REPORT_DIR")
  "$PYTHON" -m multimodal_comms.benchmarks.hiddenbench.runtime.cli run \
    "${common[@]}" --compressor identity --report-name sweep_full_identity
  "$PYTHON" -m multimodal_comms.benchmarks.hiddenbench.runtime.cli run \
    "${common[@]}" --compressor learned --selector-model "$MODEL" \
    --selector-rate "${SELECTOR_RATE:-0.5}" --report-name sweep_full_learned
  "$PYTHON" -m experiments.hiddenbench.programs.aggregate_sweep "$REPORT_DIR"
fi
