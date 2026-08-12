#!/usr/bin/env bash
# Learnable-compressor sweep: learned extractive selector at two budgets,
# with and without stateful dedup composition, on both protocols.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python}"
mkdir -p outputs/hiddenbench/logs/sweep

COMMON="--config ${HIDDENBENCH_CONFIG:-experiments/hiddenbench/configs/config.example.yaml} --num-tasks 16 --seed 7 --rounds 6 --no-full-profile"

run_condition () {
  local name="$1"; shift
  if [ -f "outputs/hiddenbench/reports/sweep_${name}.json" ] && python3 -c "
import json,sys
d=json.load(open('outputs/hiddenbench/reports/sweep_${name}.json'))
sys.exit(0 if d['metadata'].get('status')=='complete' else 1)
" 2>/dev/null; then
    echo "[skip] ${name} already complete"
    return
  fi
  echo "[run ] ${name}: $*"
  "$PY" -m multimodal_comms.benchmarks.hiddenbench.runtime.cli run $COMMON \
    --report-name "sweep_${name}" "$@" \
    > "outputs/hiddenbench/logs/sweep/${name}.log" 2>&1
  echo "[done] ${name} (exit $?)"
}

run_condition full_learned50        --compressor learned --selector-rate 0.5
run_condition full_learned50_dedup  --compressor learned --selector-rate 0.5 --selector-dedup
run_condition full_learned35_dedup  --compressor learned --selector-rate 0.35 --selector-dedup
run_condition delta_learned50       --protocol delta --compressor learned --selector-rate 0.5
run_condition delta_learned50_dedup --protocol delta --compressor learned --selector-rate 0.5 --selector-dedup
run_condition delta_learned35_dedup --protocol delta --compressor learned --selector-rate 0.35 --selector-dedup

echo "LEARNED SWEEP COMPLETE"
