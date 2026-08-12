#!/usr/bin/env bash
# Phase 2: middleware-only methods, pushed harder.
# - stateful novelty: per-receiver memory drops repeats AND paraphrases of
#   already-shown content (middleware analogue of delta at sentence level,
#   works without touching the protocol)
# - llmlingua2 at a more aggressive rate
# - combinations on the delta protocol
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

# Stateful novelty: the middleware move that subsumes delta's savings
run_condition full_novelty_state   --compressor novelty --novelty-stateful
run_condition delta_novelty_state  --protocol delta --compressor novelty --novelty-stateful

# Aggressive token pruning
run_condition full_lingua25        --compressor llmlingua2 --lingua-rate 0.25
run_condition delta_lingua25       --protocol delta --compressor llmlingua2 --lingua-rate 0.25

# Cheap extras
run_condition delta_window2        --protocol delta --compressor window --window-rounds 2

echo "SWEEP PHASE2 COMPLETE"
