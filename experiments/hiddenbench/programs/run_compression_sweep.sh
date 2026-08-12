#!/usr/bin/env bash
# Compression sweep: fixed (non-learnable) channel-compression methods
# across both protocols, on a fixed task subset with the local Qwen3-4B server.
#
# Rounds are capped at 6 for ALL conditions: at 15 rounds the uncompressed
# full-history protocol overflows Qwen3-4B's 40960-token context, which would
# drop tasks selectively from the baseline and bias the comparison.
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

# --- full_history protocol (original) ---
run_condition full_identity
run_condition full_window2   --compressor window --window-rounds 2
run_condition full_novelty   --compressor novelty
run_condition full_lingua    --compressor llmlingua2
run_condition full_concise   --message-style concise
run_condition full_schema    --message-style schema

# --- delta protocol (no retransmission) ---
run_condition delta_identity --protocol delta
run_condition delta_novelty  --protocol delta --compressor novelty
run_condition delta_lingua   --protocol delta --compressor llmlingua2
run_condition delta_concise  --protocol delta --message-style concise
run_condition delta_schema   --protocol delta --message-style schema

echo "SWEEP COMPLETE"
