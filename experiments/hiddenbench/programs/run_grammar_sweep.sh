#!/usr/bin/env bash
# Grammar compressor sweep: BPE grammar-based compression
# across DeepSeek v4-flash sender/receiver with protocol delta.
#
# The grammar compressor uses a precomputed global codebook of common
# multi-word phrases. No GPU needed — pure string substitution.
#
# Run from the hiddenbench/ directory.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python}"
mkdir -p outputs/hiddenbench/logs/sweep

COMMON="--config ${HIDDENBENCH_CONFIG:-experiments/hiddenbench/configs/config_deepseek.yaml} --num-tasks 16 --seed 7 --rounds 6 --no-full-profile --protocol delta"
REPORT_PREFIX="sweep_ds4_grammar"

run_condition () {
  local name="$1"; shift
  if [ -f "outputs/hiddenbench/reports/${REPORT_PREFIX}_${name}.json" ] && python3 -c "
import json,sys
d=json.load(open('outputs/hiddenbench/reports/${REPORT_PREFIX}_${name}.json'))
sys.exit(0 if d['metadata'].get('status')=='complete' else 1)
" 2>/dev/null; then
    echo "[skip] ${name} already complete"
    return
  fi
  echo "[run ] ${name}: $*"
  "$PY" -m multimodal_comms.benchmarks.hiddenbench.runtime.cli run $COMMON \
    --report-name "${REPORT_PREFIX}_${name}" "$@" \
    > "outputs/hiddenbench/logs/sweep/${REPORT_PREFIX}_${name}.log" 2>&1
  echo "[done] ${name} (exit $?)"
}

# Identity baseline
run_condition identity --compressor identity

# Grammar compressor (lossless, deterministic, no GPU)
run_condition grammar --compressor grammar

echo "SWEEP COMPLETE"
