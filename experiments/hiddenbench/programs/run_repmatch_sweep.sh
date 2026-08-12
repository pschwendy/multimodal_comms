#!/usr/bin/env bash
# Representational-match compression sweep on deepseek-v4-flash.
# Five conditions (identity baseline + the 4 rep-match compressors) under both
# the full-history and delta protocols. Condition names are prefixed ds4_ so
# The benchmark aggregate program combines them with the other condition reports.
#
# The two repmatch_rewriter conditions need data/repmatch_grpo/final; they run
# LAST and block until that checkpoint exists, so this sweep can be launched in
# parallel with train_repmatch_grpo.py. Everything else only needs
# data/repmatch_selector.joblib (train it before launching this).
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python}"
mkdir -p outputs/hiddenbench/logs/sweep

COMMON="--config ${HIDDENBENCH_CONFIG:-experiments/hiddenbench/configs/config_deepseek.yaml} --num-tasks 16 --seed 7 --rounds 6 --no-full-profile"

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

# --- Conditions needing only the selector artifact (or nothing) ---
run_condition ds4_full_identity          --compressor identity
run_condition ds4_full_repmatch_selector --compressor repmatch_selector
run_condition ds4_full_saliency          --compressor saliency
run_condition ds4_full_repmatch_bestofk  --compressor repmatch_bestofk

run_condition ds4_delta_identity          --protocol delta --compressor identity
run_condition ds4_delta_repmatch_selector --protocol delta --compressor repmatch_selector
run_condition ds4_delta_saliency          --protocol delta --compressor saliency
run_condition ds4_delta_repmatch_bestofk  --protocol delta --compressor repmatch_bestofk

# --- Rewriter conditions: only if a trained checkpoint exists ---
# The first GRPO run collapsed into degenerate repetition; a valid checkpoint
# is promoted to data/repmatch_grpo/final only after passing a hand-inspection
# gate. If it never does, these conditions are skipped rather than transmitting
# garbage (or hanging the sweep forever).
if [ -d data/repmatch_grpo/final ]; then
  echo "[info] GRPO checkpoint present; running rewriter conditions."
  run_condition ds4_full_repmatch_rewriter  --compressor repmatch_rewriter
  run_condition ds4_delta_repmatch_rewriter --protocol delta --compressor repmatch_rewriter
else
  echo "[skip] no data/repmatch_grpo/final (rewriter training did not pass inspection); skipping rewriter conditions"
fi

echo "REPMATCH SWEEP COMPLETE"
