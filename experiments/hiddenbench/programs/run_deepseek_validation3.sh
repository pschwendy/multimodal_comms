#!/usr/bin/env bash
# Validate the 2026-07-13 frontier winners on deepseek-v4-flash
# (deepseek-chat was retired from the API; ds_ baselines are not comparable).
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python}"
mkdir -p outputs/hiddenbench/logs/sweep

COMMON="--config ${HIDDENBENCH_CONFIG:-experiments/hiddenbench/configs/config_deepseek.yaml} --model deepseek-v4-flash --num-tasks 16 --seed 7 --rounds 6 --no-full-profile"

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

run_condition ds4_full_identity
run_condition ds4_full_novelty_state     --compressor novelty --novelty-stateful
run_condition ds4_full_backref           --compressor backref
run_condition ds4_full_backref_codebook2 --compressor stack --stack backref,codebook
run_condition ds4_full_backref_floor     --compressor backref --backref-drop-floor 0.10

echo "DS4 VALIDATION COMPLETE"
