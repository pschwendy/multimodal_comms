#!/usr/bin/env bash
# Token-filter sweep: REINFORCE-trained token-level filtering policy
# across multiple tau thresholds, using DeepSeek v4-flash as sender/receiver.
#
# The tokenfilter compressor loads a frozen Qwen3-4B proxy + trained policy
# head on GPU 4 (free, avoids conflict with vLLM on 0,1 and repserver on 2).
# Higher tau = fewer tokens survive = more aggressive deletion.
#
# Run from the hiddenbench/ directory.
set -u

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python}"
mkdir -p outputs/hiddenbench/logs/sweep

COMMON="--config ${HIDDENBENCH_CONFIG:-experiments/hiddenbench/configs/config_deepseek.yaml} --num-tasks 16 --seed 7 --rounds 6 --no-full-profile --protocol delta --tokenfilter-device cuda:4"
REPORT_PREFIX="sweep_ds4_tokenfilter"

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

# Identity baseline (no compression)
run_condition identity --compressor identity

# Token filter at various tau thresholds
# tau = 1 - keep_rate. Higher tau = fewer tokens kept.
# tau=0.3 → keep top 70% of tokens
# tau=0.5 → keep top 50%
# tau=0.7 → keep top 30%
run_condition tokenfilter_t030  --compressor tokenfilter --tokenfilter-tau 0.30
run_condition tokenfilter_t050  --compressor tokenfilter --tokenfilter-tau 0.50
run_condition tokenfilter_t070  --compressor tokenfilter --tokenfilter-tau 0.70

echo "SWEEP COMPLETE"
