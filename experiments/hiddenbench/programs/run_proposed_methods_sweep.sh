#!/usr/bin/env bash
# Sweep for the 5 "external semantic compression" proposed methods:
#   certspan     - certified span deletion (compress-only, lossy, certified)
#   semfallback  - semantically verified fallback ladder (compress-only, meta)
#   pdiff        - predictive-diff channel coding (compress+decompress, lossless)
#   telegraphic  - telegraphic encoding + generative reinflation (compress+decompress, lossy)
#   ratediff     - rate-controlled predictive-diff (compress+decompress, unified lossy<->lossless)
#
# HiddenBench + deepseek-v4-flash only, per current experiment scope. Reuses
# the existing sweep_ds4_full_identity / sweep_ds4_delta_identity reports as
# the baseline (already complete, same 16-task/seed7/6-round config) instead
# of re-spending API budget on identity.
#
# Prereqs:
#   - training.services.repserver running on :8100 (certspan, semfallback, ratediff,
#     telegraphic all call its /rep_batch endpoint for the certificate).
#   - GPU headroom for the frozen Qwen2.5-0.5B-Instruct model used by pdiff/
#     telegraphic/ratediff (default cuda:3; cheap, ~1GB).
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

echo "[info] baselines reused from sweep_ds4_full_identity / sweep_ds4_delta_identity"

# --- full_history protocol ---
run_condition ds4_full_certspan \
  --compressor certspan --certspan-eps 0.08

run_condition ds4_full_semfallback \
  --compressor semfallback --semfallback-eps 0.10

run_condition ds4_full_pdiff \
  --compressor pdiff --pdiff-device cuda:3

run_condition ds4_full_telegraphic \
  --compressor telegraphic --telegraphic-eps 0.20 --telegraphic-device cuda:3

run_condition ds4_full_ratediff_eps02 \
  --compressor ratediff --ratediff-eps 0.02 --pdiff-device cuda:3

run_condition ds4_full_ratediff_eps06 \
  --compressor ratediff --ratediff-eps 0.06 --pdiff-device cuda:3

# --- delta protocol (redundancy is already removed by the protocol itself;
#     tests whether pdiff/ratediff still find exploitable predictability in
#     genuinely-new content) ---
run_condition ds4_delta_pdiff \
  --protocol delta --compressor pdiff --pdiff-device cuda:3

run_condition ds4_delta_ratediff_eps02 \
  --protocol delta --compressor ratediff --ratediff-eps 0.02 --pdiff-device cuda:3

echo "PROPOSED METHODS SWEEP COMPLETE"
