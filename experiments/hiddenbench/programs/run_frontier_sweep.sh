#!/usr/bin/env bash
# Frontier sweep: semantic back-references (LZ77-style), shared-phrase
# codebook (LZ78-style, lossless), adaptive learned dedup (no budget),
# their composition, and the classical-compression strawman (zlib+base64).
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

run_condition full_backref          --compressor backref
run_condition full_codebook         --compressor codebook
run_condition full_backref_codebook --compressor stack --stack backref,codebook
run_condition full_adaptive         --compressor adaptive
run_condition delta_adaptive        --protocol delta --compressor adaptive
run_condition delta_backref         --protocol delta --compressor backref
run_condition full_gzip64           --compressor gzip64

echo "FRONTIER SWEEP COMPLETE"
