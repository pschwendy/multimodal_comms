#!/usr/bin/env bash
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"; cd "$ROOT"
export PYTHONPATH="$ROOT/src:$ROOT${PYTHONPATH:+:$PYTHONPATH}"
if [[ "${QUICK:-0}" == 1 ]]; then
  "${PYTHON:-python}" -m multimodal_comms.apps.collab_overcooked.main \
    --order "${ORDER:-baked_bell_pepper}" --horizon "${HORIZON:-120}" --episode 1 \
    --gpt_model "${MODEL:-deepseek-v4-flash}" --compressor "${METHOD:-identity}" \
    --channel-scope "${CHANNEL_SCOPE:-episode}" \
    --log_dir "${REPORT_DIR:-outputs/experiments/collab_overcooked}"
else
  bash experiments/collab_overcooked/programs/run_collab_sweep.sh
  bash experiments/collab_overcooked/programs/run_eval_pipeline.sh
fi
