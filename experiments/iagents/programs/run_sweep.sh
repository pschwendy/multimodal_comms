#!/bin/bash
# Run the 5-condition compression sweep in parallel on the eval subset, then
# aggregate. Usage: run_sweep.sh <eval.jsonl> <selector_tau>
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python}"
SP="${LOG_DIR:-$REPO_ROOT/outputs/iagents/logs}"
mkdir -p "$SP" "$REPO_ROOT/outputs/iagents/reports"
EVAL=${1:?eval jsonl}
TAU=${2:-0.05}
MR=2

run() { # name  extra-args...
  local name=$1; shift
  "$PY" -m experiments.iagents.programs.run_offline_eval --dataset "$EVAL" --condition "$name" \
      --max-round $MR --out "outputs/iagents/reports/eval_${name}.json" "$@" \
      > "$SP/sweep_${name}.log" 2>&1
  echo "done $name"
}

echo "=== sweep start $(date +%H:%M:%S) eval=$EVAL tau=$TAU ==="
run identity &
run saliency --param rate=0.4 &
run repmatch_bestofk &
run repmatch_selector --param tau=$TAU --param model_path=data/repmatch_selector.joblib &
run repmatch_rewriter --param model_path=data/repmatch_grpo/final --param rate=0.4 --param device=cuda:3 &
wait
echo "=== all conditions done $(date +%H:%M:%S) ==="

"$PY" -m experiments.iagents.programs.aggregate_sweep \
   outputs/iagents/reports/eval_identity.json outputs/iagents/reports/eval_saliency.json \
   outputs/iagents/reports/eval_repmatch_bestofk.json outputs/iagents/reports/eval_repmatch_selector.json \
   outputs/iagents/reports/eval_repmatch_rewriter.json --out outputs/iagents/reports/sweep_needle.md
echo "SWEEP_DONE"
