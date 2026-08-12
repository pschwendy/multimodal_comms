#!/bin/bash
# Runs the full A2 + A1 training chain once the identity harvest completes.
# Logs to scratchpad/chain.log; writes CHAIN_DONE / CHAIN_FAIL markers.
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python}"
TRAIN_PYTHON="${TRAIN_PYTHON:-$PY}"
SP="${LOG_DIR:-$REPO_ROOT/outputs/iagents/logs}"
HARVEST="${1:-outputs/iagents/reports/harvest_identity.json}"
mkdir -p "$SP" outputs/iagents/training

step() { echo "=== [$(date +%H:%M:%S)] $* ==="; }

step "waiting for harvest report"
until grep -q -- "-> $HARVEST" "$SP/harvest.log" 2>/dev/null; do
  if ! pgrep -f "run_offline_eval.py --dataset .*harvest_subset" >/dev/null 2>&1; then
     if [ ! -f "$HARVEST" ]; then echo "CHAIN_FAIL: harvest proc gone, no report"; exit 1; fi
     break
  fi
  sleep 20
done
step "harvest done: $(grep -c '\"correct\"' $HARVEST 2>/dev/null) results"

step "A2: build repmatch selector dataset"
"$PY" -m training.benchmark_adaptation.iagents.build_repmatch_dataset "$HARVEST" --out outputs/iagents/training/repmatch_train.jsonl || { echo "CHAIN_FAIL: A2 build"; exit 1; }

step "A2: train selector -> data/repmatch_selector.joblib"
"$PY" -m training.benchmark_adaptation.iagents.train_repmatch_selector --data outputs/iagents/training/repmatch_train.jsonl --out outputs/iagents/training/repmatch_selector.joblib || { echo "CHAIN_FAIL: A2 train"; exit 1; }

step "A1: harvest GRPO rewriter views"
"$PY" -m training.benchmark_adaptation.iagents.harvest_repmatch_data "$HARVEST" --out outputs/iagents/training/repmatch_rewriter_train.jsonl || { echo "CHAIN_FAIL: A1 harvest"; exit 1; }

step "A1: GRPO train on GPU5 (agent env)"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}" "$TRAIN_PYTHON" -m training.benchmark_adaptation.iagents.train_repmatch_grpo --steps 120 --max-examples 300 || { echo "CHAIN_FAIL: A1 GRPO"; exit 1; }

step "CHAIN_DONE"
