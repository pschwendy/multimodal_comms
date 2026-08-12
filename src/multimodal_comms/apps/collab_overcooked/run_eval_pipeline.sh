#!/usr/bin/env bash
# Runs evaluation.py for every completed sweep entry (per run_collab_sweep.sh's
# .done markers). Standard eval_utils.py metrics need no API calls.
set -u
cd "$(dirname "$0")"
mkdir -p logs/collab_eval

for marker in logs/collab_sweep/*.done; do
  [ -e "$marker" ] || continue
  base="$(basename "$marker" .done)"
  run_tag="${base%__*}"
  order="${base##*__}"
  out="eval_result/${run_tag}/${order}/evaluation_result.json"
  if [ -f "$out" ]; then
    echo "[skip] ${run_tag} / ${order}"
    continue
  fi
  echo "[eval] ${run_tag} / ${order}"
  conda run -n collab_oc --no-capture-output python evaluation.py \
    --test_mode fix_task --model "$run_tag" --order "$order" \
    > "logs/collab_eval/${run_tag}__${order}.log" 2>&1
done
echo "EVAL PIPELINE COMPLETE"
