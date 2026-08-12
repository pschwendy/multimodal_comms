#!/usr/bin/env bash
# Continuation of the sweep for the remaining conditions (backref_codebook,
# novelty_state, adaptive), restricted to each model's demonstrated-solvable
# tier: identity/backref/backref_floor already completed full 30-task scope
# before this restriction was decided, so those are left untouched. Uses the
# same .done marker convention/directory as run_collab_sweep_model.sh, so any
# already-completed backref_codebook tasks (e.g. DeepSeek's 8/10 so far) are
# correctly skipped, not redone.
set -u
cd "$(dirname "$0")"
mkdir -p logs/collab_sweep

MODEL="$1"  # qwen | ds
HORIZON=120
if [ "$MODEL" = "qwen" ]; then
  ORDERS=(1_baked_bell_pepper 1_baked_sweet_potato 1_boiled_egg 1_boiled_mushroom 1_boiled_sweet_potato)
  PREFIX="Qwen3-4B"
else
  ORDERS=(
    1_baked_bell_pepper 1_baked_sweet_potato 1_boiled_egg 1_boiled_mushroom 1_boiled_sweet_potato
    2_baked_potato_slices 2_baked_pumpkin_slices 2_boiled_corn_slices 2_boiled_green_bean_slices 2_boiled_potato_slices
  )
  PREFIX="deepseek-v4-flash"
fi
strip_prefix () { echo "$1" | sed -E 's/^[0-9]+_//'; }

run_task () {
  local run_tag="$1" compressor="$2" kwargs="$3" order_raw="$4"
  local order; order="$(strip_prefix "$order_raw")"
  local done_marker="logs/collab_sweep/${run_tag}__${order}.done"
  if [ -f "$done_marker" ]; then
    echo "[skip] ${run_tag} / ${order}"
    return
  fi
  echo "[run ] ${run_tag} / ${order}"
  local model_args=()
  if [ "$MODEL" = "qwen" ]; then
    model_args=(--gpt_model "Qwen/Qwen3-4B" --model_dirname "" --local_server_api "http://localhost:8000/v1")
  else
    model_args=(--gpt_model "deepseek-v4-flash")
  fi
  timeout 1200 conda run -n collab_oc --no-capture-output python main.py \
    --horizon "$HORIZON" --order "$order" "${model_args[@]}" \
    --compressor "$compressor" --compressor-kwargs "$kwargs" --run-tag "$run_tag" \
    > "logs/collab_sweep/${run_tag}__${order}.log" 2>&1
  if [ $? -eq 0 ]; then
    touch "$done_marker"
    echo "[done] ${run_tag} / ${order}"
  else
    echo "[FAIL] ${run_tag} / ${order} (see log)"
  fi
}

run_condition () {
  local run_tag="$1" compressor="$2" kwargs="$3"
  for o in "${ORDERS[@]}"; do
    run_task "$run_tag" "$compressor" "$kwargs" "$o"
  done
  echo "[COND DONE] ${run_tag}"
}

run_condition "${PREFIX}__backref_codebook" stack '{"stack": "backref,codebook"}'
run_condition "${PREFIX}__novelty_state"    novelty '{"stateful": true}'
run_condition "${PREFIX}__adaptive"         adaptive "{}"

echo "RESTRICTED MODEL SWEEP COMPLETE: ${MODEL}"
