#!/usr/bin/env bash
# Full Collab-Overcooked channel-compression sweep: all 30 tasks, both models,
# the HiddenBench frontier conditions. Idempotent via .done marker files, so
# it is safe to interrupt and re-run. Conditions run in priority order so the
# highest-value comparisons land first if the sweep doesn't finish.
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python}"
mkdir -p logs/collab_sweep

HORIZON=120
ORDERS=(
  1_baked_bell_pepper 1_baked_sweet_potato 1_boiled_egg 1_boiled_mushroom 1_boiled_sweet_potato
  2_baked_potato_slices 2_baked_pumpkin_slices 2_boiled_corn_slices 2_boiled_green_bean_slices 2_boiled_potato_slices
  3_baked_bell_pepper_soup 3_baked_carrot_soup 3_baked_mushroom_soup 3_baked_potato_soup 3_baked_pumpkin_soup
  4_sliced_bell_pepper_and_corn_stew 4_sliced_bell_pepper_and_lentil_stew 4_sliced_eggplant_and_chickpea_stew
  4_sliced_pumpkin_and_chickpea_stew 4_sliced_zucchini_and_chickpea_stew
  5_mashed_broccoli_and_bean_patty 5_mashed_carrot_and_chickpea_patty 5_mashed_cauliflower_and_lentil_patty
  5_mashed_potato_and_pea_patty 5_mashed_sweet_potato_and_bean_patty
  6_potato_carrot_and_onion_patty 6_romaine_lettuce_pea_and_tomato_patty 6_sweet_potato_spinach_and_mushroom_patty
  6_taro_bean_and_bell_pepper_patty 6_zucchini_green_pea_and_onion_patty
)
# order filenames carry a level prefix; --order matches by substring so strip it
strip_prefix () { echo "$1" | sed -E 's/^[0-9]+_//'; }

# run_task <model: qwen|ds> <run_tag> <compressor> <kwargs_json> <order_raw>
run_task () {
  local model="$1" run_tag="$2" compressor="$3" kwargs="$4" order_raw="$5"
  local order; order="$(strip_prefix "$order_raw")"
  local done_marker="logs/collab_sweep/${run_tag}__${order}.done"
  if [ -f "$done_marker" ]; then
    echo "[skip] ${run_tag} / ${order}"
    return
  fi
  echo "[run ] ${run_tag} / ${order}"

  local model_args=()
  if [ "$model" = "qwen" ]; then
    model_args=(--gpt_model "Qwen/Qwen3-4B" --model_dirname "" --local_server_api "http://localhost:8000/v1")
  else
    model_args=(--gpt_model "deepseek-v4-flash")
  fi

  "$PY" -m multimodal_comms.apps.collab_overcooked.main \
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
  local model="$1" run_tag="$2" compressor="$3" kwargs="$4"
  for o in "${ORDERS[@]}"; do
    run_task "$model" "$run_tag" "$compressor" "$kwargs" "$o"
  done
  echo "[COND DONE] ${run_tag}"
}

# Priority order: identity baselines first, then the frontier candidates
# (both models), then reference/comparison conditions.
run_condition qwen "Qwen3-4B__identity"          identity "{}"
run_condition ds   "deepseek-v4-flash__identity" identity "{}"

run_condition qwen "Qwen3-4B__backref"          backref "{}"
run_condition ds   "deepseek-v4-flash__backref" backref "{}"

run_condition qwen "Qwen3-4B__backref_floor"          backref '{"drop_floor": 0.10}'
run_condition ds   "deepseek-v4-flash__backref_floor" backref '{"drop_floor": 0.10}'

run_condition qwen "Qwen3-4B__backref_codebook"          stack '{"stack": "backref,codebook"}'
run_condition ds   "deepseek-v4-flash__backref_codebook" stack '{"stack": "backref,codebook"}'

run_condition qwen "Qwen3-4B__novelty_state"          novelty '{"stateful": true}'
run_condition ds   "deepseek-v4-flash__novelty_state" novelty '{"stateful": true}'

run_condition qwen "Qwen3-4B__adaptive"          adaptive "{}"
run_condition ds   "deepseek-v4-flash__adaptive" adaptive "{}"

echo "COLLAB SWEEP COMPLETE"
