#!/usr/bin/env bash
# Single-model stream of the full sweep: run_collab_sweep.sh split by model
# (qwen | ds) so both backends run concurrently — they hit independent
# servers/APIs and touch disjoint run_tag namespaces, so there is no shared
# state between the two streams. Uses the same .done marker convention as
# run_collab_sweep.sh, so work already completed there is skipped here too.
set -u
cd "$(dirname "$0")"
mkdir -p logs/collab_sweep

MODEL="$1"  # qwen | ds
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

if [ "$MODEL" = "qwen" ]; then
  PREFIX="Qwen3-4B"
else
  PREFIX="deepseek-v4-flash"
fi

run_condition "${PREFIX}__identity"          identity "{}"
run_condition "${PREFIX}__backref"          backref "{}"
run_condition "${PREFIX}__backref_floor"          backref '{"drop_floor": 0.10}'
run_condition "${PREFIX}__backref_codebook"          stack '{"stack": "backref,codebook"}'
run_condition "${PREFIX}__novelty_state"          novelty '{"stateful": true}'
run_condition "${PREFIX}__adaptive"          adaptive "{}"

echo "MODEL SWEEP COMPLETE: ${MODEL}"
