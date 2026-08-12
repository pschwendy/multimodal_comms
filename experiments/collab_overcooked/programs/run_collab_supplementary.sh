#!/usr/bin/env bash
# Supplementary add-ons (after the primary matrix): gzip64 strawman vs
# identity on a handful of tasks/one model (sanity-reconfirm the "classical
# compression is anti-compression in a token channel" finding generalizes),
# and a timestep-vs-episode channel-scope ablation for backref, Qwen, full 30.
set -u
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_ROOT"
export PYTHONPATH="$REPO_ROOT/src:$REPO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
PY="${PYTHON:-python}"
mkdir -p logs/collab_sweep

HORIZON=120
ORDERS_FULL=(
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
ORDERS_SMALL=(1_boiled_egg 2_boiled_potato_slices 3_baked_potato_soup 4_sliced_eggplant_and_chickpea_stew)
ORDERS_QWEN_TIER=(1_baked_bell_pepper 1_baked_sweet_potato 1_boiled_egg 1_boiled_mushroom 1_boiled_sweet_potato)

strip_prefix () { echo "$1" | sed -E 's/^[0-9]+_//'; }

run_task () {
  local model="$1" run_tag="$2" compressor="$3" kwargs="$4" scope="$5" order_raw="$6"
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
  timeout 1200 "$PY" -m multimodal_comms.apps.collab_overcooked.main \
    --horizon "$HORIZON" --order "$order" "${model_args[@]}" \
    --compressor "$compressor" --compressor-kwargs "$kwargs" --channel-scope "$scope" \
    --run-tag "$run_tag" \
    > "logs/collab_sweep/${run_tag}__${order}.log" 2>&1
  if [ $? -eq 0 ]; then
    touch "$done_marker"
    echo "[done] ${run_tag} / ${order}"
  else
    echo "[FAIL] ${run_tag} / ${order} (see log)"
  fi
}

echo "--- gzip64 strawman (Qwen, 4 tasks) ---"
for o in "${ORDERS_SMALL[@]}"; do
  run_task qwen "Qwen3-4B__gzip64_strawman" gzip64 "{}" episode "$o"
done

echo "--- channel-scope ablation: backref, timestep vs episode (Qwen, level-1 tier) ---"
for o in "${ORDERS_QWEN_TIER[@]}"; do
  run_task qwen "Qwen3-4B__backref_timestepscope" backref "{}" timestep "$o"
done

echo "SUPPLEMENTARY SWEEP COMPLETE"
