"""Aggregate Collab-Overcooked per-order metrics by recipe level."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

LEVELS = {
    "level_1": ["baked_bell_pepper", "baked_sweet_potato", "boiled_egg", "boiled_mushroom", "boiled_sweet_potato"],
    "level_2": ["baked_potato_slices", "baked_pumpkin_slices", "boiled_corn_slices", "boiled_green_bean_slices", "boiled_potato_slices"],
    "level_3": ["baked_bell_pepper_soup", "baked_carrot_soup", "baked_mushroom_soup", "baked_potato_soup", "baked_pumpkin_soup"],
    "level_4": ["sliced_bell_pepper_and_corn_stew", "sliced_bell_pepper_and_lentil_stew", "sliced_eggplant_and_chickpea_stew", "sliced_pumpkin_and_chickpea_stew", "sliced_zucchini_and_chickpea_stew"],
    "level_5": ["mashed_broccoli_and_bean_patty", "mashed_carrot_and_chickpea_patty", "mashed_cauliflower_and_lentil_patty", "mashed_potato_and_pea_patty", "mashed_sweet_potato_and_bean_patty"],
    "level_6": ["potato_carrot_and_onion_patty", "romaine_lettuce_pea_and_tomato_patty", "sweet_potato_spinach_and_mushroom_patty", "taro_bean_and_bell_pepper_patty", "zucchini_green_pea_and_onion_patty"],
}
ORDER_TO_LEVEL = {order: level for level, orders in LEVELS.items() for order in orders}


def convert(input_path: str | Path, output_path: str | Path) -> None:
    data = pd.read_csv(input_path)
    data["level"] = data["order"].map(ORDER_TO_LEVEL)
    metric_columns = [column for column in data.columns if column not in {"model", "order", "level"}]
    data[metric_columns] = data[metric_columns].apply(pd.to_numeric, errors="coerce")
    grouped = data.groupby(["model", "level"])[metric_columns].mean()
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    grouped.to_csv(output_path)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="eval_result/statistics_data.csv")
    parser.add_argument("--output", default="eval_result/converted_data.csv")
    args = parser.parse_args()
    convert(args.input, args.output)


if __name__ == "__main__":
    main()
