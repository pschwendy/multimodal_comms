import time
import datetime
import os,re
import json
import datetime
from argparse import ArgumentParser
import pandas as pd
from rich import print as rprint

from multimodal_comms.apps.collab_overcooked.eval_utils import Evaluation, ExpLog
from distutils.util import strtobool

models = ["gpt-4o"]

orders = ["apple","carrot", "onion", "potato", "tofu", "baked_bell_pepper", "baked_sweet_potato", "boiled_egg", "boiled_mushroom", "boiled_sweet_potato", "baked_potato_slices", "baked_pumpkin_slices", "boiled_corn_slices", "boiled_green_bean_slices", "boiled_potato_slices", "baked_bell_pepper_soup", "baked_carrot_soup", "baked_mushroom_soup", "baked_potato_soup", "baked_pumpkin_soup", "sliced_bell_pepper_and_corn_stew", "sliced_bell_pepper_and_lentil_stew", "sliced_eggplant_and_chickpea_stew", "sliced_pumpkin_and_chickpea_stew", "sliced_zucchini_and_chickpea_stew", "mashed_broccoli_and_bean_patty", "mashed_carrot_and_chickpea_patty", "mashed_cauliflower_and_lentil_patty", "mashed_potato_and_pea_patty", "mashed_sweet_potato_and_bean_patty", "potato_carrot_and_onion_patty", "romaine_lettuce_pea_and_tomato_patty", "sweet_potato_spinach_and_mushroom_patty", "taro_bean_and_bell_pepper_patty", "zucchini_green_pea_and_onion_patty"]


def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))

def main(variant):

    if variant['mode'] == 'exp':

        if variant['test_mode'] == 'fix_task':
            auto_order_list = []
            if variant['order'] == 'AUTO':
                exp_log = ExpLog(variant['log_dir'] + '/' + variant['model'])
                if exp_log.__len__() == 0:
                    print(
                        f"[evaluation] No evaluable logs found: missing experiment_*.json under {variant['log_dir']}/{variant['model']}"
                    )
                    return
                for idx in range(exp_log.__len__()):
                    auto_order_list.append(exp_log.get_secondary_order_list(idx, 0)[0])
                eval = Evaluation(order_name_list=auto_order_list,exp_log=exp_log)
            else:
                exp_log = ExpLog(variant['log_dir']+ '/' + variant['model'] + '/' + variant['order'])
                if exp_log.__len__() == 0:
                    print(
                        f"[evaluation] No evaluable logs found: missing experiment_*.json under {variant['log_dir']}/{variant['model']}/{variant['order']}"
                    )
                    return
                for idx in range(exp_log.__len__()):
                    auto_order_list.append(variant['order'])
                eval = Evaluation(order_name_list=auto_order_list,exp_log=exp_log)

            # NOTE: In fix_task mode we used to only construct Evaluation, but never call evaluate().
            # Keep behavior consistent with build_in mode: write results into the corresponding log directory.
            if variant.get("save", True):
                if variant["order"] == "AUTO":
                    save_dir = os.path.join(variant["save_dir"], variant["model"])
                else:
                    save_dir = os.path.join(
                        variant["save_dir"], variant["model"], variant["order"]
                    )
                os.makedirs(save_dir, exist_ok=True)
                try:
                    eval.evaluate(save_dir)
                except Exception as e:
                    print(f"[evaluation] Evaluation failed: {e}")
                    raise

        if variant['test_mode'] == 'build_in':
            for model in models:
                for order in orders:
                    auto_order_list = []
                    variant['model'] = model
                    variant['order'] = order
                    exp_log = ExpLog(variant['log_dir']+ '/' + variant['model'] + '/' + variant['order'])
                    if exp_log.__len__() == 0:
                        print(
                            f"[evaluation] Skip: no experiment_*.json under {variant['log_dir']}/{variant['model']}/{variant['order']}"
                        )
                        continue
                    for idx in range(exp_log.__len__()):
                        auto_order_list.append(variant['order'])
                    eval = Evaluation(order_name_list=auto_order_list,exp_log=exp_log)

                    #print(eval.evaluate(variant['save_dir']))
                    try:
                        save_dir = os.path.join(
                            variant["save_dir"], variant["model"], variant["order"]
                        )
                        os.makedirs(save_dir, exist_ok=True)
                        eval.evaluate(save_dir)
                    except Exception as e:
                        print(f"[evaluation] Evaluation failed ({variant['model']}/{variant['order']}): {e}")
                        raise




if __name__ == '__main__':
    
    parser = ArgumentParser(description='OvercookedAI Experiment')

    parser.add_argument('--model', type=str, default='gpt-3.5-turbo-0125', help='Number of episodes')

    parser.add_argument('--K', type=int, default=0, help="The number of dialogues you want to retrieve.")
    
    # 
    parser.add_argument('--mode', type=str, default='exp', choices=['exp', 'develop'], help='exp mode run step-by-step, demo mode run via traj')                                
    parser.add_argument('--test_mode', type=str, default='fix_task', choices=['fix_task', 'build_in'])
    parser.add_argument('--save', type=boolean_argument, default=True, help='Whether save the result')
    parser.add_argument('--log_dir', type=str, default='data', help='dir to read experiment logs from')
    parser.add_argument('--debug', type=boolean_argument, default=True, help='debug mode')
    parser.add_argument('--order', type=str,default='AUTO', help='1 task order name, "AUTO" represents automatic recognition')
    parser.add_argument('--recipe_dir', type=str, default='prompts/recipe', help='The dir of the recipe and reference')
    parser.add_argument('--save_dir', type=str, default='eval_result', help='dir to write evaluation results to')


    args = parser.parse_args()
    variant = vars(args)


    start_time = time.time()
    main(variant)
    end_time = time.time()
    print(f"\n=======Finshed all=========\n")
    print(f"Cost time : {end_time - start_time:.3f}s-----\n\n")
