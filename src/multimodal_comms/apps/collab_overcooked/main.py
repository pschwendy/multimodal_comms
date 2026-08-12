import time
import datetime
import os
import json
import datetime
from argparse import ArgumentParser
import numpy as np
from rich import print as rprint
import copy
from collections import deque

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  
os.environ["CUDA_VISIBLE_DEVICES"] = "-1" 
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*cuBLAS factory.*")  # ignore "Unable to register cuBLAS factory" due to use tf-CPU

from distutils.util import strtobool
from multimodal_comms.apps.collab_overcooked.settings import PROMPT_DIR

def boolean_argument(value):
    """Convert a string value to boolean."""
    return bool(strtobool(value))

def check_recipe_parse(variant):
    recipe_name_list = os.listdir(PROMPT_DIR / "recipe")
    recipe_filename = ""
    for r in recipe_name_list:
        if variant['order'] in r.lower():
            recipe_filename = r
            break
    if recipe_filename == "":
        raise ValueError("Not valid order name!")
    else:
        return True


VERSION = "bundled"

from multimodal_comms.apps.collab_overcooked.overcooked_ai_py.mdp.overcooked_mdp import OvercookedGridworld, OvercookedState
from multimodal_comms.apps.collab_overcooked.overcooked_ai_py.mdp.overcooked_env import OvercookedEnv
from multimodal_comms.apps.collab_overcooked.overcooked_ai_py.agents.agent import AgentGroup
from multimodal_comms.apps.collab_overcooked.overcooked_ai_py.mdp.actions import Action
from multimodal_comms.apps.collab_overcooked.collab.modules import statistics_dict, tokenizer,model, turn_statistics_dict
from multimodal_comms.apps.collab_overcooked.collab.web_util import output_to_port, check_port_in_use, change_port
import socket
from multimodal_comms.apps.collab_overcooked.utils import (
    combine_statistic_dict,
    get_example_embedding,
    make_agent,
)
from multimodal_comms.apps.collab_overcooked.channel_adapter import ChannelAdapter


def main(variant):

    layout = variant['layout']
    horizon = variant['horizon']
    episode = variant['episode']

    mode = variant['mode']
    
    mdp = OvercookedGridworld.from_layout_name(layout)

    #set order according to parser
    if variant['order'] !="" and check_recipe_parse(variant):
        mdp.start_order_list = [variant['order']]
        # 1 task mode
        mdp.one_task_mode = True

    env = OvercookedEnv(mdp, horizon=horizon)
    env.reset()

    
    p0_algo = variant['p0']
    p1_algo = variant['p1']
    print(f"\n===P0 agent: {p0_algo} | P1 agent: {p1_algo}===\n")


    start_time = time.time()
    results = []

    actor_num = 0
    actor_list = ['chef','assistant']
    for i in range(episode):  
        
        agents_list = []

        current_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        save_dir = f"{variant['statistics_save_dir']}/{variant['model_tag']}/{args.order}"
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        filename = f"{save_dir}/experiment_{current_time}_{args.order}.json"
        channel_stats_filename = f"{save_dir}/channel_stats_{current_time}_{args.order}.json"

        if mode == 'develop':
            """
            You can customize the 'action_list' and 'parm' to test the environment
            """
            action_list = []
            parm = []

            env.reset()
            r_total = 0
            for t in range(horizon):
                s_t = env.state
                # print(s_t.timestep, env.t)
                print(f'\n>>>>>>>>>>>>>time: {t}<<<<<<<<<<<<<<<<<<<<<\n')
                print(env.mdp.state_string(s_t).replace('ø', 'o'))


                obs, reward, done, env_info = env.step(action_list[t], parm[t])
                print(env.mdp.get_utensil_states(s_t))
                ml_actions = obs.ml_actions
                skills = f""
                for i, ml_action in enumerate(ml_actions):
                    if ml_action == None:
                        continue
                    skills += f"P{i} finished <{ml_action}>. "
                print(skills)

                r_total += reward
                rprint("[red]" + f'r: {reward} | total: {r_total}\n\n')
            break

        
        channel_adapter = ChannelAdapter(
            variant['compressor'],
            channel_scope=variant['channel_scope'],
            **json.loads(variant['compressor_kwargs'] or '{}'),
        )
        channel_adapter.reset_episode()

        for alg in [p0_algo, p1_algo]:
            if alg == "LLMPair":
                if mode!="human":
                    assert variant['gpt_model']!=None, print(f'you should choose a gpt model')
                if mode == "OpenSource":
                    assert os.path.exists(variant['model_dirname']) is True, print(f"you should input right open-source model absolute path")
                print(f"\n----Use {variant['gpt_model']}----\n")
                if variant['gpt_model'] == "human":
                    assert check_port_in_use(variant["local_server_api"]) is True,print(f"port {variant['local_server_api']} is busy")
                    change_port(variant["local_server_api"])
                gpt_model = variant['gpt_model']
                model_dirname = variant['model_dirname']
                local_server_api = variant['local_server_api']
                retrival_method = variant['retrival_method']
                K = variant['K']
                agent = make_agent(alg, mdp, layout, model=gpt_model, model_dirname=model_dirname,local_server_api=local_server_api,
                                   retrival_method=retrival_method, K=K,actor=actor_list[actor_num])
                agent.channel = channel_adapter
            else:
                agent = make_agent(alg, mdp, layout)
            agents_list.append(agent)
            actor_num += 1

        team = AgentGroup(*agents_list)
        team.reset()

        env.reset()
        r_total = 0
        episode_start_time = time.time()

        
        if mode == 'exp':
            for t in range(horizon):
                s_t = env.state
                # print(s_t.timestep, env.t)
                print(f'\n>>>>>>>>>>>>>time: {t}<<<<<<<<<<<<<<<<<<<<<\n')
                map = env.mdp.state_string(s_t).replace('ø', 'o')
                print(map)   
                a_t, ingredient_for_pickup = team.joint_action(s_t) 
                print(a_t)
                dialogue_t = team.reset_dialogue()
                print(f"\n-----------Controller-----------\n")    
                print(f"action: P0 {Action.to_char(a_t[0])} | P1 {Action.to_char(a_t[1])}")
                parm = ingredient_for_pickup

                obs, reward, done, env_info = env.step(a_t,parm)

                ml_actions = obs.ml_actions
                skills = f""
                for i, ml_action in enumerate(ml_actions):
                    if ml_action == None:
                        continue
                    skills += f"P{i} finished <{ml_action}>. "
                print(skills)

                r_total += reward
                if reward>0:
                    statistics_dict['total_order_finished'].append(s_t.current_k_order[0])
                    team.agents[1].teammate_ml_actions.append({'timestamp':t,'action':"deliver_soup()"})
                rprint("[red]" + f'r: {reward} | total: {r_total}\n\n')
                print(f"P0's real behavior: {team.agents[1].teammate_ml_actions}")
                print(f"P1's real behavior: {team.agents[0].teammate_ml_actions}")


                #save statistics 
                turn_statistics_dict_agent0 = team.agents[0].turn_statistics_dict
                turn_statistics_dict_agent1 = team.agents[1].turn_statistics_dict

                turn_statistics_dict_both = combine_statistic_dict(turn_statistics_dict_agent0,turn_statistics_dict_agent1,map,reward)

                statistics_dict['total_timestamp'].append(t)
                statistics_dict['total_score'] = r_total
                statistics_dict['total_action_list'][0] = team.agents[1].teammate_ml_actions
                statistics_dict['total_action_list'][1] = team.agents[0].teammate_ml_actions
                statistics_dict['content'].append(turn_statistics_dict_both)
                #statistics_dict['end_time'] = time.strftime("%Y-%m-%d %H:%M:%S")
                with open(filename, 'w') as f:
                    json.dump(statistics_dict,f,indent=4)

                channel_stats_out = channel_adapter.finalize_stats()
                channel_stats_out['input_tokens'] = sum(
                    a.planner.cumulative_input_tokens for a in team.agents
                )
                channel_stats_out['output_tokens'] = sum(
                    sum(turn_statistics_dict_both['statistical_data']['communication'][idx]['token'])
                    for idx in (0, 1)
                )
                channel_stats_out['wall_time_seconds'] = time.time() - episode_start_time
                channel_stats_out['compressor'] = variant['compressor']
                channel_stats_out['channel_scope'] = variant['channel_scope']
                channel_stats_out['success'] = bool(r_total > 0)
                channel_stats_out['steps'] = t + 1
                with open(channel_stats_filename, 'w') as f:
                    json.dump(channel_stats_out, f, indent=4)

                if variant['test_mode'] == 'fix_task':
                    if reward != 0:
                        print("Task successed!")
                        #Human-eval: set task success message
                        if variant['gpt_model'] == "human":
                            for a in range(len(team.agents)):
                                output_to_port(f"agent{a}","Success!",mission="success",port=variant['local_server_api'])
                        break
            #Human-eval: set task failed message
            if variant['gpt_model'] == "human":
                for a in range(len(team.agents)):
                    output_to_port(f"agent{a}","Fail to finish task in time!",mission="fail",port=variant['local_server_api'])
        print(f"Episode {i+1}/{episode}: {r_total}\n====\n\n")
        results.append(r_total)
   
    end_time = time.time()
    print(f"Cost time : {end_time - start_time:.3f}s-----\n\n")


    
if __name__ == '__main__':

    parser = ArgumentParser(description='OvercookedAI Experiment')

    # these are basis parses
    parser.add_argument('--layout', '-l', type=str, default='new_env', choices=['new_env'])
    parser.add_argument('--p0',  type=str, default='LLMPair', choices=['LLMPair', 'Human'], help='Algorithm for P0 agent 0')
    parser.add_argument('--p1', type=str, default='LLMPair', choices=['LLMPair', 'Human'], help='Algorithm for P1 agent 1')
    parser.add_argument('--horizon', type=int, default=120, help='Horizon steps in one game')
    parser.add_argument('--episode', type=int, default=1, help='Number of episodes')

    # these parsers are only required when using LLMPair.

    # model:'gpt-3.5-turbo-0125', 'gpt-3.5-turbo', 'gpt-4', 'gpt-4o','gpt-o1mini','gpt4-turbo','llama3-8B','Llama-3.1-8B-Instruct','Llama-3.1-70B-Instruct',"Yi-1.2-34B","yi-lightning","yi-large",'yi-medium',"Qwen2.5-7B-Instruct","Qwen2.5-72B-Instruct","Qwen2.5-14B-Instruct","Qwen2.5-32B-Instruct",'claude3_sonnet'
    parser.add_argument('--gpt_model', type=str, default='gpt-3.5-turbo-0125')
    
    parser.add_argument('--retrival_method', type=str, default="recent_k", choices=['recent_k', 'bert_topk'], help='Use similarity-based(BERT, CLIP) retrieval or retrieve recent K history in dialog.')
    parser.add_argument('--K', type=int, default=0, help="The number of dialogues you want to retrieve.")

    # 
    parser.add_argument('--model_dirname', type=str, default='.', help='absolute path of open-source model')      
    parser.add_argument('--local_server_api', type=str, default= "http://localhost:8000/v1", help='IP and port address to connect with local open source llm')     
    parser.add_argument('--mode', type=str, default='exp', choices=['exp', 'debug_validator', 'develop'], help='exp mode run step-by-step, demo mode run via traj')                                
    parser.add_argument('--test_mode', type=str, default='fix_task', choices=['fix_task', 'fix_time'])
    parser.add_argument('--save', type=boolean_argument, default=True, help='Whether save the result')
    parser.add_argument('--log_dir', type=str, default=None, help='dir to save result')
    parser.add_argument('--debug', type=boolean_argument, default=True, help='debug mode')
    parser.add_argument('--order', type=str, default="", help='1 task order name')

    #
    parser.add_argument('--statistics_save_dir', type=str, default='data', help='save directory of LLM statistics')

    # channel-compression middleware (mirrors hiddenbench.channel.COMPRESSOR_REGISTRY)
    parser.add_argument('--compressor', type=str, default='identity',
                         choices=['identity', 'window', 'novelty', 'llmlingua2', 'learned',
                                  'rewriter', 'backref', 'codebook', 'adaptive', 'gzip64', 'stack'],
                         help='Channel compressor (identity = original, unmodified protocol)')
    parser.add_argument('--compressor-kwargs', type=str, default='{}',
                         help='JSON dict of extra kwargs for the compressor constructor')
    parser.add_argument('--channel-scope', type=str, default='episode', choices=['episode', 'timestep'],
                         help='How often stateful compressors reset: once per episode or per timestep')
    parser.add_argument('--run-tag', type=str, default='',
                         help='Overrides the log-directory model tag (e.g. "Qwen-Qwen3-4B__backref_floor"); '
                              'the real --gpt_model string is still used for API dispatch')

    args = parser.parse_args()
    variant = vars(args)
    variant['model_tag'] = args.run_tag if args.run_tag else args.gpt_model

    start_time = time.time()
    main(variant)
    end_time = time.time()
    print(f"\n=======Finshed all=========\n")
    print(f"Cost time : {end_time - start_time:.3f}s-----\n\n")
