import json
import os
import pandas as pd
import matplotlib.pyplot as plt
import argparse
import numpy as np
from tqdm import tqdm
import tiktoken
import re


enc = tiktoken.encoding_for_model("gpt-4o")

PUZZLE_LIST = ["SimpleWire", "Telehealth", "Who", "Led", "Memory", "KeyPad", "Password", "Colour", "Maze", "Atm"]
CLOSED_SOURCE_MODELS = ["GPT4o", "GPT4V", "Gemini"]
OPEN_SOURCE_MODELS = ["Human", "LLAVA", "QwenVL", "InternVL", "LLaMA32", "Random"]

def calculate_token_usage(conversation, encoder, response_data=None):
    solver_tokens = 0
    expert_tokens = 0
    total_solver_reasoning_tokens = 0
    total_expert_reasoning_tokens = 0

    # Add reasoning tokens from response_data if provided
    if response_data:
        for i, entry in enumerate(response_data):
            if i % 2 == 0:
                total_solver_reasoning_tokens += entry["completion_tokens_details"].get("reasoning_tokens", 0)
            else:
                total_expert_reasoning_tokens += entry["completion_tokens_details"].get("reasoning_tokens", 0)

    for message in conversation:
        role = message.get("from")
        value = message.get("value", "")
        total_tokens = len(encoder.encode(value))

        # Match either <think>...</think> or <REASONING>...</REASONING>
        reasoning_match = re.search(r"<(?:think|REASONING)>(.*?)</(?:think|REASONING)>", value, re.DOTALL)
        reasoning_tokens = 0
        if reasoning_match:
            reasoning_text = reasoning_match.group(1)
            reasoning_tokens = len(encoder.encode(reasoning_text))

        normal_tokens = total_tokens - reasoning_tokens

        if role == "SOLVER":
            solver_tokens += normal_tokens
            total_solver_reasoning_tokens += reasoning_tokens
        elif role == "EXPERT":
            expert_tokens += normal_tokens
            total_expert_reasoning_tokens += reasoning_tokens

    return solver_tokens, expert_tokens, total_solver_reasoning_tokens, total_expert_reasoning_tokens


def setup_plot_style(is_black_background=False):
   plt.rcParams.update({
       'pdf.fonttype': 42,
       'font.size': 14,
       'axes.labelsize': 14,
       'axes.titlesize': 14,
       'xtick.labelsize': 12,
       'ytick.labelsize': 12,
       'figure.titlesize': 14,
       'legend.fontsize': 14,
       'pdf.use14corefonts': True
   })
   if is_black_background:
       plt.style.use('dark_background')
       plt.rcParams.update({
           'ytick.color': 'w',
           'xtick.color': 'w',
           'axes.labelcolor': 'w',
           'axes.edgecolor': 'w'
       })
       
setup_plot_style()

def load_file_jsonl(path):
   if not os.path.exists(path):
       return None
   with open(path) as f:
        return [json.loads(row) for row in f]

def main(args):
    result_folder = args.result_folder
    conv_length_data = {}

    print("Going through possible conversation lengths ...")
    for conv_length in tqdm(range(1, 11)):
        all_info = {}

        for file in sorted(os.listdir(result_folder)):
            info = file.split("_")
            solver = info[0]
            if len(info) >= 2:
                expert = solver
            #expert = info[2] if len(info) >= 2 else solver
            all_info[f"{solver} SOLVER <-> {expert} EXPERT"] = {}
            overall_completed = []
            overall_mistakes = []
            overall_partial_scores = []
            overall_average_conversation_length = []
            overall_average_word_count = []
            grand_total = 0
            
            for puzzle in os.listdir(os.path.join(result_folder, file)):
                total_runs = len(os.listdir(os.path.join(result_folder, file, puzzle)))
                mistakes_for_each_run = []
                word_count_each_run = []
                partial_scores = []
                success_scores = []
                conversation_length_each_run = []
                word_count = 0
                for run in os.listdir(os.path.join(result_folder, file, puzzle)):
                    mistakes = 0
                    solver_expert_messages = 0
                    total_lines = 0
                    conversation = load_file_jsonl(os.path.join(result_folder, file, puzzle, run, "conversation.jsonl"))
                    if not conversation:
                        print(f"Warning, {os.path.join(result_folder, file, puzzle, run)} had no conversation.jsonl file!!!")
                        continue
                    for message in conversation:
                        if solver_expert_messages > 2 * conv_length:
                            break
                        if message['from'] == 'SOLVER' or message['from'] == 'EXPERT':
                            solver_expert_messages += 1
                        total_lines += 1

                    conversation = conversation[:total_lines + 1]

                    if len(conversation) < 4:
                        continue
                    if "score" in conversation[-1] and conversation[-1]["score"] != "None":
                        
                        if conversation[-2]["value"].startswith("Puzzle successfully finished"):
                            partial_scores.append(1.0)
                            overall_partial_scores.append(1.0)
                            
                        else:
                            partial_scores.append(float(conversation[-1]["score"]) * 1)
                            overall_partial_scores.append(float(conversation[-1]["score"]))
                    else: # Just to deal with cases where there is no partial scoring
                        if conversation[-2]["value"].startswith("Puzzle successfully finished") or "value" in conversation[-1] and conversation[-1]["value"].startswith("Puzzle successfully finished"):
                            partial_scores.append(1)
                            overall_partial_scores.append(1)
                        else:
                            partial_scores.append(0)
                            overall_partial_scores.append(0)

                    if conversation[-2]["value"].startswith("Puzzle successfully finished") or "value" in conversation[-1] and conversation[-1]["value"].startswith("Puzzle successfully finished"):
                        success_scores.append(1)
                        overall_completed.append(1)
                    else:
                        success_scores.append(0)
                        overall_completed.append(0)
                        
                    solver_expert_messages = 0
                    for message in conversation:
                        if message["from"] == "SOLVER" or message["from"] == "EXPERT":
                            solver_expert_messages += 1
                            if message['value']:
                                word_count += len(message["value"].split(" "))
                        if message["from"] == "ENVIRONMENT" and "value" in message and message["value"].startswith("That action seems to have been a mistake."):
                            mistakes += 1

                    mistakes_for_each_run.append(mistakes)
                    overall_mistakes.append(mistakes)
                    conversation_length_each_run.append(solver_expert_messages // 2)
                    word_count_each_run.append(word_count)
                    overall_average_word_count.append(word_count)
                    overall_average_conversation_length.append(solver_expert_messages // 2)
                    grand_total += 1
                    
                avg_mistakes = 0 if len(mistakes_for_each_run) == 0 else sum(mistakes_for_each_run) / len(mistakes_for_each_run)
                
                mean_success = sum(success_scores) / total_runs
                mean_partial_success = sum(partial_scores) / total_runs
                mean_conv_length = sum(conversation_length_each_run) / len(conversation_length_each_run)
                mean_word_count = sum(word_count_each_run) / max(1, len(word_count_each_run))
                
                all_info[f"{solver} SOLVER <-> {expert} EXPERT"][puzzle] = {}
                all_info[f"{solver} SOLVER <-> {expert} EXPERT"][puzzle]["Average Success Rate"] = sum(success_scores) / total_runs
                all_info[f"{solver} SOLVER <-> {expert} EXPERT"][puzzle]["STDDEV Success Rate"] =  np.sqrt(mean_success * (1 - mean_success) / len(success_scores))
                
                all_info[f"{solver} SOLVER <-> {expert} EXPERT"][puzzle]["Average Word Count"] = mean_word_count
                all_info[f"{solver} SOLVER <-> {expert} EXPERT"][puzzle]["STDDEV Word Count"] = np.std(word_count_each_run, ddof=1) / np.sqrt(max(1, len(word_count_each_run)))
                
                all_info[f"{solver} SOLVER <-> {expert} EXPERT"][puzzle]["Average Mistake Rate"] = avg_mistakes
                all_info[f"{solver} SOLVER <-> {expert} EXPERT"][puzzle]["STDDEV Mistake Rate"] = np.std(mistakes_for_each_run, ddof=1) / np.sqrt(max(1, len(conversation_length_each_run)))
                
                all_info[f"{solver} SOLVER <-> {expert} EXPERT"][puzzle]["Average Conversation Length"] = mean_conv_length
                all_info[f"{solver} SOLVER <-> {expert} EXPERT"][puzzle]["STDDEV Conversation Length"] = np.std(conversation_length_each_run, ddof=1) / np.sqrt(max(1, len(conversation_length_each_run)))
                
                all_info[f"{solver} SOLVER <-> {expert} EXPERT"][puzzle]["Highest Partial Score"] = max(partial_scores)
                all_info[f"{solver} SOLVER <-> {expert} EXPERT"][puzzle]["Average Partial Score"] = mean_partial_success
                all_info[f"{solver} SOLVER <-> {expert} EXPERT"][puzzle]["STDDEV Partial Score"] = np.std(partial_scores, ddof=1) / np.sqrt(max(1, len(partial_scores)))
                
            all_info[f"{solver} SOLVER <-> {expert} EXPERT"]["Overall"] = {}
            all_info[f"{solver} SOLVER <-> {expert} EXPERT"]["Overall"]["Overall Average Success Rate"] = sum(overall_completed) / max(1, len(overall_completed)) * 100
            
            mean_overall_sucess = sum(overall_completed) / max(1, len(overall_completed))
            all_info[f"{solver} SOLVER <-> {expert} EXPERT"]["Overall"]["STDDEV Average Success Rate"] = np.sqrt(mean_overall_sucess * (1 - mean_overall_sucess) / len(overall_completed))
            
            all_info[f"{solver} SOLVER <-> {expert} EXPERT"]["Overall"]["Overall Average Mistake Rate"] = sum(overall_mistakes) / max(1, len(overall_mistakes))
            all_info[f"{solver} SOLVER <-> {expert} EXPERT"]["Overall"]["STDDEV Average Mistake Rate"] = np.std(overall_mistakes, ddof=1) / np.sqrt(max(1, len(overall_mistakes)))
            
            
            all_info[f"{solver} SOLVER <-> {expert} EXPERT"]["Overall"]["Overall Average Partial Score"] = sum(overall_partial_scores) / max(1, len(overall_partial_scores)) * 100
            all_info[f"{solver} SOLVER <-> {expert} EXPERT"]["Overall"]["STDDEV Average Partial Score"] = np.std(overall_partial_scores, ddof=1) / np.sqrt(max(1, len(overall_partial_scores)))
            
            
            all_info[f"{solver} SOLVER <-> {expert} EXPERT"]["Overall"]["Overall Average Conversation Length"] = sum(overall_average_conversation_length) / max(1, len(overall_average_conversation_length))
            all_info[f"{solver} SOLVER <-> {expert} EXPERT"]["Overall"]["STDDEV Average Conversation Length"] = np.std(overall_average_conversation_length, ddof=1) / np.sqrt(max(1, len(overall_average_conversation_length)))
            
            all_info[f"{solver} SOLVER <-> {expert} EXPERT"]["Overall"]["Overall Average Word Count"] =  sum(overall_average_word_count) / max(1, len(overall_average_word_count))
            #all_info[f"{solver} SOLVER <-> {expert} EXPERT"]["Overall"]["STDDEV Average Word Count"] =  sum(overall_average_word_count) / max(1, len(overall_average_word_count))
            

        conv_length_data[conv_length] = all_info

    for metric in ["Partial Score", "Success Rate", "Mistake Rate", "Conversation Length"]:
        table_rows = []
        results = pd.DataFrame()
        final_result_table = pd.DataFrame()
        for pair in all_info:
            chunks = pair.split(" ")
            solver, expert = chunks[0], chunks[-2]
            flattened_data = []
            for puzzle, stats in all_info[pair].items():
                row = {'Puzzle': puzzle}
                row.update(stats)
                flattened_data.append(row)

            # Create a DataFrame
            df = pd.DataFrame(flattened_data)
            df['Solver'] = solver
            df['Expert'] = expert
            results = pd.concat([results, df]).reset_index(drop=True)

            # Display the table
            df_row = {}
            solver_expert_pair = f"{solver[:-5]} & {expert[:-5]}"
            df_row['Agents'] = solver_expert_pair

            table_row = f"{solver[:-5]} & {expert[:-5]} & "
            for puzzle in PUZZLE_LIST:
                
                stat = df[df["Puzzle"] == f"{puzzle}Puzzle"]['Average ' + metric]
                if len(stat) == 0:
                    stat = "?"
                else:
                    
                    if metric in ['Partial Score', 'Success Rate']:
                        p_score = round(df[df["Puzzle"] == f"{puzzle}Puzzle"]['Average ' + metric].item() * 100)
                        stddev = round(df[df["Puzzle"] == f"{puzzle}Puzzle"]['STDDEV ' + metric].item() * 100, 2)
                        stat = f"{p_score} ± {stddev:.1f}"
                    else:
                        p_score = round(df[df["Puzzle"] == f"{puzzle}Puzzle"]['Average ' + metric].item(), 2)
                        stddev = round(df[df["Puzzle"] == f"{puzzle}Puzzle"]['STDDEV ' + metric].item(), 2)
                        
                        stat = f"{p_score:.2f} ± {stddev:.1f}"
                        

                df_row[puzzle] = stat 
                table_row += f"{stat} & "

            overall_metric = f'{round(all_info[pair]["Overall"]["Overall Average " + metric], 2):.2f}'
            
            stddev = all_info[f"{solver} SOLVER <-> {expert} EXPERT"]["Overall"][f"STDDEV Average {metric}"]
            if metric in ['Partial Score', 'Success Rate']:
                stddev *= 100
            df_row['overall'] = stat = f"{overall_metric} ± {stddev:.1f}"
            df_row = pd.DataFrame(df_row, index=[0])
            final_result_table = pd.concat([df_row, final_result_table]).reset_index(drop=True)

            table_row += overall_metric
            table_rows.append(table_row)

        print(f"====={metric} Results=====")
        print(final_result_table.sort_values(by='overall'))
        for i, row in final_result_table.sort_values(by='overall', ascending=False).iterrows():
            trow = ""
            trow += row['Agents'].split(" & ")[0] + " & "
            for p in PUZZLE_LIST:
                trow += f"{row[p]} & "
            trow += f"{row['overall']}"
            print(trow)

    print("Gathering token usage statistics ...")
    token_usage_info = {}
    for file in sorted(os.listdir(result_folder)):
        info = file.split("_")
        solver = info[0]
        if len(info) >= 2:
            expert = solver
        #all_info[f"{solver} SOLVER <-> {expert} EXPERT"] = {}
        token_usage_info[f"{solver} SOLVER <-> {expert} EXPERT"] = {}
        overall_solver_tokens = []
        overall_expert_tokens = []
        overall_solver_reasoning_tokens = []
        overall_expert_reasoning_tokens = []

        for puzzle in os.listdir(os.path.join(result_folder, file)):
            solver_tokens_per_puzzle = []
            expert_tokens_per_puzzle = []
            solver_reasoning_tokens_per_puzzle = []
            expert_reasoning_tokens_per_puzzle = []

            for run in os.listdir(os.path.join(result_folder, file, puzzle)):
                conversation = load_file_jsonl(os.path.join(result_folder, file, puzzle, run, "conversation.jsonl"))
                if not conversation:
                    print(f"Warning, {os.path.join(result_folder, file, puzzle, run)} had no conversation.jsonl file!!!")
                    continue
                response_data = None
                if os.path.exists(os.path.join(result_folder, file, puzzle, run, "response_data.jsonl")):
                    response_data = load_file_jsonl(os.path.join(result_folder, file, puzzle, run, "response_data.jsonl")) 
                solver_tokens, expert_tokens, solver_reasoning_tokens, expert_reasoning_tokens = calculate_token_usage(conversation, enc, response_data = response_data)
                if solver == "HumanAgent":
                    expert_tokens = solver_tokens
                    expert_reasoning_tokens = solver_reasoning_tokens
                solver_tokens_per_puzzle.append(solver_tokens)
                expert_tokens_per_puzzle.append(expert_tokens)
                overall_solver_tokens.append(solver_tokens)
                overall_expert_tokens.append(expert_tokens)
                
                solver_reasoning_tokens_per_puzzle.append(solver_reasoning_tokens)
                expert_reasoning_tokens_per_puzzle.append(expert_reasoning_tokens)
                overall_solver_reasoning_tokens.append(solver_reasoning_tokens)
                overall_expert_reasoning_tokens.append(expert_reasoning_tokens)

            token_usage_info[f"{solver} SOLVER <-> {expert} EXPERT"][puzzle] = {
                    "Average Solver Tokens": np.mean(solver_tokens_per_puzzle) if solver_tokens_per_puzzle else 0,
                    "STDERR Solver Tokens": np.std(solver_tokens_per_puzzle, ddof=1) / np.sqrt(len(solver_tokens_per_puzzle)) if solver_tokens_per_puzzle else 0,
                    "Average Expert Tokens": np.mean(expert_tokens_per_puzzle) if expert_tokens_per_puzzle else 0,
                    "STDERR Expert Tokens": np.std(expert_tokens_per_puzzle, ddof=1) / np.sqrt(len(expert_tokens_per_puzzle)) if expert_tokens_per_puzzle else 0,
                    "Average Solver Reasoning Tokens": np.mean(solver_reasoning_tokens_per_puzzle) if solver_reasoning_tokens_per_puzzle else 0,
                    "STDERR Solver Reasoning Tokens": np.std(solver_reasoning_tokens_per_puzzle, ddof=1) / np.sqrt(len(solver_reasoning_tokens_per_puzzle)) if solver_reasoning_tokens_per_puzzle else 0,
                    "Average Expert Reasoning Tokens": np.mean(expert_reasoning_tokens_per_puzzle) if expert_reasoning_tokens_per_puzzle else 0,
                    "STDERR Expert Reasoning Tokens": np.std(expert_reasoning_tokens_per_puzzle, ddof=1) / np.sqrt(len(expert_reasoning_tokens_per_puzzle)) if expert_reasoning_tokens_per_puzzle else 0,
                }

        token_usage_info[f"{solver} SOLVER <-> {expert} EXPERT"]["Overall"] = {
            "Overall Average Solver Tokens": np.mean(overall_solver_tokens) if overall_solver_tokens else 0,
            "Overall STDERR Solver Tokens": np.std(overall_solver_tokens, ddof=1) / np.sqrt(len(overall_solver_tokens)) if overall_solver_tokens else 0,
            "Overall Average Expert Tokens": np.mean(overall_expert_tokens) if overall_expert_tokens else 0,
            "Overall STDERR Expert Tokens": np.std(overall_expert_tokens, ddof=1) / np.sqrt(len(overall_expert_tokens)) if overall_expert_tokens else 0,
            "Overall Average Solver Reasoning Tokens": np.mean(overall_solver_reasoning_tokens) if overall_solver_reasoning_tokens else 0,
            "Overall STDERR Solver Reasoning Tokens": np.std(overall_solver_reasoning_tokens, ddof=1) / np.sqrt(len(overall_solver_reasoning_tokens)) if overall_solver_reasoning_tokens else 0,
            "Overall Average Expert Reasoning Tokens": np.mean(overall_expert_reasoning_tokens) if overall_expert_reasoning_tokens else 0,
            "Overall STDERR Expert Reasoning Tokens": np.std(overall_expert_reasoning_tokens, ddof=1) / np.sqrt(len(overall_expert_reasoning_tokens)) if overall_expert_reasoning_tokens else 0,
        }


    # Generate token usage table
    for metric in ["Solver Tokens", "Expert Tokens", "Solver Reasoning Tokens", "Expert Reasoning Tokens"]:
        table_rows = []
        final_result_table = pd.DataFrame()
        for pair in token_usage_info:
            chunks = pair.split(" ")
            solver, expert = chunks[0], chunks[-2]
            df_row = {}
            solver_expert_pair = f"{solver[:-5]} & {expert[:-5]}"
            df_row['Agents'] = solver_expert_pair

            table_row = f"{solver[:-5]} & {expert[:-5]} & "
            for puzzle in PUZZLE_LIST:
                if puzzle + "Puzzle" in token_usage_info[pair]:
                    avg_tokens = token_usage_info[pair][puzzle + "Puzzle"]["Average " + metric]
                    stderr_tokens = token_usage_info[pair][puzzle + "Puzzle"]["STDERR " + metric]
                    stat = f"{avg_tokens:.2f} ± {stderr_tokens:.2f}"
                else:
                    stat = "?"
                df_row[puzzle] = stat
                table_row += f"{stat} & "

            overall_metric = token_usage_info[pair]["Overall"]["Overall Average " + metric]
            df_row['overall'] = stat = f"{overall_metric:.2f}"
            df_row = pd.DataFrame(df_row, index=[0])
            final_result_table = pd.concat([df_row, final_result_table]).reset_index(drop=True)

            table_row += f"{overall_metric:.2f}"
            table_rows.append(table_row)

        print(f"====={metric} Results=====")
        print(final_result_table.sort_values(by='overall'))
            
        fig, ax = plt.subplots(figsize=(10,7))
        #to_plot = {"Human": "#98D8AA", "OneVision":"#ff7f00", "GPT4o": "#8D6F64", "InternVL": "#b15928", "GPT4V": "#a6cee3", "Gemini": "skyblue", "LLAVA": "#F7D060", "QwenVL": "#b069db", "LLaMA32": "#7358DC", "Random": "#CECECE", "GPT": "#fb9a99"}
        to_plot = {"Human": "#48CFAD", 
                "GPT": "#ED5565",
                "GPT4o": "#4FC1E9", 
                "Gemini": "#FFCE54",
                "GPT4V": "#A0D468",
                "QwenVL": "#AC92EC",
                "LLaMA32": "#2d568c",
                "InternVL": "#EC87C0",
                "Random": "#CCD1D9",
                "LLAVA": "black",
                "OneVision":"#656D78",
                "LLaVACoT": "#ffab0f",
                }

        stat_to_plot = 'Success Rate'

        label_remapping = {
            "GPT": "o4-mini",
            "LLAVA": "LLaVA",
            "LLaMA32": "LLaMA 3.2",
            "OneVision": "R1-OneVision",
            "LLaVACoT": "LLaVA-CoT",
        }

        reasoning_models = ["o4-mini", "R1-OneVision", "LLaVA-CoT"]
        proprietary_general_models = ["Gemini", "GPT4o", "GPT4V"]
        open_source_models = ["LLaVA", "QwenVL", "InternVL", "LLaMA32"]

        for model in sorted(conv_length_data[1].keys(), key = lambda x: conv_length_data[10][x]["Overall"]["Overall Average "+ stat_to_plot], reverse=True):
            linestyle = None
            linewidth = 4
            label = model.split()[0][:-5]
            marker_size = 100
            
            fig_label = label_remapping.get(label, label)
            

            if "HumanAgent" in model:
                marker = "d"
                marker_size = 150
            elif fig_label in reasoning_models:
                marker="s"
            elif fig_label in open_source_models:
                marker="o"
            elif fig_label in proprietary_general_models:
                marker="^"
                marker_size = 150
            else:
                marker = "o"
                

            y_values = [conv_length_data[length][model]["Overall"]["Overall Average "+ stat_to_plot] for length in range(1, 11)]

            ax.plot([0] + list(range(1, 11)), [0] + y_values, linestyle=linestyle, linewidth=linewidth, color=to_plot.get(label, "black"), zorder=-1)
            ax.scatter([0] + list(range(1, 11)), [0] + y_values, label=fig_label, marker=marker, linewidth=2, s = marker_size, color=to_plot.get(label, "black"), edgecolors='black')


        # Set x-axis ticks to be integers from 1 to 20
        ax.set_xticks(range(1, 11))
        ax.set_ylim(-1, 73)

        # Optionally, you can set the x-axis labels if you want to customize them
        ax.set_xticklabels(range(1, 11))

        ax.set_xlabel("Conversation Length", fontsize=23)
        ax.set_ylabel(f"{stat_to_plot}", fontsize=23)

        ax.tick_params(axis='x', which='major', labelsize=20)
        ax.tick_params(axis='y', which='major', labelsize=20)


        plt.legend(fontsize=15, loc='upper left', ncol=2)

        plt.tight_layout()
        plt.savefig("Conversation_Length_Success_Rate.pdf", dpi = 600)
        
        token_usages = []
        performances = []
        names = []
        scores = []
        renaming_mapping = {
            "GPT": "o4-mini",
            "GPT4V": "GPT-4V",
            "GPT4o": "GPT-4o",
            "LLaVACoT": "LLaVA-CoT",
            "OneVision": "R1-OneVision",
            "LLAVA": "LLaVA",
            "LLaMA32": "LLaMA 3.2"
        }

        for i, (pair, stats) in enumerate(token_usage_info.items()):
            total_tokens = stats["Overall"]["Overall Average Solver Tokens"] + stats["Overall"]["Overall Average Expert Tokens"] + stats["Overall"]["Overall Average Solver Reasoning Tokens"] + stats["Overall"]["Overall Average Expert Reasoning Tokens"]
            token_usages.append(total_tokens)
            name = pair.split()[0][:-5]
            names.append(renaming_mapping.get(name, name))
            performances.append(conv_length_data[10][pair]["Overall"]["Overall Average Partial Score"])
            concise_score = 1.0 / (1.0 + token_usages[i] / 1000.0)
            metric = 2 * performances[i] / 100 * concise_score / (performances[i] / 100 + concise_score)
            #metric = 0.75 * performances[i] / 100 + 0.25 * (1.0 / (1.0 + token_usages[i] / 1000.0))
            scores.append(metric)

            import matplotlib.colors as mcolors

            # Compute normalized scores
            raw_values = scores

            # Normalize to range [0, 1] for color mapping
            min_val = min(raw_values)
            max_val = max(raw_values)

            # Avoid divide-by-zero
            range_val = max_val - min_val if max_val > min_val else 1e-6
            normalized = [(val - min_val) / range_val for val in raw_values]

            # Define custom colormap from #FF6D60 to lightgreen
            custom_cmap = mcolors.LinearSegmentedColormap.from_list("custom", ["#FF6D60", "lightgreen"])

            # Map normalized values to RGBA colors using the custom colormap
            colors = [custom_cmap(val) for val in normalized]

            # Create the scatter plot
            plt.figure(figsize=(14, 7))
            scatter = plt.scatter(
                x=[np.log10(val) for val in token_usages],  # Log scale for x-axis
                y=performances,
                s=[score * 1600 for score in scores],  # Scale the size of the points
                c=colors,  # Use the colors list for point colors
                alpha=0.8,
                edgecolors='black'
            )

            # Add labels to the points
            label_fontsize = 16
            for i, name in enumerate(names):
                score = f"{scores[i]:.2f}"
                if name == "Random":
                    plt.text(np.log10(token_usages[i]) + 0.05, performances[i], name + f" ({score})", fontsize=label_fontsize, ha='left', va='bottom')
                elif name == "R1-OneVision":
                    plt.text(np.log10(token_usages[i]) - 0.03, performances[i] + 1.2, name + f" ({score})", fontsize=label_fontsize, ha='right', va='center')
                elif name == "InternVL":
                    plt.text(np.log10(token_usages[i]) + 0.04, performances[i] + 0.25, name + f" ({score})", fontsize=label_fontsize, ha='left', va='center')
                elif name == "LLaVA":
                    plt.text(np.log10(token_usages[i]) - 0.06, performances[i] - 0.5, name + f" ({score})", fontsize=label_fontsize, ha='right', va='center')
                elif name == "LLaMA 3.2":
                    plt.text(np.log10(token_usages[i]) - 0.06, performances[i] + 0.5, name + f" ({score})", fontsize=label_fontsize, ha='right', va='center')
                else:
                    plt.text(np.log10(token_usages[i]) - 0.06, performances[i], name + f" ({score})", fontsize=label_fontsize, ha='right', va='center')


            # Draw an arrow from (2, 50) to the top left corner of the plot with label "More Efficient"
            plt.annotate(
                "More Efficient",
                xy=(2, 50),  # Starting point of the arrow
                xytext=(1.5, 60),  # Ending point of the arrow
                arrowprops=dict(arrowstyle="<-", lw=2, color='black'),
                fontsize=16,
                ha='center',
                va='center'
            )

            # Set axis labels and title
            plt.xlabel('Token Usage (Log Average Tokens Used Per Puzzle)', fontsize=18)
            plt.ylabel('Performance (Partial Score %)', fontsize=18)
            plt.xticks(fontsize=16)
            plt.yticks(fontsize=16)
            # Show the plot
            plt.tight_layout()
            plt.savefig("token_usage_vs_performance.pdf", dpi=300)


        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--result_folder", type=str, default='./outputs',
                        help="Path to folder which contains results")
    args = parser.parse_args()
    main(args)