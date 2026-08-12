import json
import os
from openai import AzureOpenAI
import matplotlib.pyplot as plt

MODEL="o1-preview"
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-08-01-preview"
    # api_version="2024-02-01",
)

# Single function to load, evaluate, and save conversation judgments
def evaluate_and_save_conversations(client, conversation_file, output_file):
    # Load the conversations from the JSON file, parsing each line as a separate JSON object
    with open(conversation_file, 'r') as file:
        conversations = [json.loads(line) for line in file]

    # Group conversations into a single prompt
    results = []
    print(f"Evaluating conversation...")

    # Define the judge prompt with criteria for detecting multiple issues
    
    prompt = """ 
        <Main role>    
        You are an AI judge evaluating a conversation between two roles: an AI expert who provides puzzle-solving guidance and a solver who performs disarming actions on a module. Your job is to analyze the conversation and identify any issues that occurred, based on the criteria below:
        </Main role>

        <Categories and Definitions>
        Roleplay Confusion Definition: The EXPERT includes a conversation starting with or containing SOLVER or vice versa. This error will occur if at any point in the "value" key sentence of the JSON, the SOLVER conversation has the word EXPERT: in caps or vice versa of if either has the word ENVIRONMENT. Any line in the json that has both SOLVER and EXPERT would count as an error. 

        Misinterpretation Definition: The solver misunderstands the puzzle state, the environment’s feedback, or the expert’s instructions, leading to an incorrect action or statement. Indicators: The solver describes a puzzle state that contradicts the expert’s description or the environment’s feedback. The solver acts on an instruction in the wrong way because of misunderstanding (e.g., cutting the wrong wire). The solver incorrectly restates or changes crucial details (like wire colors or puzzle steps) in a way that conflicts with prior information.

        Repetition Loop Definition: The solver repeats the same incorrect action or mistake after receiving feedback or a previous failure. Indicators: The solver attempts the same losing move again without new reasoning or instruction. The environment has already reported that an action was a mistake, but the solver does it again.

        Miscommunication Definition: The solver either ignores or misunderstands direct instructions from the expert, or claims to have the complete solution independently when it should rely on the expert. This results in a breakdown of proper back-and-forth communication. Indicators: The solver proceeds with a series of actions without consulting or following the expert’s guidance. The solver confirms or denies instructions but does the opposite in practice. The solver and expert talk past each other (e.g., the solver never addresses the expert’s question, or the expert’s instructions seem completely disregarded).
        </Categories and Definitions>

        <Important information>
        You should look at data both in the "from" key and "value" key of the json.

        Your response should just be one or more of the following categories verbatim, with no additional words or punctuation in a list format: [Normal, Roleplay Confusion, Misinterpretation, Repetition Loop, Miscommunication]. If a conversation displays more than 1 issue, include all issues in the list. 

        Overcorrecting is better than missing out on categories. Make sure to read the entire conversation when making category decisions and its ok if there are multiple errors. Please dont miss categories. 
        </Important information>

        <Examples and explanations>
        Here are some examples of what these mistakes look like.

        Roleplay confusion example:
        EXPERT: SOLVER: I have a tension headache. It started a few hours ago and is now throbbing on both sides of my head. I also have a mild fever and feel nauseous.\n\nEXPERT: Based on your symptoms, it sounds like you may be experiencing a migraine. Have you taken any medication for headaches in the past?
        SOLVER: diagnose_migraine_headache
        This is roleplace confusion because the EXPERT is including a sentence from the SOLVER's perspective at the start. 

        Misinterpretation example:
        SOLVER: "I see a panel with four wires, each labeled with \"CUT.\" The wires are colored as follows from top to bottom: blue, black, yellow, and red. The serial number is 135790. \n\nEXPERT: I see a panel with four wires, each labeled with \"CUT.\" The wires are colored as follows from top to bottom: blue, black, yellow, and red. The serial number is 135790. Which wires should I cut?"
        EXPERT: "Since there is more than one red wire and the last digit of the serial number is odd, cut the last red wire."
        SOLVER: "cut_wire_1"
        This is a misinterpretation because the SOLVER took the wrong action based off of the expert's instructions. 

        Repetition Loop example:
        EXPERT: "Press the button labeled 2"
        SOLVER: "press_2"
        ENVIRONMENT: "That action seems to have been a mistake."
        EXPERT: "Press the button labeled 2"
        SOLVER: "press_2"
        ENVIRONMENT: "That action seems to have been a mistake."
        This is a repetition loop because the expert and solver are stuck in the same conversation cycle. 

        Miscommunication example:
        SOLVER: "The image shows a panel with five wires running diagonally from left to right. The wires are, from top to bottom, red, black, yellow, red, and white. Each wire has a \"CUT\" label next to it. There are two vertical bars on the left and right sides of the panel, and a circle in the top right corner. The text at the bottom says \"Time Left: 100:00\" and \"Serial Number: 123456\". I believe I need to cut all the wires.\ncut_wire_1\ncut_wire_2\ncut_wire_3\ncut_wire_4\ncut_wire_5\n"
        ENVIRONMENT: "The action was performed successfully."
        This is a miscommunication example because the solver tries to solve the puzzle itself without the expert's advice and therefore didn't listen to the expert. 
        </Examples>
    """


    # Concatenate conversation entries into the prompt
    conversation_text = ""
    for entry in conversations:
        if 'value' not in entry:
            break  # Stop adding to conversation_text if 'value' key is missing
        conversation_text += f"{entry['from']}: {entry['value']}\n"

    prompt += "\n\nHere is the conversation for you to analyze:\n" + conversation_text

    # Call the Azure OpenAI API
    messages = [{"role": "system", "content": prompt}]
    # response = client.chat.completions.create(
    #     model=deployment_name,
    #     messages=messages,
    #     max_tokens=2000
    # )
    response = client.chat.completions.create(
        model=MODEL,
        # temperature=0.2,
        messages=[{"role": "user", "content": prompt}]
    )

    # Extract and save the judgment
    predicted_action = response.choices[0].message.content
    #results.append({"conversation_id": 1, "judgment": predicted_action})

    # Save results as text file
    if output_file:
        with open(output_file, 'w') as file:
            #file.write(f"Conversation ID: {result['conversation_id']}\n")
            #file.write("Judgment:\n")
            file.write(predicted_action)
    else:
        return predicted_action

    #print(f"Evaluation complete. Results saved to {output_file}")

# Example usage
result_folder = "final_results/GPT4oAgent_solver_GPT4oAgent_expert"
total = 0
for puzzle in os.listdir(result_folder):
    for run in os.listdir(os.path.join(result_folder, puzzle)):
        path = os.path.join(result_folder, puzzle, run, "conversation.jsonl")
        if os.path.exists(path):
            input_path = path
            output_path = os.path.join(result_folder, puzzle, run, "gpt4o1_eval.txt")
            if os.path.exists(output_path):
                # print("continuing")
                continue
            evaluate_and_save_conversations(client, input_path, output_path)
            total += 1

# calibration = ["Normal", "Roleplay", "Misinterpretation", "Miscommunication"]
# for cat in os.listdir("calibration"):
#     for conv in os.listdir(os.path.join("calibration", cat)):
#         judgement = evaluate_and_save_conversations(client, os.path.join("calibration", cat, conv), None)
#         print(judgement, cat)

names = ["normal", "roleplay confusion", "misinterpretation", "repetition loop", "miscommunication"]
categories = {x: 0 for x in names}        
for puzzle in os.listdir(result_folder):
    for run in os.listdir(os.path.join(result_folder, puzzle)):
        output_path = os.path.join(result_folder, puzzle, run, "gpt4o1_eval.txt")
        if not os.path.exists(output_path):
            continue
        total += 1
        with open(output_path, "r") as f:
            text = f.read()
            text = text.replace("**", "").replace("[", "").replace("]", "").lower()
            
#             if text not in names:
#                 continue
            
            categories[text] += 1
print(total)
key_to_label_mapping = {
    "normal": "Normal",
    "roleplay confusion": "Roleplay",
    "misinterpretation": "Interpretation",
    "repetition loop": "Repetition",
    "miscommunication": "Communication"
}

# Extract keys (categories) and values (percentages)
labels = [key_to_label_mapping[x] for x in key_to_label_mapping]
sizes = [categories[x] for x in key_to_label_mapping]
print(labels, sizes)
# Colors for the pie chart
to_plot = {"GPT4o": "#b069db", "InternVL": "#FF6D60", "GPT4V": "#98D8AA", "Gemini": "skyblue", "LLAVA": "#F7D060", "QwenVL": "#8D6F64", "LLaMA32": "#7358DC", "Random": "#CECECE"}

colors = ['#b069db', '#FF6D60', '#98D8AA', 'skyblue', '#F7D060']
wedgeprops = {'edgecolor': 'black', 'linewidth': 1.5}
font_properties = {'fontsize': 18}

# Create the pie chart
plt.figure(figsize=(6, 6))
plt.pie(
    sizes, 
    labels=labels, 
    autopct='%1.1f%%', 
    startangle=90, 
    colors=colors,
    wedgeprops=wedgeprops,
    textprops=font_properties)

# Equal aspect ratio ensures the pie chart is circular
plt.axis('equal')
plt.tight_layout()

plt.savefig("piechart.pdf", dpi=300)
