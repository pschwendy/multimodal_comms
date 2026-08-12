import json
import os
from openai import AzureOpenAI
import matplotlib.pyplot as plt

MODEL="o1-preview"
os.environ["AZURE_OPENAI_ENDPOINT"] ="YOUR API ENDPOINT"
os.environ["AZURE_OPENAI_API_KEY"] = "<YOUR API KEY>"
client = AzureOpenAI(
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version="2024-08-01-preview"
    # api_version="2024-02-01",
)

def evaluate_and_save_conversations(client, conversation_file, output_file):
    # Load the conversations from the JSON file, parsing each line as a separate JSON object
    with open(conversation_file, 'r') as file:
        conversations = [json.loads(line) for line in file]

    # Group conversations into a single prompt
    results = []
    # print(f"Evaluating conversation...")

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
    # print(predicted_action)
    return predicted_action

calibration = ["roleplay", "misinterpretation", "miscommunication", "repetition"]
result = {}
for cat in os.listdir("callibration"):
    if cat == "normal":
        continue
    print(f"---------------------{cat}------------------")
    print("-----------------------------------------------")
    result[cat] = {"error": 0, "total": 0}
    for conv in os.listdir(os.path.join("callibration", cat)):
        judgement1 = evaluate_and_save_conversations(client, os.path.join("callibration", cat, conv), None)
        judgement2 = evaluate_and_save_conversations(client, os.path.join("callibration", cat, conv), None)
        judgement3 = evaluate_and_save_conversations(client, os.path.join("callibration", cat, conv), None)
        # print(judgement, cat)
        j1 = cat.lower() in judgement1.lower()
        # j2 = cat.lower() in judgement2.lower()
        # j3 = cat.lower() in judgement3.lower()
        if j1 < 1: #sum([j1, j2, j3])
            print(conv, judgement1, cat) #judgement2, judgement3
            result[cat]["error"] += 1
        result[cat]["total"] += 1

err = 0
total = 0
for k, v in result.items():
    print(k, f"Total samples: {v['total']}")
    total += v['total']
    print("Error percetnage: ", v["error"]*100/v["total"])
    err += v["error"]
print("Overall error percetnage: ", err/total)

"""You are an AI judge evaluating a conversation between an AI expert and a solver disarming a module. 
Your job is to analyze this conversation and identify any issues that occurred, based on these updated criteria:

1. **Roleplay Confusion**: Did the AI expert mistakenly think it was the solver, or vice versa? Look for instances where the roles were misunderstood, leading to communication breakdown or failure to solve the puzzle. It could also be the expert including conversation as a solve or vice versa.
2. **Misinterpretation**: Did the solver misunderstand the current puzzle state or any signals provided by the expert? Identify any cases where this misunderstanding led to incorrect responses or failed instructions that led to there being an error reported by the environment.
3. **Repetition Loop**: Did the solver repeat previous incorrect actions, even after encountering the same situation before? Identify any instances where the solver repeated an incorrect state-action pair.
4. **Miscommunication**: Did the solver ignore or misunderstand the expert's instructions, or attempt to solve the puzzle independently as though it were the expert by claiming it has a solution to the puzzle? Identify any cases of miscommunication resulting from these actions.

Your response should just be one or more of the following categories verbatim, with no additional words or punctuation in a list format: [Normal, Roleplay Confusion, Misinterpretation, Repetition Loop, Miscommunication]. If a conversation displays more than 1 issue, include all issues in the list.

Here are some examples of what these mistakes look like.

Roleplay confusion example:
EXPERT: SOLVER: I have a tension headache. It started a few hours ago and is now throbbing on both sides of my head. I also have a mild fever and feel nauseous.\n\nEXPERT: Based on your symptoms, it sounds like you may be experiencing a migraine. Have you taken any medication for headaches in the past?
SOLVER: diagnose_migraine_headache

Misinterpretation example:
SOLVER: "I see a panel with four wires, each labeled with \"CUT.\" The wires are colored as follows from top to bottom: blue, black, yellow, and red. The serial number is 135790. \n\nEXPERT: I see a panel with four wires, each labeled with \"CUT.\" The wires are colored as follows from top to bottom: blue, black, yellow, and red. The serial number is 135790. Which wires should I cut?"
EXPERT: "Since there is more than one red wire and the last digit of the serial number is odd, cut the last red wire."
SOLVER: "cut_wire_1"

Repetition Loop example:
EXPERT: "Press the button labeled 2"
SOLVER: "press_2"
ENVIRONMENT: "That action seems to have been a mistake."
EXPERT: "Press the button labeled 2"
SOLVER: "press_2"
ENVIRONMENT: "That action seems to have been a mistake."

Miscommunication example:
SOLVER: "The image shows a panel with five wires running diagonally from left to right. The wires are, from top to bottom, red, black, yellow, red, and white. Each wire has a \"CUT\" label next to it. There are two vertical bars on the left and right sides of the panel, and a circle in the top right corner. The text at the bottom says \"Time Left: 100:00\" and \"Serial Number: 123456\". I believe I need to cut all the wires.\ncut_wire_1\ncut_wire_2\ncut_wire_3\ncut_wire_4\ncut_wire_5\n"
ENVIRONMENT: "The action was performed successfully."
"""