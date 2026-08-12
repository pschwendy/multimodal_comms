import os
from multimodal_comms.apps.comma.paths import CONFIG_ROOT, IMAGE_ROOT
import random
import json
import base64


def read_json_file(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}


class Agent:
    def __init__(self, role="EXPERT", config={}) -> None:
        # List of strings. Each element (string) is a message alternating between solver and expert
        self.conversation = []

        self.role = role

        key_path = os.environ.get("COMMA_KEYS_FILE", str(CONFIG_ROOT / "keys.json"))
        if not os.path.exists(key_path):
            raise RuntimeError("set COMMA_KEYS_FILE for the historical single-agent workflow")
        key_config = json.load(open(key_path))
        self.prompts = json.load(open(CONFIG_ROOT / "prompts.json"))


        if role == "ONE_AGENT":
            self.prompt = self.prompts["ONE_AGENT"]
        else:
            raise Exception("Invalid role")
        self.history = self.prompts["history"]
        self.API_KEY = key_config["AZURE_OPENAI_API_KEY"]
        self.manual_image = None
        self.manual = self.prompts["SimpleWirePuzzle"]

    def clear(self):
        self.conversation = []

    def set_module(self, module):
        self.module = module
        self.manual = self.prompts.get(
            self.module.__str__(), self.prompts["SimpleWirePuzzle"])
        manual = IMAGE_ROOT / "manuals" / (self.module.__str__() + ".jpg")
        if manual.exists():
            self.manual_image = str(manual)
        else:
            self.manual_image = None
    
    def step(self, puzzle, message):
        return puzzle.execute_action(message)

    def get_feedback(self, cur_puzzle, history_puzzle, made_mistake):
        if cur_puzzle != history_puzzle:
            return "Here comes a new puzzle. Let's start working on it."
        elif made_mistake:
            return "That action seems to have been a mistake. A red light popped up on the puzzle."
        else:
            return "I have performed the action."
    
    def respond_with_image(self, width, height, actions, message):
        image_data = self.capture_screen_area(12, 12, width - 24, height + 70)
        response = self.respond(image_data, actions, f"{message}")  
        return response 


    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_conversation_history_string(self, image_data=None, actions=None, message=None, use_history=False, model=None):
        llm_input = ""
        if not self.manual_image:
            llm_input = self.manual if self.role in ["ONE_AGENT"] else ""

        if use_history:
            llm_input = self.prompt + "\n\n" + self.history + "\n\n" + llm_input + "\n\n"
            data = read_json_file("history/past_runs.json")
            if data[self.module.__str__()] is not None:
                llm_input += data[self.module.__str__()] + "\n\n"
        else:
            if llm_input != "":
                llm_input = self.prompt + "\n\n" + llm_input + "\n\n"
            else:
                llm_input = self.prompt + "\n\n"

        if self.role == "ONE_AGENT":
            speaker = "ONE_AGENT"
        else:
            raise Exception("Invalid role")

        if message != None:
            self.conversation.append(message)

        for mess in self.conversation:
            if speaker == "SOLVER":
                llm_input += f"SOLVER: {mess}\n\n"
                speaker = "EXPERT"
            elif speaker == "EXPERT":
                llm_input += f"EXPERT: {mess}\n\n"
                speaker = "SOLVER"
            elif speaker == "Assistant":
                llm_input += f"Assistant: {mess}\n\n"
                speaker = "Assistant"

        llm_input += f"{speaker}: "


        # Solver prompt
        if image_data != None and actions != None:
            action_string = "The available actions are:\n"
            for action in actions:
                name = action['name']
                action_string += f"{name}\n"
            action_string += "\n"
            
            if model in ["gpt4o", "gpt4v"]:
                messages = []

                base64_image = self.encode_image(image_data)

                messages.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                    },
                })

                messages.append({"type": "text", "text": action_string})
                messages.append({"type": "text", "text": llm_input})
                # print(messages)
                return messages
            elif model in ["qwenVL"]:

                messages = [
                    
                    {'image': image_data},
                    {'text': action_string},
                    {'text': llm_input}
                ]
                return messages
            elif model in ["llava"]:

                prompt = f"[INST] <image>\n{action_string}\n{llm_input} [/INST]"
                return prompt
            elif model in ["internVL"]:
                prompt = f'<image>\n {action_string}\n {llm_input}'
                #print(prompt)
                return prompt
            elif model in ["internVLX"]:
                prompt = f'<ImageHere>{action_string}\n {llm_input}'
                return prompt
            elif model in ["random"]:
                return action_string
            elif model in ["human"]:
                prompt = f'{llm_input}\n{action_string}'
                return prompt
            else:
                print(model)
                raise Exception("Invalid model")

        if self.manual_image:
            return self.manual_image, llm_input
        return llm_input

    # image_data: Image data for the puzzle (only used for solver)
    # actions: List of actions which can be performed
    # message: None if first message by Solver, otherwise a string which
    # is the question/response from solver/expert
    def respond(self, image_data, actions, message):

        random_action = random.choice(actions[1:])
        action_name = random_action['name']
        self.conversation.append(action_name)
        return action_name
