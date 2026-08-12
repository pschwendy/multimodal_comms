import os
import random
import json
import base64
from pathlib import Path
from sys import platform
from multimodal_comms.apps.comma.modules.util import get_screenshot_of_puzzle_window

APP_ROOT = Path(__file__).resolve().parents[1]

def read_json_file(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file)
    except (FileNotFoundError, json.JSONDecodeError):
        return {}

class Agent:
    def __init__(self, role="EXPERT", config={}, conversation_format="concatenated") -> None:
        """
        Initialize the Agent.

        Args:
            role (str): Role of the agent ("EXPERT" or "SOLVER").
            conversation_format (str): Format of the conversation history.
                                       Options: "concatenated", "structured".
        """
        # List of messages. Each element is a dict with roles and messages.
        self.conversation = []
        self.response_data = []
        self.config = config
        self.device = config.get("DEVICE", "cpu")
        self.role = role
        self.conversation_format = conversation_format
        self.prompts = read_json_file(APP_ROOT / "config" / "prompts.json")
        

        if role.startswith("EXPERT"):
            self.prompt = self.prompts["EXPERT"]
        elif role.startswith("SOLVER"):
            self.prompt = self.prompts["SOLVER"]
        else:
            raise Exception("Invalid role")
        
        self.manual_image = None
        self.manual = self.prompts.get("SimpleWirePuzzle", "")

    def clear(self):
        self.conversation = []
        self.response_data = []

    def capture_screen_area(self, width, height):

        screen_image = get_screenshot_of_puzzle_window("Puzzle")
        if platform in ('win32', 'darwin'):
            screen_image = screen_image.crop((12, self.module.game_manager.title_bar_height, 512, 512+100))
        else:
            screen_image = screen_image.crop((0, self.module.game_manager.title_bar_height, 500, 512+70))

        if not os.path.exists(self.module.game_manager.save_dir):
            os.makedirs(self.module.game_manager.save_dir)

        for i in range(100):
            temp_filename = f"{self.module.game_manager.save_dir}/{i}.png"
            if not os.path.exists(temp_filename):
                screen_image.save(temp_filename)
                break

        return temp_filename

    def set_module(self, module):
        self.module = module
        self.manual = self.prompts.get(
            str(self.module), self.prompts.get("SimpleWirePuzzle", ""))
        manual_path = APP_ROOT / "images" / "manuals" / f"{self.module}.jpg"
        self.manual_image = manual_path if os.path.exists(manual_path) else None

    def step(self, message):
        
        self.module.game_manager.add_to_conversations({"from": self.role, "value": message})
        self.module.game_manager.display_message(message, self.role)
        
        if self.__str__() == "HumanAgent":
            potential_actions = message.split("\\n")        
        else:    
            potential_actions = message.split("\n")        
        self.module.game_manager.cur_step += 1
        for potential_action in potential_actions:
            if self.module.finished:
                return True
            
            self.module.execute_action(potential_action)
   
            if self.module.game_manager.cur_step > self.module.game_manager.MAX_CONVERSATION_TURNS or self.module.mistakes >= self.module.game_manager.MAX_MISTAKES:
                self.module.finished = True
                if self.module.mistakes >= self.module.game_manager.MAX_MISTAKES:
                    error_message = f"Solver has made {self.module.mistakes} mistakes. Moving on to the next puzzle ..."
                else:
                    error_message = "Solver has reached the maximum number of steps. Moving on to the next puzzle."
                with open(os.path.join(self.module.game_manager.save_dir, "conversation.jsonl"), "a") as f:
                    f.write(json.dumps(
                        {"from": "ENVIRONMENT", "value": error_message}) + "\n")
                
                return True
            

    def encode_image(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    def get_conversation_history_string(self, image_data=None, actions=None, model=None):
        
        if self.conversation_format not in ["concatenated", "structured"]:
            raise ValueError("Invalid conversation_format. Choose 'concatenated' or 'structured'.")

        if self.conversation_format == "concatenated":
            llm_input = self.manual if self.role.startswith("EXPERT") else ""
            
            if self.module.solver_private_info and self.role.startswith('SOLVER'):
                llm_input += f"{self.module.solver_private_info} Do not share this information with anyone.\n"
            
            llm_input = f"{self.prompt}\n\n" if not llm_input else f"{self.prompt}\n\n{llm_input}\n\n"
            
            for mess in self.conversation:
                llm_input += f"{mess['from']}: {mess['value']}\n\n"

            llm_input += f"{self.role}: "

            if image_data and actions:
                action_string = "The available actions are:\n" + "\n".join(actions.keys()) + "\n\n"
                action_string += f"When you want to perform an action, follow this format:\n{self.module.game_manager.action_delimeter}action_name\n\n"
                
                if model in ["gpt4o", "gpt4v", "gpt-4o", "o1-mini", "o4-mini", "o3-mini", "o3"]:
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

                    return messages
                
                # Handle other models as before
                # ...
                elif model in ["qwenVL"]:
                    prompt = f'{action_string}\n{llm_input}'
                    return prompt
                elif model in ["llava"]:
                    prompt = f"{action_string}\n{llm_input}"
                    return prompt
                elif model in ["internVL"]:
                    prompt = f'<image>\n{action_string}\n{llm_input}'
                    return prompt
                elif model in ["random"]:
                    return action_string
                elif model in ["human"]:
                    prompt = f'{action_string}\n{llm_input}'
                    return prompt
                else:
                    prompt = f'{action_string}\n{llm_input}'
                    return prompt
            
            
            if self.manual_image:
                return self.manual_image, llm_input
            
            return llm_input

        elif self.conversation_format == "structured":
            structured_history = {"human": "", "Assistant": ""}
            if self.role == "EXPERT":
                structured_history["human"] = "SOLVER"
            elif self.role == "SOLVER":
                structured_history["human"] = "EXPERT"
            else:
                raise Exception("Invalid role")

            # Alternate between human and assistant
            role = "Assistant" if self.role == "EXPERT" else "human"
            for mess in self.conversation:
                if role == "human":
                    structured_history.setdefault("human", "")
                    structured_history["human"] += mess + "\n\n"
                    role = "Assistant"
                elif role == "Assistant":
                    structured_history.setdefault("Assistant", "")
                    structured_history["Assistant"] += mess + "\n\n"
                    role = "human"

            # Prepare the final prompt
            llm_input = f"{self.prompt}\n\n{self.history}\n\n" if use_history else self.prompt + "\n\n"

            if image_data and actions:
                action_string = "The available actions are:\n" + "\n".join([action['name'] for action in actions]) + "\n\n"
                if self.manual_image:
                    manual_info = f"This is a picture of the puzzle manual.\n{self.manual}"
                else:
                    manual_info = self.manual if self.role == "EXPERT" else ""

                # Combine all parts
                prompt = f"{llm_input}{action_string}{manual_info}\n\n"
                prompt += json.dumps(structured_history)

                return prompt

            if self.manual_image:
                return self.manual_image, json.dumps(structured_history)
            return json.dumps(structured_history)

    # image_data: Path to image data for the puzzle/manual
    # actions: List of actions which can be performed (only for solver)
    def respond(self, image_data = None, actions = None):
        pass
