from multimodal_comms.apps.comma.agents.agent import Agent
from PIL import Image
from google import genai
from google.genai import types
import time
# GeminiAgent class
class GeminiAgent(Agent):
    def __init__(self, role="SOLVER", config={}, conversation_format="concatenated") -> None:
        """
        Initialize the Gemini Agent.

        Args:
            role (str): Role of the agent ("EXPERT" or "SOLVER").
            type (str): Additional type information if needed.
            conversation_format (str): Format of the conversation history.
                                       Options: "concatenated", "structured".
        """
        super(GeminiAgent, self).__init__(role=role, config=config, conversation_format=conversation_format)
    
        self.model_name = config['MODEL_NAME']
        api_key = config['API_KEY']

        self.client = genai.Client(api_key=api_key)

    def respond(self, image_data, actions):
        ret = self.get_conversation_history_string(
            image_data=image_data,
            actions=actions)
        
        user_messages = []

        # Handle concatenated format
        if isinstance(ret, tuple):
            image_path, llm_input = ret
            user_messages = [
                Image.open(image_path), f"This is a picture of the puzzle manual. {llm_input}"
            ]
        else:
            llm_input = ret
            if image_data:
                user_messages = [Image.open(image_data), llm_input]
            else:
                user_messages = llm_input

    
        response = self.client.models.generate_content(
            model=self.model_name,
            contents=user_messages,
            config=types.GenerateContentConfig(
                temperature=0.0,
                top_p=1.0,
                candidate_count=1,
                seed=42,
                max_output_tokens=self.config.get("max_output_len", 1024),
                stop_sequences=["STOP!"],
                presence_penalty=0.0,
                frequency_penalty=0.0,
            )
        )
                
        time.sleep(15)

        return response.text