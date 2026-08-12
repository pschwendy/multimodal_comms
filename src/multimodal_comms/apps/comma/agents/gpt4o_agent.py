from openai import AzureOpenAI
from multimodal_comms.apps.comma.agents.agent import Agent
import json
import os

# GPT4oAgent class
class GPT4oAgent(Agent):
    def __init__(self, role="SOLVER", config={}, conversation_format="concatenated") -> None:
        """
        Initialize the GPT4oAgent.

        Args:
            role (str): Role of the agent ("EXPERT" or "SOLVER").
            type (str): Additional type information if needed.
            conversation_format (str): Format of the conversation history.
                                       Options: "concatenated", "structured".
        """
        super(GPT4oAgent, self).__init__(role=role, config=config, conversation_format=conversation_format)
    
        deployment_name = 'gpt-4o'
        api_key = config['API_KEY']
        api_version = config['API_VERSION']  # Update if needed
        api_base = config['API_BASE']  # Replace with your endpoint

        self.client = AzureOpenAI(
            api_key=api_key,
            api_version=api_version,
            base_url=f"{api_base}openai/deployments/{deployment_name}",
        )

    def respond(self, image_data, actions):
        ret = self.get_conversation_history_string(
            image_data=image_data,
            actions=actions,
            model="gpt4o"
        )
        

        user_messages = []

        if self.conversation_format == "structured":
            if isinstance(ret, tuple):
                image_path, llm_input = ret
                base64_image = self.encode_image(image_path)
                user_messages = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        }
                    },
                    {
                        "role": "user",
                        "type": "text",
                        "content": f"This is a picture of the puzzle manual. {llm_input}"
                    }
                ]
            else:
                llm_input = ret
                user_messages = llm_input  # Assuming JSON string

            # Prepare the messages based on the structured format
            response = self.client.chat.completions.create(
                model='gpt-4o',
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_messages}
                ],
                max_tokens=self.config.get("max_output_len", 1024),
                temperature=0
            )
        else:
            # Handle concatenated format
            if isinstance(ret, tuple):
                image_path, llm_input = ret
                base64_image = self.encode_image(image_path)
                user_messages = [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}",
                        }
                    },
                    {
                        "role": "user",
                        "type": "text",
                        "content": f"This is a picture of the puzzle manual. {llm_input}"
                    }
                ]
            else:
                llm_input = ret
                user_messages = llm_input

            messages = [
                {"role": "system", "content": "You are a helpful assistant."},
                {"role": "user", "content": user_messages}
            ]   
            
            if self.role.lower().startswith("expert"):
                messages.append({"role": "user", "content": f"Limit your response to {self.module.game_manager.EXPERT_MESSAGE_WORD_LIMIT} words."})
            try:
                response = self.client.chat.completions.create(
                    model='gpt-4o',
                    messages=messages,
                    max_tokens=self.config.get("max_output_len", 1024),
                    temperature=0
                )
            except:
                return "There was an issue getting a response ..."

        predicted_action = response.choices[0].message.content.strip()
        
        return predicted_action