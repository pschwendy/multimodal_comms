from openai import AzureOpenAI
from azure.ai.inference import ChatCompletionsClient
from azure.ai.inference.models import SystemMessage, UserMessage
from azure.core.credentials import AzureKeyCredential
from multimodal_comms.apps.comma.agents.agent import Agent
import json
import os

# GPTAgent class
class GPTAgent(Agent):
    def __init__(self, role="SOLVER", config={}, conversation_format="concatenated") -> None:
        """
        Initialize the GPT4oAgent.

        Args:
            role (str): Role of the agent ("EXPERT" or "SOLVER").
            type (str): Additional type information if needed.
            conversation_format (str): Format of the conversation history.
                                       Options: "concatenated", "structured".
        """
        super(GPTAgent, self).__init__(role=role, config=config, conversation_format=conversation_format)
    
        self.deployment_name = config.get('MODEL_NAME', 'gpt-4o')
        print(f"Initializing {self.deployment_name} {self.role} Model ...")
        api_key = config['API_KEY']
        api_base = config['API_BASE']  # Replace with your endpoint
        self.model_name = os.getenv("DEPLOYMENT_NAME", self.deployment_name)
       
        # Set the AZURE_CLIENT_ID, AZURE_CLIENT_SECRET, and AZURE_TENANT_ID environment variables
        self.client = AzureOpenAI(
            api_version="2024-12-01-preview",
            azure_endpoint=api_base,
            api_key=api_key,
        )



    def respond(self, image_data, actions):
        ret = self.get_conversation_history_string(
            image_data=image_data,
            actions=actions,
            model=self.deployment_name
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
                model=self.deployment_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": user_messages}
                ],
                max_tokens=300
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

            # messages = [
            #     #{"role": "system", "content": "You are a helpful assistant."},
            #     {"role": "user", "content": user_messages}
            # ]   
            
            # if self.role.lower().startswith("expert"):
            #     messages.append({"role": "user", "content": f"Limit your response to {self.module.game_manager.EXPERT_MESSAGE_WORD_LIMIT} words."})
            
            try:
                response = self.client.chat.completions.create(
                    messages=[
                        {
                            "role": "system",
                            "content": "You are a helpful assistant.",
                        },
                        {
                            "role": "user",
                            "content": user_messages,
                        }
                    ],
                    max_completion_tokens=100000,
                    model=self.deployment_name,
                    top_p = 1.0
                )
                
            except Exception as e:
                return "There was an issue getting a response"

        if response.choices[0].message.content:
            response_data = response.usage.to_dict()
            save_dir = self.module.game_manager.save_dir
            if not os.path.exists(save_dir):
                os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "response_data.jsonl"), "a") as f:
                f.write(json.dumps(response_data) + "\n")

            # Got a response, extract the reasoning + total token info, save it into a data structure
            predicted_action = response.choices[0].message.content.strip()
        else:
            #print(response)
            return "There was an issue getting a response..."
        
        return predicted_action