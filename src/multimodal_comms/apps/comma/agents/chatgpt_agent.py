from openai import AzureOpenAI
from multimodal_comms.apps.comma.agents.agent import Agent
import os
import json

class ChatGPTAgent(Agent):
    def __init__(self, role="EXPERT", config = {}) -> None:
        super(ChatGPTAgent, self).__init__(role)

    
    def respond(self, image_data, actions, message):
        llm_input = self.get_conversation_history_string(image_data=None, actions=None, message=message)

        client = AzureOpenAI(
            api_key = self.API_KEY,  
            api_version="2024-02-01",
            azure_endpoint = "https://chatgpt-simulation.openai.azure.com/"
        )

        deployment_name='gpt-35-turbo-instruct' #This will correspond to the custom name you chose for your deployment when you deployed a model. Use a gpt-35-turbo-instruct deployment. 
            
        response = client.completions.create(model=deployment_name, prompt=llm_input, max_tokens=self.config.get(["max_output_len"], 1024), temperature=0)
        response_text = response.choices[0].text
        self.conversation.append(response_text)
        return response_text
