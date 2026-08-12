from multimodal_comms.apps.comma.agents.agent import Agent
import requests
import time

class GPT4VAgent(Agent):
    def __init__(self, role="SOLVER", config={}) -> None:
        super(GPT4VAgent, self).__init__(role, config=config)
        self.conversation = []

        api_version = config['API_VERSION']
        api_base = config['API_BASE']
        api_key = config['API_KEY']
        
        self.headers = {
            "Content-Type": "application/json",
            "api-key": api_key,
        }
        
        self.ENDPOINT = f"{api_base}openai/deployments/gpt-4v/chat/completions?api-version={api_version}"

    def clear(self):
        self.conversation = []

    # Given conversation history, respond to message.
    def respond(self, image_data, actions):
        ret = self.get_conversation_history_string(
            image_data=image_data, actions=actions, model="gpt4v")
        
        user_messages = []

        if type(ret) == tuple:
            image_path, llm_input = ret
            base64_image = self.encode_image(image_path)
            user_messages.append({
                "type": "image_url",
                "image_url": {
                    "url": f"data:image/jpeg;base64,{base64_image}",
                }
            })
            user_messages.append({"type": "text", "text": "This is a picture of the puzzle manual. " + llm_input})
        else:
            user_messages = ret
            
        messages = [{"role": "system", "content": "You are a helpful assistant."},
                         {"role": "user", "content": user_messages}]
        
        if self.role.lower().startswith("expert"):
            messages.append({"role": "user", "content": f"Limit your response to {self.module.game_manager.EXPERT_MESSAGE_WORD_LIMIT} words."})
        
        # Payload for the request
        payload = {
            "messages": messages,
            "temperature": 0,
            "top_p": 1,
            "max_tokens": self.config.get("max_output_len", 1024)
        }

        
        
        # Send request
        response = None
        while not response:
            try:
                response = requests.post(self.ENDPOINT, headers=self.headers, json=payload)
                response.raise_for_status()  # Will raise an HTTPError if the HTTP request returned an unsuccessful status code
                response = response.json()
            except requests.RequestException as e:
                print(f"Failed to make the request. Error: {e}")
                predicted_action = "There was a problem getting a response ..."
                time.sleep(5)
                return predicted_action

                

        predicted_action = response['choices'][0]['message']['content']
        return predicted_action
