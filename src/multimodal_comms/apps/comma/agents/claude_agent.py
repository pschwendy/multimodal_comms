import anthropic
from multimodal_comms.apps.comma.agents.agent import Agent
import time

# ClaudeAgent class
class ClaudeAgent(Agent):
    def __init__(self, role="SOLVER", config={}, conversation_format="concatenated") -> None:
        """
        Initialize the ClaudeAgent.

        Args:
            role (str): Role of the agent ("EXPERT" or "SOLVER").
            type (str): Additional type information if needed.
            conversation_format (str): Format of the conversation history.
                                       Options: "concatenated", "structured".
        """
        super(ClaudeAgent, self).__init__(role=role, config=config, conversation_format=conversation_format)
    
        api_key = config['API_KEY']
        self.model_name = config.get("MODEL_NAME", "claude-3-5-sonnet-20241022")
        self.client = anthropic.Anthropic(
            api_key=api_key
        )

    def respond(self, image_data, actions):
        ret = self.get_conversation_history_string(
            image_data=image_data,
            actions=actions
        )
        

        user_messages = []

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
            
        response = self.client.messages.create(
            model=self.model_name,
            max_tokens=self.config.get("max_output_len", 1024),
            messages=messages
        )
        
        time.sleep(15)
        
        return response.content