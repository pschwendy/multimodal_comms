from multimodal_comms.apps.comma.agents.agent import Agent
import sys


class HumanAgent(Agent):
    def __init__(self, role="SOLVER", config={}) -> None:
        super(HumanAgent, self).__init__(role=role, config=config)
        # List of strings. Each element (string) is a message alternating between solver and expert
        self.conversation = []

    def clear(self):
        self.conversation = []

    # Given conversation history, respond to message.
    def respond(self, image_data, actions):
        llm_input = self.get_conversation_history_string(
            image_data=image_data, actions=actions, model="human")
  
        sys.stdout.write(f"{llm_input}\nContinue the conversation: ")
        sys.stdout.flush()
        
        response = input()  # Use plain input for user response
 
        predicted_action = response

        return predicted_action
    
    def __str__(self):
        return "HumanAgent"
