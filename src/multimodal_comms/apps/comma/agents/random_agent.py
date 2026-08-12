from multimodal_comms.apps.comma.agents.agent import Agent
import random


class RandomAgent(Agent):
    def __init__(self, role="SOLVER", config={}) -> None:
        super(RandomAgent, self).__init__(role, config)
      

    def clear(self):
        self.conversation = []

    # Given conversation history, respond to message.
    def respond(self, image_data, actions):
        
        if self.role.startswith('EXPERT'):
            return ""
        
        ret = self.get_conversation_history_string(
            image_data=image_data, actions=actions, model="random")
        
        actions = self.module.actions
        return f"{self.module.game_manager.action_delimeter}" + random.choice(list(actions.keys()))

