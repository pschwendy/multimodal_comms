import os
import json
from openai import OpenAI
from multimodal_comms.apps.comma.agents.agent import Agent


class DeepSeekAgent(Agent):
    """Text-only LLM agent backed by the DeepSeek API.

    Modeled on GPTAgent but uses a plain openai.OpenAI client pointed at the
    DeepSeek endpoint. Because "deepseek" is not in agent.py's vision-model
    branch list, get_conversation_history_string falls through to the generic
    text-only prompt path automatically. The rendered screenshot (if any) is
    discarded on purpose: this agent structurally cannot see the board, so it
    is only meaningful on puzzles that hand the Solver textual private state
    (AtmPuzzle, TelehealthPuzzle).
    """

    def __init__(self, role="SOLVER", config={}, conversation_format="concatenated") -> None:
        super(DeepSeekAgent, self).__init__(
            role=role, config=config, conversation_format=conversation_format)

        api_key = config.get("API_KEY") or os.getenv("DEEPSEEK_API_KEY")
        base_url = config.get("API_BASE") or os.getenv("DEEPSEEK_BASE_URL") or "https://api.deepseek.com/v1"
        # Task pass pins deepseek-v4-flash. Allow config / dedicated env override
        # but do NOT read the shared DEEPSEEK_MODEL (=deepseek-chat) so the two
        # never silently diverge.
        self.model_name = (
            config.get("MODEL_NAME")
            or os.getenv("COMMA_DEEPSEEK_MODEL")
            or "deepseek-v4-flash"
        )
        self.max_tokens = int(config.get("MAX_TOKENS", os.getenv("COMMA_DEEPSEEK_MAX_TOKENS", "4000")))

        print(f"Initializing DeepSeek {self.model_name} {self.role} Model ...")
        self.client = OpenAI(api_key=api_key, base_url=base_url)

    def respond(self, image_data, actions):
        ret = self.get_conversation_history_string(
            image_data=image_data, actions=actions, model="deepseek")

        # Text-only: if a (manual_image, text) tuple came back, keep the text.
        if isinstance(ret, tuple):
            _, llm_input = ret
        else:
            llm_input = ret

        try:
            response = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": "You are a helpful assistant."},
                    {"role": "user", "content": llm_input},
                ],
                max_tokens=self.max_tokens,
                top_p=1.0,
            )
        except Exception as e:
            print(f"[DeepSeekAgent] API error: {e}")
            return "There was an issue getting a response"

        save_dir = getattr(self.module.game_manager, "save_dir", None)
        if response.usage is not None and save_dir:
            os.makedirs(save_dir, exist_ok=True)
            with open(os.path.join(save_dir, "response_data.jsonl"), "a") as f:
                f.write(json.dumps(response.usage.to_dict()) + "\n")

        content = response.choices[0].message.content
        if content:
            return content.strip()
        return "There was an issue getting a response..."
