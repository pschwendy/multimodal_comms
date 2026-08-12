from transformers import Qwen2VLForConditionalGeneration,AutoModelForCausalLM, AutoTokenizer, AutoProcessor
from transformers.generation import GenerationConfig
from PIL import Image
from qwen_vl_utils import process_vision_info
import torch
torch.manual_seed(1234)

from multimodal_comms.apps.comma.agents.agent import Agent

class QwenVLAgent(Agent):
    def __init__(self, role="SOLVER", config={}) -> None:
        super(QwenVLAgent, self).__init__(role, config)

        self.model_size = config.get("MODEL_SIZE", "2B")
        self.model = Qwen2VLForConditionalGeneration.from_pretrained(f"Qwen/Qwen2-VL-{self.model_size}-Instruct", torch_dtype="auto").to(self.device)
        self.processor = AutoProcessor.from_pretrained(f"Qwen/Qwen2-VL-{self.model_size}-Instruct", max_pixels=256 * 28 * 28)

        # Specify hyperparameters for generation
        self.model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

    # Given conversation history, respond to message.
    def respond(self, image_data, actions):

        ret = self.get_conversation_history_string(
            image_data=image_data, actions=actions, model="qwenVL")

        image = None
        if type(ret) == tuple:
            image_path, llm_input = ret
            llm_input = "This is a picture of the puzzle manual. " + llm_input
            image = Image.open(image_path)
            messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                            },
                            {"type": "text", "text": llm_input},
                         ],
                    }
                ]
        else:
            llm_input = ret                
            if image_data != None:
                image = Image.open(image_data)
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "image",
                            },
                            {"type": "text", "text": llm_input},
                         ],
                    }
                ]
            else:
                messages = [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": llm_input},
                         ],
                    }
                ]

        text_prompt = self.processor.apply_chat_template(messages, add_generation_prompt=True)

        inputs = self.processor(
            text=[text_prompt], images=None if not image else [image], padding=True, return_tensors="pt"
        )

        inputs = inputs.to(self.device)
        output_ids = self.model.generate(**inputs, max_new_tokens=self.config.get("max_output_len", 1024), temperature=0.01)
        generated_ids = [
            output_ids[len(input_ids) :]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = self.processor.batch_decode(
            generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        predicted_action = output_text[0].strip()
        return predicted_action
