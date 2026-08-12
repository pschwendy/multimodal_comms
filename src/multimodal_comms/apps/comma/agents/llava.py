from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import torch
from PIL import Image
from multimodal_comms.apps.comma.agents.agent import Agent



class LLAVAAgent(Agent):
    def __init__(self, role="SOLVER", config={}) -> None:
        super(LLAVAAgent, self).__init__(role, config)

        self.model_size = config.get("MODEL_SIZE", "7B")
        self.processor = LlavaNextProcessor.from_pretrained(f"llava-hf/llava-v1.6-mistral-7b-hf")

        self.model = LlavaNextForConditionalGeneration.from_pretrained(f"llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16) 
        self.model.to(self.device)
        
        # reference: https://huggingface.co/llava-hf/llava-v1.6-mistral-7b-hf


    # Given conversation history, respond to message.
    def respond(self, image_data, actions):
        ret = self.get_conversation_history_string(
            image_data=image_data, actions=actions, model="llava")

        image = None
        if type(ret) == tuple:
            image_path, llm_input = ret
            llm_input = "This is a picture of the puzzle manual. " + llm_input
            image = Image.open(image_path).convert('RGB')
        else:
            llm_input = ret
            if image_data != None:
                image = Image.open(image_data).convert('RGB')

        llm_input = "[INST] <image>\n" + llm_input + " [/INST]" 
        
        if image is None:
            image_features_placeholder = Image.new('RGB', (224, 224), color=(0, 0, 0))
            inputs = self.processor(images=[image_features_placeholder], text=llm_input, return_tensors="pt").to(self.device)
        else:
            inputs = self.processor(images=[image], text=llm_input, return_tensors="pt").to(self.device)


        # Encode the input prompt to determine its token length
        input_ids = inputs['input_ids']
        prompt_length = input_ids.shape[1]  # Length of the input prompt tokens

        # Autoregressively complete the prompt
        output = self.model.generate(**inputs, max_new_tokens=self.config.get("max_output_len", 1024), temperature=0, num_beams=1)

        # Slice the generated output to exclude the input prompt
        generated_ids = output[0, prompt_length:]  # Exclude the prompt tokens
        predicted_action = self.processor.decode(generated_ids, skip_special_tokens=True)

        return predicted_action