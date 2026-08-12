from transformers import AutoModelForCausalLM 
from transformers import AutoProcessor
import torch
from PIL import Image
from multimodal_comms.apps.comma.agents.agent import Agent



class PhiVisionAgent(Agent):
    def __init__(self, role="SOLVER", config={}) -> None:
        super(PhiVisionAgent, self).__init__(role, config)

        model_id = "microsoft/Phi-3-vision-128k-instruct" 

        self.model = AutoModelForCausalLM.from_pretrained(model_id, device_map="cuda", trust_remote_code=True, torch_dtype="auto", _attn_implementation='eager').to(self.device) # use _attn_implementation='eager' to disable flash attention

        self.processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True) 
        


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

        llm_input = "<|image_1|>\n" + llm_input

        messages = [ 
            {"role": "user", "content": llm_input}
        ] 

        prompt = self.processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)


        if image is None:
            image_features_placeholder = Image.new('RGB', (224, 224), color=(0, 0, 0))
            inputs = self.processor(prompt, [image_features_placeholder], return_tensors="pt").to(self.device) 
        else:
            inputs = self.processor(prompt, [image], return_tensors="pt").to(self.device)

        generation_args = { 
            "max_new_tokens": self.config.get("max_output_len", 1024), 
            "temperature": 0.0, 
            "do_sample": False, 
        } 

        generate_ids = self.model.generate(**inputs, eos_token_id=self.processor.tokenizer.eos_token_id, **generation_args) 

        # remove input tokens 
        generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:]
        response = self.processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0] 

        return response