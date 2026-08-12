from multimodal_comms.apps.comma.agents.agent import Agent
import torch
import math
import torchvision.transforms as T
from PIL import Image
from torchvision.transforms.functional import InterpolationMode
from transformers import AutoModel, AutoTokenizer


class InternVLAgent(Agent):
    def __init__(self, role="SOLVER", config={}) -> None:
        super(InternVLAgent, self).__init__(role=role)

        def split_model(model_name):
            device_map = {}
            world_size = torch.cuda.device_count()
            if world_size == 0:
                print("No GPUs on this machine...")
                return None
                
            num_layers = {
                'InternVL2-1B': 24, 'InternVL2-2B': 24, 'InternVL2-4B': 32, 'InternVL2-8B': 32,
                'InternVL2-26B': 48, 'InternVL2-40B': 60, 'InternVL2-Llama3-76B': 80}[model_name]
            
            # Since the first GPU will be used for ViT, treat it as half a GPU.
            num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
            num_layers_per_gpu = [num_layers_per_gpu] * world_size
            num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
            layer_cnt = 0
            for i, num_layer in enumerate(num_layers_per_gpu):
                for j in range(num_layer):
                    device_map[f'language_model.model.layers.{layer_cnt}'] = i
                    layer_cnt += 1
            device_map['vision_model'] = 0
            device_map['mlp1'] = 0
            device_map['language_model.model.tok_embeddings'] = 0
            device_map['language_model.model.embed_tokens'] = 0
            device_map['language_model.output'] = 0
            device_map['language_model.model.norm'] = 0
            device_map['language_model.lm_head'] = 0
            device_map[f'language_model.model.layers.{num_layers - 1}'] = 0

            return device_map

        self.model_size = config.get("MODEL_SIZE", "1B")
        path = f"OpenGVLab/InternVL2-{self.model_size}"
        device_map = split_model(f'InternVL2-{self.model_size}')
        self.model = AutoModel.from_pretrained(
            path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
            device_map=device_map).eval()

        
        self.tokenizer = AutoTokenizer.from_pretrained(path, trust_remote_code=True, use_fast=False)
        # set the max number of tiles in `max_num`
        
        self.generation_config = dict(
            num_beams=1,
            max_new_tokens=1024,
            do_sample=False,
        )


    def clear(self):
        self.conversation = []

    def build_transform(self, input_size):
        IMAGENET_MEAN = (0.485, 0.456, 0.406)
        IMAGENET_STD = (0.229, 0.224, 0.225)

        MEAN, STD = IMAGENET_MEAN, IMAGENET_STD
        transform = T.Compose([
            T.Lambda(lambda img: img.convert('RGB') if img.mode != 'RGB' else img),
            T.Resize((input_size, input_size), interpolation=InterpolationMode.BICUBIC),
            T.ToTensor(),
            T.Normalize(mean=MEAN, std=STD)
        ])
        return transform

    def find_closest_aspect_ratio(self, aspect_ratio, target_ratios, width, height, image_size):
        best_ratio_diff = float('inf')
        best_ratio = (1, 1)
        area = width * height
        for ratio in target_ratios:
            target_aspect_ratio = ratio[0] / ratio[1]
            ratio_diff = abs(aspect_ratio - target_aspect_ratio)
            if ratio_diff < best_ratio_diff:
                best_ratio_diff = ratio_diff
                best_ratio = ratio
            elif ratio_diff == best_ratio_diff:
                if area > 0.5 * image_size * image_size * ratio[0] * ratio[1]:
                    best_ratio = ratio
        return best_ratio

    def dynamic_preprocess(self, image, min_num=1, max_num=6, image_size=448, use_thumbnail=False):
        orig_width, orig_height = image.size
        aspect_ratio = orig_width / orig_height

        # calculate the existing image aspect ratio
        target_ratios = set(
            (i, j) for n in range(min_num, max_num + 1) for i in range(1, n + 1) for j in range(1, n + 1) if
            i * j <= max_num and i * j >= min_num)
        target_ratios = sorted(target_ratios, key=lambda x: x[0] * x[1])

        # find the closest aspect ratio to the target
        target_aspect_ratio = self.find_closest_aspect_ratio(
            aspect_ratio, target_ratios, orig_width, orig_height, image_size)

        # calculate the target width and height
        target_width = image_size * target_aspect_ratio[0]
        target_height = image_size * target_aspect_ratio[1]
        blocks = target_aspect_ratio[0] * target_aspect_ratio[1]

        # resize the image
        resized_img = image.resize((target_width, target_height))
        processed_images = []
        for i in range(blocks):
            box = (
                (i % (target_width // image_size)) * image_size,
                (i // (target_width // image_size)) * image_size,
                ((i % (target_width // image_size)) + 1) * image_size,
                ((i // (target_width // image_size)) + 1) * image_size
            )
            # split the image
            split_img = resized_img.crop(box)
            processed_images.append(split_img)
        assert len(processed_images) == blocks
        if use_thumbnail and len(processed_images) != 1:
            thumbnail_img = image.resize((image_size, image_size))
            processed_images.append(thumbnail_img)
        return processed_images

    def load_image(self, image_file, input_size=448, max_num=6):
        image = Image.open(image_file).convert('RGB')
        transform = self.build_transform(input_size=input_size)
        images = self.dynamic_preprocess(image, image_size=input_size, use_thumbnail=True, max_num=max_num)
        pixel_values = [transform(image) for image in images]
        pixel_values = torch.stack(pixel_values)
        return pixel_values

    # Given conversation history, respond to message.
    def respond(self, image_data, actions):
        ret = self.get_conversation_history_string(
            image_data=image_data, actions=actions, model="internVL")

        if type(ret) == tuple:
            image_path, llm_input = ret
            llm_input = "This is a picture of the puzzle manual. " + llm_input
            pixel_values = self.load_image(image_path, max_num=6).to(torch.bfloat16).to(self.device)
        else:
            llm_input = ret

        if image_data == None:
            pixel_values = None
        else:
            pixel_values = self.load_image(image_data, max_num=6).to(torch.bfloat16).to(self.device)
        
        if len(self.tokenizer(llm_input).input_ids) > 8192:
            self.conversation.append("Conversation longer than context length ...")
            return "Conversation longer than context length ..."
        response = self.model.chat(self.tokenizer, pixel_values, llm_input, self.generation_config)
        
        predicted_action = response
        return predicted_action