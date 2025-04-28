from pathlib import Path
# from datetime import datetime
import torch
from transformers import AutoModel, AutoModelForCausalLM, TextStreamer, AutoTokenizer, AutoModelForSeq2SeqLM, set_seed, pipeline, T5Tokenizer, T5ForConditionalGeneration, BloomTokenizerFast, BloomForCausalLM, BertTokenizer, BertForMaskedLM, DebertaV2Config, DebertaV2Tokenizer, AlbertTokenizer, AlbertModel
# from transformers.models.deberta.modeling_deberta import ContextPooler
from ..components.tree import PRIMERE_ROOT
import os
import json
import random
import re
from ..components import utility
import folder_paths

class PromptEnhancerLLM:
    def __init__(self, model_path: str = "flan-t5-small"):
        PRIMERE_CUSTOMPATH = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'LLM')
        ROOT_PATH = PRIMERE_CUSTOMPATH
        COMFY_LLM_PATH = os.path.join(folder_paths.models_dir, 'LLM')
        model_access = os.path.join(PRIMERE_CUSTOMPATH, model_path)
        if os.path.isdir(model_access) == False:
            model_access = os.path.join(COMFY_LLM_PATH, model_path)
            ROOT_PATH = COMFY_LLM_PATH
        self.model_path = model_path
        self.model_fullpath = model_access
        self.device = torch.cuda.current_device()  # "cuda" if torch.cuda.is_available() else "cpu"

        if '-promptenhancing' in self.model_path.lower() and '-instruct' in self.model_path.lower():
            baseRepo = self.model_path[:self.model_path.lower().index("-promptenhancing")]
            loraRepo = self.model_path
            model_access = os.path.join(ROOT_PATH, baseRepo)
            lora_access = os.path.join(ROOT_PATH, loraRepo)

            self.tokenizer = AutoTokenizer.from_pretrained(model_access, clean_up_tokenization_spaces=False)
            self.model = AutoModelForCausalLM.from_pretrained(model_access, torch_dtype=torch.bfloat16).to(self.device)
            self.model.load_adapter(lora_access)
        else:
            if "t5" in model_path.lower():
                self.tokenizer = T5Tokenizer.from_pretrained(model_access, clean_up_tokenization_spaces=False, ignore_mismatched_sizes=True)
                try:
                    self.model = T5ForConditionalGeneration.from_pretrained(model_access, ignore_mismatched_sizes=True, device_map="auto")
                except Exception:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(model_access, ignore_mismatched_sizes=True)
            elif "bloom-" in model_path.lower():
                self.tokenizer = BloomTokenizerFast.from_pretrained(model_access, clean_up_tokenization_spaces=False, ignore_mismatched_sizes=True)
                self.model = BloomForCausalLM.from_pretrained(model_access, ignore_mismatched_sizes=True, device_map="auto")
            elif "bert" in model_path.lower() and "deberta" not in model_path.lower() and "albert" not in model_path.lower():
                self.tokenizer = BertTokenizer.from_pretrained(model_access, clean_up_tokenization_spaces=False, ignore_mismatched_sizes=True)
                self.model = BertForMaskedLM.from_pretrained(model_access, ignore_mismatched_sizes=True, return_dict=True, is_decoder=False)
            elif "deberta-" in model_path.lower():
                self.tokenizer = DebertaV2Tokenizer.from_pretrained(model_access, clean_up_tokenization_spaces=False)
                self.config = DebertaV2Config.from_pretrained(model_access)
                self.model = AutoModel.from_pretrained(model_access, ignore_mismatched_sizes=True)
            elif "granite-" in model_path.lower():
                device = "auto"
                self.tokenizer = AutoTokenizer.from_pretrained(model_access, clean_up_tokenization_spaces=False)
                self.model = AutoModelForCausalLM.from_pretrained(model_access, device_map=device, ignore_mismatched_sizes=True)
                self.model.eval()
            elif "salamandra-" in model_path.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(model_access, clean_up_tokenization_spaces=False)
                self.model = AutoModelForCausalLM.from_pretrained(model_access, device_map="auto", torch_dtype=torch.bfloat16, ignore_mismatched_sizes=True)
            elif "albert-" in model_path.lower():
                self.tokenizer = AlbertTokenizer.from_pretrained(model_access, clean_up_tokenization_spaces=False)
                self.model = AlbertModel.from_pretrained(model_access, ignore_mismatched_sizes=True)
            elif "LeX-Enhancer-" in model_path.lower():
                self.tokenizer = AutoTokenizer.from_pretrained(model_access)
                self.model = AutoModelForCausalLM.from_pretrained(model_access, device_map="auto", torch_dtype=torch.bfloat16)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_access, clean_up_tokenization_spaces=False)
                try:
                    self.model = AutoModelForCausalLM.from_pretrained(model_access, ignore_mismatched_sizes=True)
                except Exception:
                    self.model = AutoModelForSeq2SeqLM.from_pretrained(model_access, ignore_mismatched_sizes=True)

            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    def enhance_prompt(self, input_text: str, seed: int = 1, precision: bool = True, configurator: str = "default_settings", multiply_max_length: float = 1):
        default_settings = {
            "do_sample": True,
            "temperature": 0.9,
            "top_k": 8,
            "max_length": 80,
            "num_return_sequences": 1,
            "repetition_penalty": 1.2,
            "penalty_alpha": 0.6,
            "no_repeat_ngram_size": 1,
            "early_stopping": False,
            "top_p": 0.4,
            "num_beams": 6,
        }

        variant_params = configVariants(configurator)
        configurator_name = 'high quality'
        if 'ConfigName' in variant_params:
            configurator_name = variant_params['ConfigName']
            del variant_params['ConfigName']
        # instruction = f"You are my text to image prompt enhancer, convert input user text to better {configurator_name} stable diffusion text-to-image prompt. Ignore additional text and questions, return only the enhanced prompt as raw text: "
        instruction = f"Create {configurator_name} 1 prompt for modern text-to-image text2image stable diffusion models: "
        settings = {**default_settings, **variant_params}

        if multiply_max_length != 1:
            if "max_length" in settings:
                original_maxl = int(settings['max_length'])
                settings['max_length'] = int(round(original_maxl * multiply_max_length))

        if seed is not None and int(seed) > 1:
            random.seed(seed)
            newseed = random.randint(1, (2**32) - 1)
            set_seed(newseed)
            torch.manual_seed(newseed)
        else:
            set_seed(1)
            torch.manual_seed(1)

        forceFP16 = ['t5-efficient-base-dm256', 'Llama-3.2-3B', 'Llama-3.2-3B-Instruct']
        forceFP32 = ['t5-efficient-base-dm512']

        if (precision == False or self.model_path in forceFP16) and self.model_path not in forceFP32:
            self.model.half()

        with torch.no_grad():
            if '-promptenhancing' in self.model_path.lower() and '-instruct' in self.model_path.lower():
                # self.model.to(self.device)
                messages = [{"role": "system", "content": instruction}, {"role": "user", "content": input_text}]

                inputs = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, return_tensors='pt')
                encoding = self.tokenizer(inputs, return_tensors="pt").to(self.device)

                generation_config = self.model.generation_config
                generation_config.pad_token_id = self.tokenizer.eos_token_id
                generation_config.eos_token_id = self.tokenizer.eos_token_id
                generation_config.repetition_penalty = settings['repetition_penalty']
                generation_config.do_sample = settings['do_sample']
                generation_config.max_new_tokens = 96
                generation_config.temperature = settings['temperature']
                generation_config.top_p = settings['top_p']
                generation_config.num_return_sequences = settings['num_return_sequences']

                with torch.inference_mode():
                    outputs = self.model.generate(
                        input_ids=encoding.input_ids,
                        attention_mask=encoding.attention_mask,
                        generation_config=generation_config
                    )

                enhanced_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

            else:
                if "deberta-" in self.model_path.lower() and "-instruct" not in self.model_path.lower():
                    enhanced_text = 'This moodel type not supported....'
                elif "albert-" in self.model_path.lower() and "-instruct" not in self.model_path.lower():
                    enhanced_text = 'This moodel type not supported....'

                elif "granite-" in self.model_path.lower():
                    messages = [{"role": "system", "content": instruction}, {"role": "user", "content": input_text}]
                    chat_sample = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = self.tokenizer(chat_sample, return_tensors="pt").to(self.device)

                    if 'max_length' in settings:
                        del settings['max_length']
                    if 'max_new_tokens' in settings:
                        del settings['max_new_tokens']

                    output = self.model.generate(
                        **inputs,
                        max_new_tokens=100,
                        **settings
                    )
                    enhanced_text = self.tokenizer.decode(output[0], skip_special_tokens=True)
                    full_result = enhanced_text

                    result = re.findall('\"(.*)\"', full_result)
                    if result is not None and (len(result)) > 0:
                        random.seed(seed)
                        full_result_random = random.choice(result)
                        if len(full_result_random.split()) <= 6:
                            special_result = re.findall('\n\d\.(.*)\n', full_result)
                            if special_result is not None and (len(special_result)) > 0:
                                full_result_random = random.choice(special_result)
                                if len(full_result_random.split()) > 6:
                                    full_result = full_result_random
                        else:
                            full_result = full_result_random

                    desc_result = re.findall('Description(.*)', full_result)
                    if desc_result is not None and (len(desc_result)) > 0:
                        full_result = desc_result[0].replace('*', '').replace(':', '')

                    enhanced_text = full_result

                elif "salamandra-" in self.model_path.lower():
                    messages = [{"role": "system", "content": instruction}, {"role": "user", "content": input_text}]
                    # date_string = datetime.today().strftime('%Y-%m-%d')
                    prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

                    if 'max_length' in settings:
                        del settings['max_length']
                    if 'max_new_tokens' in settings:
                        del settings['max_new_tokens']

                    inputs = self.tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")
                    outputs = self.model.generate(
                        input_ids=inputs.to(self.device),
                        max_new_tokens=200,
                        **settings
                    )

                    enhanced_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                elif "zamba" in self.model_path.lower():
                    messages = [{"role": "system", "content": instruction}, {"role": "user", "content": input_text}]
                    chat_sample = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = self.tokenizer(chat_sample, return_tensors='pt', add_special_tokens=False).to(self.device)

                    outputs = self.model.generate(
                        **inputs,
                        return_dict_in_generate=False,
                        output_scores=False,
                        use_cache=True,
                        num_beams=1,
                        **settings
                    )

                    enhanced_text = self.tokenizer.decode(outputs[0])

                elif "gpt-neo-" in self.model_path.lower() or 'gpt2' in self.model_path.lower() and "-instruct" not in self.model_path.lower():
                    generator = pipeline('text-generation', model=self.model_fullpath)

                    outputs = generator(
                        instruction + input_text,
                        **settings,
                    )
                    enhanced_text = outputs[0]['generated_text']

                elif "llama-" in self.model_path.lower() and "-instruct" not in self.model_path.lower():
                    generator = pipeline('text-generation', model=self.model_fullpath, torch_dtype=torch.bfloat16, device_map="auto")

                    outputs = generator(
                        instruction + input_text,
                        max_new_tokens=128
                    )
                    enhanced_text = outputs[0]['generated_text']

                elif "-instruct" in self.model_path.lower():
                    messages = [{"role": "system", "content": instruction}, {"role": "user", "content": input_text}]
                    generator = pipeline('text-generation', model=self.model_fullpath, torch_dtype=torch.bfloat16, device_map="auto")

                    generation_config = self.model.generation_config
                    generation_config.pad_token_id = self.tokenizer.eos_token_id
                    generation_config.eos_token_id = self.tokenizer.eos_token_id
                    generation_config.repetition_penalty = settings['repetition_penalty']
                    generation_config.do_sample = settings['do_sample']
                    generation_config.max_new_tokens = 256
                    generation_config.temperature = settings['temperature']
                    generation_config.top_p = settings['top_p']
                    generation_config.num_return_sequences = settings['num_return_sequences']

                    outputs = generator(
                        messages,
                        generation_config=generation_config
                    )

                    full_result = outputs[0]['generated_text'][-1]['content']
                    result = re.findall('\"(.*)\"', full_result)
                    if result is not None and (len(result)) > 0:
                        random.seed(seed)
                        full_result_random = random.choice(result)
                        if len(full_result_random.split()) <= 6:
                            special_result = re.findall('\n\d\.(.*)\n', full_result)
                            if special_result is not None and (len(special_result)) > 0:
                                full_result_random = random.choice(special_result)
                                if len(full_result_random.split()) > 6:
                                    full_result = full_result_random
                        else:
                            full_result = full_result_random

                    enhanced_text = full_result

                elif "smollm2-" in self.model_path.lower():
                    self.model.to(self.device)
                    messages = [{"role": "user", "content": instruction + input_text}]
                    input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.device)

                    if 'repetition_penalty' in settings:
                        del settings['repetition_penalty']
                    if 'max_new_tokens' in settings:
                        del settings['max_new_tokens']

                    outputs = self.model.generate(
                        inputs,
                        max_new_tokens=256,
                        repetition_penalty=1.2,
                        **settings,
                    )

                    enhanced_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

                elif "lex-enhancer" in self.model_path.lower():
                    '''SYSTEM_TEMPLATE = (
                        "A conversation between User and Assistant. The user asks a question, and the Assistant solves it. "
                        # "The assistant first thinks about the reasoning process in the mind and then provides the user with the answer. "
                        # "The reasoning process and answer are enclosed within <think> </think> and <answer> </answer> tags, respectively, i.e., "
                        # "<think> reasoning process here </think> <answer> answer here </answer>."
                    )'''

                    '''simple_caption = input_text #"A thank you card with the words very much, with the text on it: \"VERY\" in black, \"MUCH\" in yellow."'''
                    '''def create_chat_template(user_prompt):
                        return [
                            {"role": "system", "content": SYSTEM_TEMPLATE},
                            {"role": "user", "content": user_prompt},
                            {"role": "assistant", "content": "<think>"}
                        ]'''

                    '''def create_direct_template(user_prompt):
                        return user_prompt  # + "<think>"'''

                    '''def create_user_prompt(simple_caption):
                        return (
                            # "Below is the simple caption of an image with text. Please deduce the detailed description of the image based on this simple caption. "
                            # "Note: 1. The description should only include visual elements and should not contain any extended meanings. "
                            # "2. The visual elements should be as rich as possible, such as the main objects in the image, their respective attributes, "
                            # "the spatial relationships between the objects, lighting and shadows, color style, any text in the image and its style, etc. "
                            # "3. The output description should be a single paragraph and should not be structured. "
                            # "4. The description should avoid certain situations, such as pure white or black backgrounds, blurry text, excessive rendering of text, "
                            # "or harsh visual styles. "
                            # "The detailed caption should be human readable and fluent. "
                            # "Avoid using vague expressions such as \"may be\" or \"might be\"; the generated caption must be in a definitive, narrative tone. "
                            # "Do not use negative sentence structures, such as \"there is nothing in the image,\" etc. The entire caption should directly describe the content of the image. "
                            # "The entire output should be limited to 200 words."
                            # f"SIMPLE CAPTION: {simple_caption}"
                            f"Enhace this prompt for Flux based text to image workflow: {simple_caption}"
                        )'''

                    # messages = create_direct_template(create_user_prompt(simple_caption))
                    # input_ids = self.tokenizer.encode(instruction + input_text, return_tensors="pt").to(self.model.device)
                    # streamer = TextStreamer(self.tokenizer, skip_special_tokens=True, clean_up_tokenization_spaces=True)

                    messages = [{"role": "user", "content": instruction + input_text}]
                    input_text = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                    inputs = self.tokenizer.encode(input_text, return_tensors="pt").to(self.model.device)

                    output = self.model.generate(
                        inputs,
                        **settings
                    )
                    enhanced_text = self.tokenizer.decode(output[0], skip_special_tokens=True)  # "*" * 80

                else:
                    self.model.to(self.device)
                    inputs = self.tokenizer(instruction + input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
                    attention_mask = None
                    if "attention_mask" in inputs:
                        attention_mask = inputs["attention_mask"]

                    if "gemma-" in self.model_path.lower() or 'flux-prompt' in self.model_path.lower() or 'smollm-' in self.model_path.lower():
                        if 'max_length' in settings:
                            del settings['max_length']
                        if 'max_new_tokens' in settings:
                            del settings['max_new_tokens']
                        settings['max_length'] = 256
                        settings['max_new_tokens'] = 256

                    outputs = self.model.generate(
                        inputs["input_ids"],
                        attention_mask=attention_mask,
                        **settings,
                        pad_token_id=self.tokenizer.eos_token_id
                    )
                    enhanced_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if type(enhanced_text).__name__ == 'str':
            enhanced_text = enhanced_text.replace("Hello","").replace(instruction, ' ').replace(input_text, ' ').replace('system', ' ').replace('user', ' ').replace('assistant', ' ')
            end_of_instruction = instruction.split()[-2:]
            joined_lastwords = ' '.join(end_of_instruction).strip('\\.:,/ ')
            if joined_lastwords in enhanced_text:
                enhanced_text = enhanced_text[enhanced_text.index(joined_lastwords) + len(joined_lastwords):].strip('\\.:,/ ')
            enhanced_text = re.sub("<[b][^>]*>(.+?)</[b]>", '', enhanced_text)
            enhanced_text = re.sub(r"http\S+", "", enhanced_text)
            enhanced_text = enhanced_text.replace('<pad>', '').replace('text to image', '').replace('texttoimage', '').replace('prompt', '').replace(r'\\', '').replace('!', '.').replace("You are a helpful AI", ' ')
            enhanced_text = re.sub(r'[^a-zA-Z0-9 ."?!()]', '', enhanced_text)
            return enhanced_text.replace('named SmolLM trained by Hugging Face', '').strip('\\.:,/ ')
        else:
            return False

def PrimereLLMEnhance(modelKey = 'flan-t5-small', promptInput = 'cute cat', seed = 1, precision = True, configurator = "default", multiply_max_length = 1):
    PRIMERE_CUSTOMPATH = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'LLM')
    COMFY_LLM_PATH = os.path.join(folder_paths.models_dir, 'LLM')
    model_access = os.path.join(PRIMERE_CUSTOMPATH, modelKey)
    if os.path.isdir(model_access) == False:
        model_access = os.path.join(COMFY_LLM_PATH, modelKey)

    if os.path.isdir(model_access) == True:
        enhancer = PromptEnhancerLLM(modelKey)
        promptInput = utility.DiT_cleaner(promptInput)
        enhanced = enhancer.enhance_prompt(promptInput, seed=seed, precision=precision, configurator=configurator, multiply_max_length=multiply_max_length)
        return enhanced
    else:
        return False


def getPromptValues(filename, value):
    CONFIG_FILE = os.path.join(PRIMERE_ROOT, 'json', filename + '.json')
    CONFIG_FILE_EXAMPLE = os.path.join(PRIMERE_ROOT, 'json', filename + '.example.json')

    if Path(CONFIG_FILE).is_file() == True:
        CONFIG_SOURCE = CONFIG_FILE
    else:
        CONFIG_SOURCE = CONFIG_FILE_EXAMPLE

    ifConfigExist = os.path.isfile(CONFIG_SOURCE)
    if ifConfigExist == True:
        with open(CONFIG_SOURCE, 'r') as openfile:
            try:
                llm_config = json.load(openfile)
                if value in llm_config:
                    return llm_config[value]
                else:
                    return None
            except ValueError as e:
                return None
    else:
        return None

def getConfigKeys(config_name):
    CONFIG_FILE = os.path.join(PRIMERE_ROOT, 'json', config_name + '.json')
    CONFIG_FILE_EXAMPLE = os.path.join(PRIMERE_ROOT, 'json', config_name + '.example.json')

    if Path(CONFIG_FILE).is_file() == True:
        CONFIG_SOURCE = CONFIG_FILE
    else:
        CONFIG_SOURCE = CONFIG_FILE_EXAMPLE

    ifConfigExist = os.path.isfile(CONFIG_SOURCE)
    if ifConfigExist == True:
        with open(CONFIG_SOURCE, 'r') as openfile:
            try:
                llm_config = json.load(openfile)
                return list(llm_config.keys())
            except ValueError as e:
                return None
    else:
        return None

def configVariants(variant):
    CONFIG_FILE = os.path.join(PRIMERE_ROOT, 'json', 'llm_enhancer_config.json')
    CONFIG_FILE_EXAMPLE = os.path.join(PRIMERE_ROOT, 'json', 'llm_enhancer_config.example.json')

    if Path(CONFIG_FILE).is_file() == True:
        CONFIG_SOURCE = CONFIG_FILE
    else:
        CONFIG_SOURCE = CONFIG_FILE_EXAMPLE

    ifConfigExist = os.path.isfile(CONFIG_SOURCE)
    if ifConfigExist == True:
        with open(CONFIG_SOURCE, 'r') as openfile:
            try:
                llm_config = json.load(openfile)
                if variant in llm_config:
                    return llm_config[variant]
                else:
                    return {}
            except ValueError as e:
                return {}
    else:
        return {}

def getValidLLMPaths(model_root):
    valid_llm_path = []
    if os.path.exists(model_root):
        allsubdirs = list(os.listdir(Path(model_root)))
        for subdir in allsubdirs:
            path_config = os.path.join(model_root, subdir, 'config.json')
            path_model_bin = os.path.join(model_root, subdir, 'pytorch_model.bin')
            path_model_st = os.path.join(model_root, subdir, 'model.safetensors')
            current_sub = os.path.join(model_root, subdir)
            adapter_config = os.path.join(model_root, subdir, 'adapter_config.json')
            adapter_model_st = os.path.join(model_root, subdir, 'adapter_model.safetensors')
            matching_files = Path(current_sub).rglob('model-0*.safetensors')
            if (os.path.exists(path_config) == True or os.path.exists(adapter_config) == True) and (os.path.exists(adapter_model_st) == True or os.path.exists(path_model_bin) == True or os.path.exists(path_model_st) == True or len(list(matching_files)) > 0):
                if "PromptEnhancing" in subdir:
                    baseRepoName = subdir[:subdir.lower().index("-promptenhancing")]
                    baseRepoPath = os.path.join(model_root, baseRepoName)
                    if os.path.exists(baseRepoPath) == True:
                        valid_llm_path.append(subdir)
                else:
                    valid_llm_path.append(subdir)

    return valid_llm_path
