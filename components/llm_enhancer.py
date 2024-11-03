from pathlib import Path
import torch
from transformers import AutoModel, AutoModelForCausalLM, AutoTokenizer, AutoModelForSeq2SeqLM, set_seed, pipeline, GPT2Tokenizer, GPT2LMHeadModel, T5Tokenizer, T5ForConditionalGeneration, BloomTokenizerFast, BloomForCausalLM, BertTokenizer, BertForMaskedLM, DebertaV2Model, DebertaV2Config, DebertaV2Tokenizer, DebertaV2ForSequenceClassification, AlbertTokenizer, AlbertModel
from transformers.models.deberta.modeling_deberta import ContextPooler
from ..components.tree import PRIMERE_ROOT
import os
import json
import random
import re

class PromptEnhancerLLM:
    def __init__(self, model_path: str = "flan-t5-small"):
        model_access = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'LLM', model_path)
        self.model_path = model_path
        self.model_fullpath = model_access

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
        elif "albert-" in model_path.lower():
            self.tokenizer = AlbertTokenizer.from_pretrained(model_access, clean_up_tokenization_spaces=False)
            self.model = AlbertModel.from_pretrained(model_access, ignore_mismatched_sizes=True)
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(model_access, clean_up_tokenization_spaces=False)
            try:
                self.model = AutoModelForCausalLM.from_pretrained(model_access, ignore_mismatched_sizes=True)
            except Exception:
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_access, ignore_mismatched_sizes=True)

        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.tokenizer.add_special_tokens({'pad_token': '[PAD]'})
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def enhance_prompt(self, input_text: str, seed: int = 1, precision: bool = True, configurator: str = "default_settings"):
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
        instruction = f"Convert text to {configurator_name} stable diffusion text-to-image prompt: "
        settings = {**default_settings, **variant_params}

        if seed is not None and int(seed) > 1:
            random.seed(seed)
            newseed = random.randint(1, (2**32) - 1)
            set_seed(newseed)
            torch.manual_seed(newseed)
        else:
            set_seed(1)
            torch.manual_seed(1)

        forceFP16 = ['t5-efficient-base-dm256']
        forceFP32 = ['t5-efficient-base-dm512']

        if precision == False:
            self.model.half()

        with torch.no_grad():
            if "deberta-" in self.model_path.lower():
                enhanced_text = 'This moodel type not supported....'
            elif "albert-" in self.model_path.lower():
                enhanced_text = 'This moodel type not supported....'

            elif "gpt-neo-" in self.model_path.lower() or 'gpt2' in self.model_path.lower():
                generator = pipeline('text-generation', model=self.model_fullpath)

                outputs = generator(
                    instruction,
                    **settings,
                )
                enhanced_text = outputs[0]['generated_text']

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

            else:
                self.model.to(self.device)
                inputs = self.tokenizer(instruction + input_text, return_tensors="pt", max_length=512, truncation=True).to(self.device)
                attention_mask = None
                if "attention_mask" in inputs:
                    attention_mask = inputs["attention_mask"]

                outputs = self.model.generate(
                    inputs["input_ids"],
                    attention_mask=attention_mask,
                    **settings,
                    pad_token_id=self.tokenizer.eos_token_id
                )
                enhanced_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)

        if type(enhanced_text).__name__ == 'str':
            enhanced_text = re.sub("<[b][^>]*>(.+?)</[b]>", '', enhanced_text)
            enhanced_text = re.sub(r"http\S+", "", enhanced_text)
            enhanced_text = enhanced_text.replace(instruction, '').replace("system\nYou are a helpful AI assistant named SmolLM, trained by Hugging Face", '').replace('\nuser', '').replace('<pad>', '').replace('text to image', '').replace('texttoimage', '').replace('prompt', '').replace(r'\\', '').replace('!', '.')
            enhanced_text = re.sub(r'[^a-zA-Z0-9 ."?!()]', '', enhanced_text)
            return enhanced_text.strip('.: ')
        else:
            return False

def PrimereLLMEnhance(modelKey = 'flan-t5-small', promptInput = 'cute cat', seed = 1, precision = True, configurator = "default"):
    model_access = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'LLM', modelKey)
    if os.path.isdir(model_access) == True:
        enhancer = PromptEnhancerLLM(modelKey)
        enhanced = enhancer.enhance_prompt(promptInput, seed=seed, precision=precision, configurator=configurator)
        return enhanced
    else:
        return False

def getConfigKeys():
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