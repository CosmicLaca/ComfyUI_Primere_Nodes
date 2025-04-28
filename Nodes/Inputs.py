from ..components.tree import TREE_INPUTS
from ..components.tree import PRIMERE_ROOT
import os
import re
from dynamicprompts.parser.parse import ParserConfig
from dynamicprompts.wildcards.wildcard_manager import WildcardManager
import chardet
import pandas
import comfy.samplers
import folder_paths
from .modules.image_meta_reader import ImageExifReader
from .modules.image_meta_reader import compatibility_handler
from ..components import utility
from pathlib import Path
import random
import string
from .modules.adv_encode import advanced_encode
from ..components import stylehandler
from .Styles import StyleParser
import nodes
from .modules.exif_data_checker import check_model_from_exif
from ..utils import comfy_dir
from ..components import hypernetwork
import json
from ..components import llm_enhancer
import datetime

class PrimereDoublePrompt:
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("PROMPT+", "PROMPT-", "SUBPATH", "MODEL", "ORIENTATION", "PREFERRED")
    FUNCTION = "get_prompt"
    CATEGORY = TREE_INPUTS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_prompt": ("STRING", {"default": "", "multiline": True}),
                "negative_prompt": ("STRING", {"default": "", "multiline": True}),
            },
            "optional": {
                "subpath": ("STRING", {"default": "", "multiline": False}),
                "model": (["None"] + folder_paths.get_filename_list("checkpoints"), {"default": "None"}),
                "orientation": (["None", "Random", "Horizontal", "Vertical"], {"default": "None"}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "id": "UNIQUE_ID",
            },
        }

    def get_prompt(self, positive_prompt, negative_prompt, extra_pnginfo, id, subpath="", model="", orientation=""):
        def debug_state(self, extra_pnginfo, id):
            workflow = extra_pnginfo["workflow"]
            for node in workflow["nodes"]:
                node_id = str(node["id"])
                name = node["type"]
                if node_id == id and name == 'PrimerePrompt':
                    if "Debug" in name or "Show" in name or "Function" in name or "Evaluate" in name:
                        continue

                    return node['widgets_values']

        rawResult = debug_state(self, extra_pnginfo, id)
        if not rawResult:
            rawResult = (positive_prompt, negative_prompt)

        if len(subpath.strip()) < 1 or subpath.strip() == 'None':
            subpath = None
        if len(model.strip()) < 1 or model.strip() == 'None':
            model = None
        if len(orientation.strip()) < 1 or orientation.strip() == 'None':
            orientation = None

        if orientation == 'Random':
            orientations = ["Horizontal", "Vertical"]
            orientation = random.choice(orientations)

        preferred = {'subpath': subpath, 'model': model, 'orientation': orientation}

        return (rawResult[0].replace('\n', ' '), rawResult[1].replace('\n', ' '), subpath, model, orientation, preferred)

class PrimereRefinerPrompt:
    RETURN_TYPES = ("STRING", "STRING", "CONDITIONING", "CONDITIONING", "TUPLE", "STRING", "INT", "MODEL", "CLIP", "VAE")
    RETURN_NAMES = ("PROMPT+", "PROMPT-", "COND+", "COND-", "PROMPT_DATA", "MODEL_VERSION", "SQUARE_SHAPE", "MODEL", "CLIP", "VAE")
    FUNCTION = "refiner_prompt"
    CATEGORY = TREE_INPUTS

    @classmethod
    def INPUT_TYPES(cls):
        LYCO_DIR = os.path.join(folder_paths.models_dir, 'lycoris')
        folder_paths.add_model_folder_path("lycoris", LYCO_DIR)
        LyCORIS = folder_paths.get_filename_list("lycoris")
        LyCORISList = folder_paths.filter_files_extensions(LyCORIS, ['.ckpt', '.safetensors'])

        REFINER_LORA = ["LORA\\" + x for x in folder_paths.get_filename_list("loras")]
        REFINER_LYCORIS = ["LYCORIS\\" + x for x in LyCORISList]
        REFINER_EMBEDDING = ["EMBEDDING\\" + x for x in folder_paths.get_filename_list("embeddings")]
        REFINER_HYPERNETWORK = ["HYPERNETWORK\\" + x for x in folder_paths.get_filename_list("hypernetworks")]

        CONCEPT_LIST = utility.SUPPORTED_MODELS[0:17]
        CONCEPT_INPUTS = {}
        for concept in CONCEPT_LIST:
            CONCEPT_INPUTS["process_" + concept.lower()] = ("BOOLEAN", {"default": True, "label_on": "PROCESS " + concept.upper(), "label_off": "IGNORE " + concept.upper()})

        return {
            "required": {
                "refiner_model": (['None'] + folder_paths.get_filename_list("checkpoints"),),
                "refiner_vae": (['None'] + folder_paths.get_filename_list("vae"),),
                "refiner_network": (['None'] + REFINER_LORA + REFINER_LYCORIS + REFINER_EMBEDDING + REFINER_HYPERNETWORK,),
                "refiner_network_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, },),
                "refiner_network_insertion": ("BOOLEAN", {"default": True, "label_on": "POSITIVE", "label_off": "NEGATIVE"}),

                "positive_refiner": ("STRING", {"default": "", "multiline": True}),
                "negative_refiner": ("STRING", {"default": "", "multiline": True}),
                "positive_refiner_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "negative_refiner_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "positive_original_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "negative_original_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "token_normalization": (["none", "mean", "length", "length+mean"],),
                "weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),
                **CONCEPT_INPUTS
            },
            "optional": {
                "clip": ("CLIP",),
                "model": ("MODEL",),
                "vae": ("VAE",),
                "positive_original": ("STRING", {"forceInput": True}),
                "negative_original": ("STRING", {"forceInput": True}),
                "model_concept": ("STRING", {"forceInput": True, "default": 'Auto'}),
                "model_version": ("STRING", {"default": 'SD1', "forceInput": True}),
                "seed_input": ("INT", {"default": 1, "min": 0, "max": utility.MAX_SEED, "forceInput": True}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "id": "UNIQUE_ID",
            },
        }

    def __init__(self):
        wildcard_dir = os.path.join(PRIMERE_ROOT, 'wildcards')
        self._wildcard_manager = WildcardManager(wildcard_dir)
        self._parser_config = ParserConfig(
            variant_start = "{",
            variant_end = "}",
            wildcard_wrap = "__"
        )

    def refiner_prompt(self, extra_pnginfo, id, token_normalization, weight_interpretation, seed_input = 1, clip = None, model = None, vae = None, refiner_model = 'None', refiner_vae = 'None', refiner_network = 'None', refiner_network_weight = 1, refiner_network_insertion = True, positive_refiner = "", negative_refiner = "", positive_original = None, negative_original = None, model_concept = 'Auto', model_version = 'SD1', positive_refiner_strength = 1, negative_refiner_strength = 1, positive_original_strength = 1, negative_original_strength = 1,
                       **kwargs):

        if seed_input <= 1:
            random.seed(datetime.datetime.now().timestamp())
            seed_input = random.randint(1000, utility.MAX_SEED)
        def refiner_debug_state(self, extra_pnginfo, id):
            workflow = extra_pnginfo["workflow"]
            for node in workflow["nodes"]:
                node_id = str(node["id"])
                name = node["type"]
                if node_id == id and name == 'PrimereRefinerPrompt':
                    if "Debug" in name or "Show" in name or "Function" in name or "Evaluate" in name:
                        continue

                    return node['widgets_values']

        rawResult = refiner_debug_state(self, extra_pnginfo, id)
        if not rawResult:
            rawResult = (positive_refiner, negative_refiner)

        output_positive = rawResult[5].replace('\n', ' ')
        output_negative = rawResult[6].replace('\n', ' ')
        final_positive = ""
        final_negative = ""
        SQUARE_SHAPE = 768
        OUTPUT_MODEL = None
        OUTPUT_VAE = None
        refiner_state = True
        embeddings_final_pos = None
        embeddings_final_neg = None
        pooled_pos = None
        pooled_neg = None

        if model_concept == 'Auto':
           model_concept = model_version
        MODEL_VERSION = model_version

        input_data = kwargs
        SUPPORTED_CONCEPTS = utility.SUPPORTED_MODELS
        SUPPORTED_CONCEPTS_UC = [x.upper() for x in SUPPORTED_CONCEPTS]
        concept_processor = []
        for inputKey, inputValue in input_data.items():
            if inputKey.startswith("process_") == True:
                conceptSignUC = inputKey[len("process_"):].upper()
                conceptIndex = SUPPORTED_CONCEPTS_UC.index(conceptSignUC)
                CONCEPT_SIGN = SUPPORTED_CONCEPTS[conceptIndex]
                concept_processor.append(inputValue)
                if inputValue == False and model_concept == CONCEPT_SIGN:
                    refiner_state = False

        if (clip is None or model is None or vae is None) and refiner_model == 'None':
            refiner_state = False
            refiner_network = 'None'

        if positive_refiner_strength != 0:
            if positive_refiner_strength != 1:
                final_positive = f'({output_positive}:{positive_refiner_strength:.2f})' if output_positive is not None and output_positive != '' else ''
            else:
                final_positive = f'{output_positive}' if output_positive is not None and output_positive != '' else ''

        if negative_refiner_strength != 0:
            if negative_refiner_strength != 1:
                final_negative = f'({output_negative}:{negative_refiner_strength:.2f})' if output_negative is not None and output_negative != '' else ''
            else:
                final_negative = f'{output_negative}' if output_negative is not None and output_negative != '' else ''

        if positive_original is not None and positive_original != "" and positive_original_strength != 0:
            if positive_original_strength != 1:
                final_positive = f'{final_positive} ({positive_original}:{positive_original_strength:.2f})'
            else:
                final_positive = f'{final_positive} {positive_original}'

        if negative_original is not None and negative_original != "" and negative_original_strength != 0:
            if negative_original_strength != 1:
                final_negative = f'{final_negative} ({negative_original}:{negative_original_strength:.2f})'
            else:
                final_negative = f'{final_negative} {negative_original}'

        final_positive = utility.DynPromptDecoder(self, final_positive.strip(' ,;'), seed_input)
        final_negative = utility.DynPromptDecoder(self, final_negative.strip(' ,;'), seed_input)

        if model is not None and vae is not None and refiner_state == True and refiner_model == "None":
            OUTPUT_MODEL = model
            OUTPUT_VAE = vae

        elif model is None and refiner_model != "None" and refiner_state == True:
            LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, refiner_model)
            OUTPUT_MODEL = LOADED_CHECKPOINT[0]
            clip = LOADED_CHECKPOINT[1]
            if refiner_vae != 'None':
                OUTPUT_VAE = nodes.VAELoader.load_vae(self, refiner_vae)[0]
            else:
                OUTPUT_VAE = LOADED_CHECKPOINT[2]

        if refiner_network != 'None' and refiner_state == True:
            network_name = refiner_network
            network_data = network_name.split('\\', 1)
            network_path = network_data[1]

            match network_data[0]:
                case "LORA":
                    lora_path = folder_paths.get_full_path("loras", network_path)
                    lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                    model_lora = OUTPUT_MODEL
                    clip_lora = clip
                    OUTPUT_MODEL, clip = comfy.sd.load_lora_for_models(model_lora, clip_lora, lora, refiner_network_weight, refiner_network_weight)

                case "LYCORIS":
                    lycoris_path = folder_paths.get_full_path("lycoris", network_path)
                    lyco = comfy.utils.load_torch_file(lycoris_path, safe_load=True)
                    model_lyco = OUTPUT_MODEL
                    clip_lyco = clip
                    OUTPUT_MODEL, clip = comfy.sd.load_lora_for_models(model_lyco, clip_lyco, lyco, refiner_network_weight, refiner_network_weight)

                case "HYPERNETWORK":
                    cloned_model = OUTPUT_MODEL
                    hypernetwork_path = folder_paths.get_full_path("hypernetworks", network_path)
                    model_hypernetwork = cloned_model.clone()

                    try:
                        patch = hypernetwork.load_hypernetwork_patch(hypernetwork_path, refiner_network_weight, False)
                    except Exception:
                        patch = None

                    if patch is not None:
                        model_hypernetwork.set_model_attn1_patch(patch)
                        model_hypernetwork.set_model_attn2_patch(patch)
                        OUTPUT_MODEL = model_hypernetwork

                case "EMBEDDING":
                    embedd_name_path = network_path
                    embedd_weight = refiner_network_weight
                    embedd_neg = refiner_network_insertion
                    embedd_name = Path(embedd_name_path).stem
                    if (embedd_weight != 1):
                        embedding_string = '(embedding:' + embedd_name + ':' + str(embedd_weight) + ')'
                    else:
                        embedding_string = 'embedding:' + embedd_name
                    if embedd_neg == True:
                        final_positive = final_positive + ', ' + embedding_string
                    else:
                        final_negative = final_negative + ', ' + embedding_string

        if refiner_state == True:
            try:
                embeddings_final_pos, pooled_pos = advanced_encode(clip, final_positive, token_normalization, weight_interpretation, w_max=1.0, apply_to_pooled=True)
                embeddings_final_neg, pooled_neg = advanced_encode(clip, final_negative, token_normalization, weight_interpretation, w_max=1.0, apply_to_pooled=True)
            except Exception:
                tokens = clip.tokenize(final_positive)
                embeddings_final_pos, pooled_pos = clip.encode_from_tokens(tokens, return_pooled = True)

                tokens = clip.tokenize(final_negative)
                embeddings_final_neg, pooled_neg = clip.encode_from_tokens(tokens, return_pooled = True)

        prompt_tuple = {}
        prompt_tuple['refiner_state'] = refiner_state
        prompt_tuple['final_positive'] = final_positive
        prompt_tuple['final_negative'] = final_negative
        prompt_tuple['clip'] = clip
        prompt_tuple['token_normalization'] = token_normalization
        prompt_tuple['weight_interpretation'] = weight_interpretation
        prompt_tuple['cond_positive'] = [[embeddings_final_pos, {"pooled_output": pooled_pos}]]
        prompt_tuple['cond_negative'] = [[embeddings_final_neg, {"pooled_output": pooled_neg}]]
        prompt_tuple['refiner_model'] = refiner_model
        prompt_tuple['refiner_vae'] = refiner_vae
        prompt_tuple['refiner_network'] = refiner_network
        prompt_tuple['refiner_network_weight'] = refiner_network_weight
        prompt_tuple['refiner_network_insertion'] = refiner_network_insertion
        prompt_tuple['model_version'] = MODEL_VERSION
        prompt_tuple['square_shape'] = SQUARE_SHAPE
        # prompt_tuple['output_model'] = OUTPUT_MODEL
        # prompt_tuple['output_vae'] = OUTPUT_VAE
        prompt_tuple['model_concept'] = model_concept
        prompt_tuple['concept_processor'] = concept_processor

        return (final_positive, final_negative, [[embeddings_final_pos, {"pooled_output": pooled_pos}]], [[embeddings_final_neg, {"pooled_output": pooled_neg}]],
                prompt_tuple,
                MODEL_VERSION, SQUARE_SHAPE,
                OUTPUT_MODEL, clip, OUTPUT_VAE)

class PrimereLLMEnhancer:
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("PROMPT", "ENHANCED_PROMPT",)
    FUNCTION = "prompt_enhancer"
    CATEGORY = TREE_INPUTS

    TENC_DIR = os.path.join(folder_paths.models_dir, 'LLM')
    LLM_PRIMERE_ROOT = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'LLM')
    valid_llm_path = llm_enhancer.getValidLLMPaths(TENC_DIR)
    valid_llm_path += llm_enhancer.getValidLLMPaths(LLM_PRIMERE_ROOT)
    configurators = llm_enhancer.getConfigKeys("llm_enhancer_config")
    if configurators == None:
        configurators = ['Default']
    else:
        configurators = ['Default'] + configurators

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "prompt": ("STRING", {"default": False, "forceInput": True}),
                "seed": ("INT", {"default": 0, "min": 0, "max": (2**32) - 1, "forceInput": True}),
                "llm_model_path": (['None'] + cls.valid_llm_path,),
                "precision": ("BOOLEAN", {"default": True, "label_on": "FP32", "label_off": "FP16"}),
                "configurator": (cls.configurators,),
                "multiply_max_length": ("FLOAT", {"default": 1, "min": 0.1, "max": 25,  "step": 0.1}),
            }
        }

    def prompt_enhancer(self, prompt, seed, llm_model_path, precision, configurator, multiply_max_length = 1):
        if llm_model_path == 'None':
            return (prompt, "",)

        enhanced_result = llm_enhancer.PrimereLLMEnhance(llm_model_path, prompt, seed, precision, configurator, multiply_max_length)
        if enhanced_result == False:
            return (prompt, "",)

        return (prompt, enhanced_result,)

class PrimereImgToPrompt:
    RETURN_TYPES = ("STRING", "TUPLE",)
    RETURN_NAMES = ("PROMPT", "SYSTEM_PROMPT",)
    FUNCTION = "img_to_prompt"
    CATEGORY = TREE_INPUTS

    T2I_DIR = os.path.join(folder_paths.models_dir, 'img2text')
    valid_t2i_path = llm_enhancer.getValidLLMPaths(T2I_DIR)
    prompts = llm_enhancer.getConfigKeys("img2prompt_config")
    if prompts == None:
        prompts = ['Default']
    else:
        prompts = ['Default'] + prompts

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE", {"forceInput": True}),
                "model_path": (['None'] + cls.valid_t2i_path,),
                "result_control": (['Custom'] + cls.prompts,),
                "custom_prompt": ("STRING", {"default": False}),
            }
        }

    def img_to_prompt(self, image, model_path, result_control, custom_prompt):
        if model_path == 'None':
            return ("", [],)
        else:
            T2I_CUSTOMPATH = os.path.join(folder_paths.models_dir, 'img2text')
            model_access = os.path.join(T2I_CUSTOMPATH, model_path)
            if os.path.isdir(model_access) == False:
                return ("", [],)

            default_prompt = ['Image of', 'Image creation art style is', 'The dominant thing is', 'The background behind the main thing is', 'Dominant colours on the picture']

            if result_control == 'Custom':
                prompts = custom_prompt.split(',')
                if type(prompts).__name__ != 'list':
                    prompts = [custom_prompt]
            elif result_control == 'Default':
                prompts = default_prompt
            else:
                prompts = llm_enhancer.getPromptValues("img2prompt_config", result_control)
            if prompts is None or len(prompts) < 1:
                prompts = default_prompt

            story_out = utility.Pic2Story(model_access, image, prompts, True, False)
            if type(story_out) == str:
                return (story_out, prompts,)
            else:
                return ("", [],)

class PrimereStyleLoader:
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("PROMPT+", "PROMPT-", "SUBPATH", "MODEL", "ORIENTATION", "PREFERRED")
    FUNCTION = "load_csv"
    CATEGORY = TREE_INPUTS

    @staticmethod
    def load_styles_csv(styles_path: str):
        fileTest = open(styles_path, 'rb').readline()
        result = chardet.detect(fileTest)
        ENCODING = result['encoding']
        if ENCODING == 'ascii':
            ENCODING = 'UTF-8'

        with open(styles_path, "r", newline = '', encoding = ENCODING) as csv_file:
            try:
                return pandas.read_csv(csv_file)
            except pandas.errors.ParserError as e:
                errorstring = repr(e)
                matchre = re.compile('Expected (\d+) fields in line (\d+), saw (\d+)')
                (expected, line, saw) = map(int, matchre.search(errorstring).groups())
                print(f'Error at line {line}. Fields added : {saw - expected}.')

    @classmethod
    def INPUT_TYPES(cls):
        STYLE_DIR = os.path.join(PRIMERE_ROOT, 'stylecsv')
        STYLE_FILE = os.path.join(STYLE_DIR, "styles.csv")
        STYLE_FILE_EXAMPLE = os.path.join(STYLE_DIR, "styles.example.csv")

        if Path(STYLE_FILE).is_file() == True:
            STYLE_SOURCE = STYLE_FILE
        else:
            STYLE_SOURCE = STYLE_FILE_EXAMPLE
        cls.styles_csv = cls.load_styles_csv(STYLE_SOURCE)

        return {
            "required": {
                "styles": (sorted(list(cls.styles_csv['name'])),),
                "use_subpath": ("BOOLEAN", {"default": False}),
                "use_model": ("BOOLEAN", {"default": False}),
                "use_orientation": ("BOOLEAN", {"default": False}),
            },
        }

    def load_csv(self, styles, use_subpath, use_model, use_orientation):
        try:
            positive_prompt = self.styles_csv[self.styles_csv['name'] == styles]['prompt'].values[0]
        except Exception:
            positive_prompt = ''

        try:
            negative_prompt = self.styles_csv[self.styles_csv['name'] == styles]['negative_prompt'].values[0]
        except Exception:
            negative_prompt = ''

        try:
            preferred_subpath = self.styles_csv[self.styles_csv['name'] == styles]['preferred_subpath'].values[0]
        except Exception:
            preferred_subpath = ''

        try:
            preferred_model = self.styles_csv[self.styles_csv['name'] == styles]['preferred_model'].values[0]
        except Exception:
            preferred_model = ''

        try:
            preferred_orientation = self.styles_csv[self.styles_csv['name'] == styles]['preferred_orientation'].values[0]
        except Exception:
            preferred_orientation = ''

        pos_type = type(positive_prompt).__name__
        neg_type = type(negative_prompt).__name__
        subp_type = type(preferred_subpath).__name__
        model_type = type(preferred_model).__name__
        orientation_type = type(preferred_orientation).__name__

        if (pos_type != 'str'):
            positive_prompt = ''
        if (neg_type != 'str'):
            negative_prompt = ''
        if (subp_type != 'str'):
            preferred_subpath = ''
        if (model_type != 'str'):
            preferred_model = ''
        if (orientation_type != 'str'):
            preferred_orientation = ''

        if len(preferred_subpath.strip()) < 1:
            preferred_subpath = None
        if len(preferred_model.strip()) < 1:
            preferred_model = None
        if len(preferred_orientation.strip()) < 1:
            preferred_orientation = None

        if use_subpath == False:
            preferred_subpath = None
        if use_model == False:
            preferred_model = None
        if use_orientation == False:
            preferred_orientation = None

        preferred = {'subpath': preferred_subpath, 'model': preferred_model, 'orientation': preferred_orientation}

        return (positive_prompt, negative_prompt, preferred_subpath, preferred_model, preferred_orientation, preferred)

class PrimereDynParser:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PROMPT",)
    FUNCTION = "dyndecoder"
    CATEGORY = TREE_INPUTS

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "dyn_prompt": ("STRING", {"multiline": True, "forceInput": True}),
                "seed": ("INT", {"default": 0, "min": -1, "max": utility.MAX_SEED, "forceInput": True}),
            }
        }

    def __init__(self):
        wildcard_dir = os.path.join(PRIMERE_ROOT, 'wildcards')
        self._wildcard_manager = WildcardManager(wildcard_dir)
        self._parser_config = ParserConfig(
            variant_start = "{",
            variant_end = "}",
            wildcard_wrap = "__"
        )

    def dyndecoder(self, dyn_prompt, seed):
        prompt = utility.DynPromptDecoder(self, dyn_prompt, seed)
        return (prompt, )

class PrimereEmbeddingHandler:
    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("PROMPT+", "PROMPT-",)
    FUNCTION = "embedding_handler"
    CATEGORY = TREE_INPUTS
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_prompt": ("STRING", {"multiline": True, "forceInput": True}),
                "negative_prompt": ("STRING", {"multiline": True, "forceInput": True}),
            }
        }

    def embedding_handler(self, positive_prompt, negative_prompt):
        return (self.EmbeddingConverter(positive_prompt), self.EmbeddingConverter(negative_prompt),)

    def EmbeddingConverter(self, text):
        word_list = text.split()
        new_word_list = [i.strip(string.punctuation) if type(i) == str else str(i) for i in word_list]

        EMBEDDINGS = folder_paths.get_filename_list("embeddings")
        text = text.replace('embedding:', '')
        reg = re.compile(".*:\d")
        matchlist = list(filter(reg.match, new_word_list))

        for embeddings_path in EMBEDDINGS:
            path = Path(embeddings_path)
            embedding_name = path.stem
            if (embedding_name in new_word_list):
                text = text.replace(embedding_name, 'embedding:' + embedding_name)
            if any((reg.match(item)) for item in new_word_list):
                if any(item for item in matchlist if item.startswith(embedding_name)) == True:
                    if f'embedding:{embedding_name}' not in text:
                        text = text.replace(embedding_name, 'embedding:' + embedding_name)

        return text

class PrimereMetaHandler:
    CATEGORY = TREE_INPUTS
    RETURN_TYPES = ("TUPLE", "TUPLE", "IMAGE")
    RETURN_NAMES = ("WORKFLOW_TUPLE", "ORIGINAL_EXIF", "LOADED_IMAGE")
    FUNCTION = "image_meta_handler"

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        return {
            "required": {
                "data_source": ("BOOLEAN", {"default": False, "label_on": "Use image meta", "label_off": "Use workflow settings"}),
                "prompt_surce": ("BOOLEAN", {"default": True, "label_on": "Meta or workflow", "label_off": "Pic2story model"}),
                "prompt_state": ("BOOLEAN", {"default": False, "label_on": "Use decoded prompt", "label_off": "Use dynamic prompt"}),
                "model": ("BOOLEAN", {"default": True, "label_on": "Meta model", "label_off": "Workflow model"}),
                "model_hash_check": ("BOOLEAN", {"default": False, "label_on": "Check model hash", "label_off": "Use model name"}),
                "sampler": ("BOOLEAN", {"default": True, "label_on": "Meta sampler", "label_off": "Workflow sampler"}),
                "scheduler": ("BOOLEAN", {"default": True, "label_on": "Meta scheduler", "label_off": "Workflow scheduler"}),
                "cfg": ("BOOLEAN", {"default": True, "label_on": "Meta CFG", "label_off": "Workflow CFG"}),
                "steps": ("BOOLEAN", {"default": True, "label_on": "Meta steps", "label_off": "Workflow steps"}),
                "seed": ("BOOLEAN", {"default": True, "label_on": "Meta seed", "label_off": "Workflow seed"}),
                "image_size": ("BOOLEAN", {"default": True, "label_on": "Meta size", "label_off": "Workflow size"}),
                "recount_image": ("BOOLEAN", {"default": False, "label_on": "Round to Standard", "label_off": "Accurate image size"}),
                "vae": ("BOOLEAN", {"default": True, "label_on": "Meta VAE", "label_off": "Workflow VAE"}),
                "force_vae": ("BOOLEAN", {"default": False, "label_on": "Baked VAE", "label_off": "Custom VAE"}),
                "model_concept": ("BOOLEAN", {"default": False, "label_on": "Meta settings", "label_off": "Workflow settings"}),

                "latent_setup": ("BOOLEAN", {"default": False, "label_on": "Meta settings", "label_off": "Workflow settings"}),
                "lora_setup": ("BOOLEAN", {"default": False, "label_on": "Meta settings", "label_off": "Workflow settings"}),
                "lycoris_setup": ("BOOLEAN", {"default": False, "label_on": "Meta settings", "label_off": "Workflow settings"}),
                "embedding_setup": ("BOOLEAN", {"default": False, "label_on": "Meta settings", "label_off": "Workflow settings"}),
                "hypernetwork_setup": ("BOOLEAN", {"default": False, "label_on": "Meta settings", "label_off": "Workflow settings"}),
                "sampler_setup": ("BOOLEAN", {"default": False, "label_on": "Meta settings", "label_off": "Workflow settings"}),
                "clip_encoder_setup": ("BOOLEAN", {"default": False, "label_on": "Meta settings", "label_off": "Workflow settings"}),
                "clip_optional_prompts": ("BOOLEAN", {"default": False, "label_on": "Meta settings", "label_off": "Workflow settings"}),
                "clip_style_prompts": ("BOOLEAN", {"default": False, "label_on": "Meta settings", "label_off": "Workflow settings"}),
                "clip_additional_keywords": ("BOOLEAN", {"default": False, "label_on": "Meta settings", "label_off": "Workflow settings"}),

                "preferred": ("BOOLEAN", {"default": False, "label_on": "From meta", "label_off": "From workflow"}),
                "use_preferred": ("BOOLEAN", {"default": False, "label_on": "Use preferred settings", "label_off": "Cancel preferred settings"}),

                "image": (sorted(files),),
            },
            "optional": {
                "workflow_tuple": ("TUPLE", {"default": None}),
            },
        }

    def image_meta_handler(self, *args, **kwargs):
        workflow_tuple = None
        original_exif = None
        is_sdxl = 0

        image_path = folder_paths.get_annotated_filepath(kwargs['image'])

        if 'workflow_tuple' in kwargs and kwargs['workflow_tuple'] is not None and kwargs['data_source'] == False:
            workflow_tuple = kwargs['workflow_tuple']
            workflow_tuple['exif_status'] = 'OFF'

            wf_model_concept = None
            if 'model_concept' in workflow_tuple:
                wf_model_concept = workflow_tuple['model_concept']

            if 'preferred' in workflow_tuple:
                prefred_settings = workflow_tuple['preferred']
                if len(prefred_settings) > 0:
                    for prefkey, prefval in prefred_settings.items():
                        if prefval is not None:
                            match prefkey:
                                case "model":
                                    if wf_model_concept == 'Normal':
                                        ValidModel = check_model_from_exif(None, prefval, prefval, False)
                                        workflow_tuple['model'] = ValidModel

                                case "orientation":
                                    origW = workflow_tuple['width']
                                    origH = workflow_tuple['height']
                                    if prefval == 'Vertical':
                                        if origW > origH:
                                            workflow_tuple['width'] = origH
                                            workflow_tuple['height'] = origW
                                    else:
                                        if origW < origH:
                                            workflow_tuple['width'] = origH
                                            workflow_tuple['height'] = origW
                                    workflow_tuple['size_string'] = str(workflow_tuple['width']) + 'x' + str(workflow_tuple['height'])

                    modelname_only = Path((workflow_tuple['model'])).stem
                    model_version = utility.get_value_from_cache('model_version', modelname_only)
                    if model_version is None and (wf_model_concept == "Normal" or wf_model_concept is None):
                        checkpointpaths = folder_paths.get_folder_paths("checkpoints")[0]
                        model_full_path = checkpointpaths + os.sep + workflow_tuple['model']
                        model_file = Path(model_full_path)
                        if model_file.is_file() == True:
                            # LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, workflow_tuple['model'])
                            # model_version = utility.getCheckpointVersion(LOADED_CHECKPOINT[0])
                            model_version = utility.getModelType(workflow_tuple['model'], 'checkpoints')
                            utility.add_value_to_cache('model_version', modelname_only, model_version)

                    workflow_tuple['model_version'] = model_version
                    if 'model_shapes' in workflow_tuple and workflow_tuple['model_shapes'] is not None:
                        wf_square_shape = workflow_tuple['model_shapes']['SD']
                    else:
                        wf_square_shape = 768
                    match model_version:
                        case 'SDXL_2048':
                            is_sdxl = 1
                            if 'model_shapes' in workflow_tuple and workflow_tuple['model_shapes'] is not None:
                                wf_square_shape = workflow_tuple['model_shapes']['SDXL']
                            else:
                                wf_square_shape = 1024
                    workflow_tuple['is_sdxl'] = is_sdxl

                    if workflow_tuple['width'] > workflow_tuple['height']:
                        orientation = 'Horizontal'
                    else:
                        orientation = 'Vertical'

                    dimensions = utility.get_dimensions_by_shape(self, 'Square [1:1]', wf_square_shape, orientation, False, True, workflow_tuple['width'], workflow_tuple['height'], 'STANDARD')
                    workflow_tuple['width'] = dimensions[0]
                    workflow_tuple['height'] = dimensions[1]
                    workflow_tuple['size_string'] = str(workflow_tuple['width']) + 'x' + str(workflow_tuple['height'])

        elif kwargs['data_source'] == True:
            if os.path.isfile(image_path):
                readerResult = ImageExifReader(image_path)
                if type(readerResult.parser).__name__ == 'dict':
                    print('Reader tool return empty, using workflow settings')
                    if 'workflow_tuple' in kwargs and kwargs['workflow_tuple'] is not None:
                        workflow_tuple = kwargs['workflow_tuple']
                        workflow_tuple['exif_status'] = 'FAILED'
                else:
                    reader = readerResult.parser
                    workflow_tuple = reader.parameter

                    meta_model_concept = None
                    if 'model_concept' in workflow_tuple:
                        meta_model_concept = workflow_tuple['model_concept']

                    original_exif = readerResult.original
                    exif_data_count = len(workflow_tuple)
                    try:
                        workflow_tuple['meta_source'] = readerResult.tool
                    except Exception:
                        workflow_tuple['meta_source'] = reader.__class__.__name__
                    if meta_model_concept == "Normal" or meta_model_concept is None:
                        workflow_tuple = compatibility_handler(workflow_tuple, workflow_tuple['meta_source'])
                    workflow_tuple['exif_status'] = 'SUCCEED'
                    workflow_tuple['exif_data_count'] = exif_data_count

                    if kwargs['prompt_state'] == True:
                        workflow_tuple['prompt_state'] = 'Decoded'
                        if 'decoded_positive' in workflow_tuple:
                            workflow_tuple['positive'] = workflow_tuple['decoded_positive']
                        if 'decoded_negative' in workflow_tuple:
                            workflow_tuple['negative'] = workflow_tuple['decoded_negative']
                    else:
                        workflow_tuple['prompt_state'] = 'Dynamic'

                    if 'workflow_tuple' in kwargs and kwargs['workflow_tuple'] is not None:
                        workflow_tuple['wf'] = {}
                        for inputkey, inputfval in kwargs['workflow_tuple'].items():
                            if inputkey not in workflow_tuple:
                                workflow_tuple['wf'][inputkey] = inputfval

            if workflow_tuple is not None:
                if len(workflow_tuple) >= 1 and 'workflow_tuple' in kwargs:
                    if kwargs['workflow_tuple'] is not None and len(kwargs['workflow_tuple']) >= 1:
                        for controlkey, controlval in kwargs.items():

                            match controlkey:
                                case "model":
                                    if controlval == False:
                                        if 'model' in kwargs['workflow_tuple']:
                                            workflow_tuple['model'] = kwargs['workflow_tuple']['model']

                                case "sampler":
                                    if controlval == False:
                                        if 'sampler' in kwargs['workflow_tuple']:
                                            workflow_tuple['sampler'] = kwargs['workflow_tuple']['sampler']

                                case "scheduler":
                                    if controlval == False:
                                        if 'scheduler' in kwargs['workflow_tuple']:
                                            workflow_tuple['scheduler'] = kwargs['workflow_tuple']['scheduler']

                                case "cfg":
                                    if controlval == False:
                                        if 'cfg' in kwargs['workflow_tuple']:
                                            workflow_tuple['cfg'] = kwargs['workflow_tuple']['cfg']

                                case "steps":
                                    if controlval == False:
                                        if 'steps' in kwargs['workflow_tuple']:
                                            workflow_tuple['steps'] = kwargs['workflow_tuple']['steps']

                                case "image_size":
                                    if controlval == False:
                                        if 'width' in kwargs['workflow_tuple'] and 'height' in kwargs['workflow_tuple']:
                                            workflow_tuple['width'] = kwargs['workflow_tuple']['width']
                                            workflow_tuple['height'] = kwargs['workflow_tuple']['height']
                                            workflow_tuple['size_string'] = str(kwargs['workflow_tuple']['width']) + 'x' + str(kwargs['workflow_tuple']['height'])

                                case "vae":
                                    if controlval == False:
                                        if 'vae' in kwargs['workflow_tuple']:
                                            workflow_tuple['vae'] = kwargs['workflow_tuple']['vae']

                                case "force_vae":
                                    if controlval == True:
                                        workflow_tuple['vae'] = 'Baked VAE'

                                case "model_concept":
                                    if controlval == False:
                                        if 'model_concept' in kwargs['workflow_tuple']:
                                            workflow_tuple['model_concept'] = kwargs['workflow_tuple']['model_concept']
                                        if 'concept_data' in kwargs['workflow_tuple']:
                                            workflow_tuple['concept_data'] = kwargs['workflow_tuple']['concept_data']

                                case "preferred":
                                    if controlval == False:
                                        if 'preferred' in kwargs['workflow_tuple']:
                                            workflow_tuple['preferred'] = kwargs['workflow_tuple']['preferred']

                                case "use_preferred":
                                    if controlval == True:
                                        if 'preferred' in workflow_tuple:
                                            if workflow_tuple['preferred']['model'] is not None:
                                                ValidModel = check_model_from_exif(None, workflow_tuple['preferred']['model'], workflow_tuple['preferred']['model'], False)
                                                workflow_tuple['model'] = ValidModel
                                            if workflow_tuple['preferred']['orientation'] is not None:
                                                origW = workflow_tuple['width']
                                                origH = workflow_tuple['height']
                                                if workflow_tuple['preferred']['orientation'] == 'Vertical':
                                                    if origW > origH:
                                                        workflow_tuple['width'] = origH
                                                        workflow_tuple['height'] = origW
                                                else:
                                                    if origW < origH:
                                                        workflow_tuple['width'] = origH
                                                        workflow_tuple['height'] = origW
                                                workflow_tuple['size_string'] = str(workflow_tuple['width']) + 'x' + str(workflow_tuple['height'])

                if 'model' not in workflow_tuple:
                    if 'wf' in workflow_tuple and 'model' in workflow_tuple['wf']:
                        workflow_tuple['model'] = workflow_tuple['wf']['model']

                meta_model_concept = None
                if 'model_concept' in workflow_tuple:
                    meta_model_concept = workflow_tuple['model_concept']

                is_sdxl = 0
                if 'model' in workflow_tuple:
                    modelname_only = Path((workflow_tuple['model'])).stem
                    model_version = utility.get_value_from_cache('model_version', modelname_only)
                    if model_version is None and meta_model_concept == 'Normal':
                        checkpointpaths = folder_paths.get_folder_paths("checkpoints")[0]
                        model_full_path = checkpointpaths + os.sep + workflow_tuple['model']
                        model_file = Path(model_full_path)
                        if model_file.is_file() == True:
                            # LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, workflow_tuple['model'])
                            # model_version = utility.getCheckpointVersion(LOADED_CHECKPOINT[0])
                            model_version = utility.getModelType(workflow_tuple['model'], 'checkpoints')
                            utility.add_value_to_cache('model_version', modelname_only, model_version)
                        else:
                            allcheckpoints = folder_paths.get_filename_list("checkpoints")
                            modelname_only = Path((allcheckpoints[0])).stem
                            workflow_tuple['model'] = allcheckpoints[0]
                            model_version = utility.get_value_from_cache('model_version', modelname_only)
                            if model_version is None:
                                checkpointpaths = folder_paths.get_folder_paths("checkpoints")[0]
                                model_full_path = checkpointpaths + os.sep + allcheckpoints[0]
                                model_file = Path(model_full_path)
                                if model_file.is_file() == True:
                                    # LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, workflow_tuple['model'])
                                    # model_version = utility.getCheckpointVersion(LOADED_CHECKPOINT[0])
                                    model_version = utility.getModelType(workflow_tuple['model'], 'checkpoints')
                                    utility.add_value_to_cache('model_version', modelname_only, model_version)
                                else:
                                    model_version = 'BaseModel_1024'

                    workflow_tuple['model_version'] = model_version
                    if 'model_shapes' in workflow_tuple and workflow_tuple['model_shapes'] is not None:
                        wf_square_shape = workflow_tuple['model_shapes']['SD']
                    else:
                        wf_square_shape = 768
                    match model_version:
                        case 'SDXL_2048':
                            is_sdxl = 1
                            if 'model_shapes' in workflow_tuple and workflow_tuple['model_shapes'] is not None:
                                wf_square_shape = workflow_tuple['model_shapes']['SDXL']
                            else:
                                wf_square_shape = 1024
                    workflow_tuple['is_sdxl'] = is_sdxl

                if kwargs['recount_image'] == True and 'width' in workflow_tuple and 'height' in workflow_tuple:
                    if workflow_tuple['width'] > workflow_tuple['height']:
                        orientation = 'Horizontal'
                    else:
                        orientation = 'Vertical'

                    dimensions = utility.get_dimensions_by_shape(self, 'Square [1:1]', wf_square_shape, orientation, False, True, workflow_tuple['width'], workflow_tuple['height'], 'STANDARD')
                    workflow_tuple['width'] = dimensions[0]
                    workflow_tuple['height'] = dimensions[1]
                    workflow_tuple['size_string'] = str(workflow_tuple['width']) + 'x' + str(workflow_tuple['height'])

                if 'workflow_tuple' in kwargs and kwargs['workflow_tuple'] is not None:
                    if (is_sdxl == 1):
                        if 'vae_name_sdxl' in kwargs['workflow_tuple'] and kwargs['workflow_tuple']['vae_name_sdxl'] is not None:
                            workflow_tuple['vae'] = kwargs['workflow_tuple']['vae_name_sdxl']
                    else:
                        if 'vae_name_sd' in kwargs['workflow_tuple'] and kwargs['workflow_tuple']['vae_name_sd'] is not None:
                            workflow_tuple['vae'] = kwargs['workflow_tuple']['vae_name_sd']

                if 'vae' not in workflow_tuple or workflow_tuple['vae'] == "" or workflow_tuple['vae'] is None:
                    if kwargs['force_vae'] == True:
                        workflow_tuple['vae'] = 'Baked VAE'
                    else:
                        workflow_tuple['vae'] = 'External VAE'

        if kwargs['seed'] == False and 'workflow_tuple' not in kwargs:
            workflow_tuple['seed'] = random.randint(1, utility.MAX_SEED)

        if kwargs['force_vae'] == True and kwargs['vae'] == False:
            workflow_tuple['vae'] = 'Baked VAE'

        if workflow_tuple is not None and 'positive' in workflow_tuple:
            PosPromptType = type(workflow_tuple['positive']).__name__
            if PosPromptType is not None and PosPromptType != 'str':
                workflow_tuple['positive'] = 'Red sportcar racing'

        if workflow_tuple is not None and 'negative' in workflow_tuple:
            NegPromptType = type(workflow_tuple['negative']).__name__
            if NegPromptType is not None and NegPromptType != 'str':
                workflow_tuple['negative'] = 'Cute cat, nsfw, nude, nudity, porn'

        if (workflow_tuple is not None and 'model_concept' in workflow_tuple and workflow_tuple['model_concept'] == 'Cascade'):
            if ('vae' not in workflow_tuple or ('vae' in workflow_tuple and workflow_tuple['vae'] != 'Baked VAE')):
                if 'concept_data' in workflow_tuple and 'cascade_stage_a' in workflow_tuple['concept_data']:
                    if workflow_tuple['concept_data']['cascade_stage_a'] is not None:
                        workflow_tuple['vae'] = workflow_tuple['concept_data']['cascade_stage_a']
                        workflow_tuple['is_sdxl'] = 1
                        workflow_tuple['model_version'] = 'SDXL_2048'

                        if workflow_tuple['width'] > workflow_tuple['height']:
                            orientation = 'Horizontal'
                        else:
                            orientation = 'Vertical'

                        dimensions = utility.get_dimensions_by_shape(self, 'Square [1:1]', workflow_tuple['model_shapes']['SDXL'], orientation, True, True, workflow_tuple['width'], workflow_tuple['height'], 'CASCADE')
                        workflow_tuple['width'] = dimensions[0]
                        workflow_tuple['height'] = dimensions[1]
                        workflow_tuple['size_string'] = str(workflow_tuple['width']) + 'x' + str(workflow_tuple['height'])
            else:
                workflow_tuple['is_sdxl'] = 1
                workflow_tuple['model_version'] = 'SDXL_2048'

                if workflow_tuple['width'] > workflow_tuple['height']:
                    orientation = 'Horizontal'
                else:
                    orientation = 'Vertical'

                dimensions = utility.get_dimensions_by_shape(self, 'Square [1:1]', workflow_tuple['model_shapes']['SDXL'], orientation, True, True, workflow_tuple['width'], workflow_tuple['height'], 'CASCADE')
                workflow_tuple['width'] = dimensions[0]
                workflow_tuple['height'] = dimensions[1]
                workflow_tuple['size_string'] = str(workflow_tuple['width']) + 'x' + str(workflow_tuple['height'])

        if (workflow_tuple is not None and 'model_concept' in workflow_tuple and workflow_tuple['model_concept'] == 'Flux'):
            if 'concept_data' in workflow_tuple and 'flux_vae' in workflow_tuple['concept_data']:
                if workflow_tuple['concept_data']['flux_vae'] is not None:
                    workflow_tuple['vae'] = workflow_tuple['concept_data']['flux_vae']
                    workflow_tuple['is_sdxl'] = 1
                    workflow_tuple['model_version'] = 'SDXL_2048'

        def DictSort(element):
            if element in utility.WORKFLOW_SORT_LIST:
                return utility.WORKFLOW_SORT_LIST.index(element)
            else:
                return len(utility.WORKFLOW_SORT_LIST)

        image_file = Path(image_path)
        if 'image' in kwargs and image_file.is_file() == True:
            img = nodes.LoadImage.load_image(self, kwargs['image'])[0]

            if kwargs['prompt_surce'] == False and workflow_tuple is None:
                workflow_tuple = {}
                workflow_tuple['exif_status'] = 'FAILED'

            if kwargs['prompt_surce'] != False and workflow_tuple is not None:
                workflow_tuple['pic2story'] = 'OFF'

            if kwargs['prompt_surce'] == False and workflow_tuple is not None:
                repo_id = "abhijit2111/Pic2Story"
                prompts = ['Image of', 'Image creation style is', 'Colours on the picture']

                story_out = utility.Pic2Story(repo_id, img, prompts, True, True)
                if type(story_out) == str:
                    workflow_tuple['pic2story'] = 'SUCCEED'
                    workflow_tuple['pic2story_positive'] = story_out
                else:
                    workflow_tuple['pic2story'] = 'FAILED'

        else:
            img = None

        if workflow_tuple is not None and len(workflow_tuple) >= 1:
            workflow_tuple = dict(sorted(workflow_tuple.items(), key=lambda pair: DictSort(pair[0])))
            workflow_tuple['setup_states'] = kwargs
            if 'workflow_tuple' in workflow_tuple['setup_states']:
                del workflow_tuple['setup_states']['workflow_tuple']

        if type(original_exif).__name__ == 'str':
            try:
                original_exif = json.loads(original_exif)
            except Exception:
                original_exif = original_exif

        if type(original_exif).__name__ == 'dict':
            try:
                if 'prompt' in original_exif:
                    original_exif['prompt'] = json.loads(original_exif['prompt'])
                if 'workflow' in original_exif:
                    original_exif['workflow'] = json.loads(original_exif['workflow'])
                if 'gendata' in original_exif:
                    original_exif['gendata'] = json.loads(original_exif['gendata'])
            except Exception:
                original_exif = original_exif

        return (workflow_tuple, original_exif, img,)

class PrimereMetaDistributor:
    CATEGORY = TREE_INPUTS
    RETURN_TYPES = ("STRING", "STRING",  "STRING", "STRING", "STRING", "STRING", "CHECKPOINT_NAME", "STRING", "STRING", "TUPLE", "VAE_NAME", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT", "FLOAT", "INT", "INT", "INT")
    RETURN_NAMES = ("PROMPT+", "PROMPT-", "PROMPT L+", "PROMPT L-", "PROMPT R+", "PROMPT R-", "MODEL", "MODEL_VERSION", "MODEL_CONCEPT", "CONCEPT_DATA", "VAE", "SAMPLER", "SCHEDULER", "STEPS", "CFG", "SEED", "WIDTH", "HEIGHT")
    FUNCTION = "expand_meta"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_tuple": ("TUPLE", {"default": None}),
            },
        }

    def expand_meta(self, workflow_tuple):
        PROCESSED_KEYS = ['pic2story', 'positive', 'negative', 'positive_l', 'negative_l', 'positive_r', 'negative_r',
                          'model', 'model_version', 'model_concept', 'concept_data', 'vae',
                          'sampler', 'scheduler', 'steps', 'cfg',
                          'seed', 'width', 'height']
        OUTPUT_TUPLE = []

        if workflow_tuple is not None and type(workflow_tuple).__name__ == 'dict':
            for outputkeys in PROCESSED_KEYS:
                if outputkeys in workflow_tuple:
                    match outputkeys:
                        case "pic2story":
                            if workflow_tuple[outputkeys] == 'SUCCEED' and 'pic2story_positive' in workflow_tuple:
                                workflow_tuple['positive'] = workflow_tuple['pic2story_positive']
                                workflow_tuple['prompt_state'] = 'Dynamic'
                                workflow_tuple['exif_status'] = 'OFF'
                                if 'decoded_positive' in workflow_tuple:
                                    del workflow_tuple['decoded_positive']
                                if 'decoded_negative' in workflow_tuple:
                                    del workflow_tuple['decoded_negative']
                                if 'pic2story_positive' in workflow_tuple:
                                    del workflow_tuple['pic2story_positive']
                                if 'exif_data_count' in workflow_tuple:
                                    del workflow_tuple['exif_data_count']
                                if 'meta_source' in workflow_tuple:
                                    del workflow_tuple['meta_source']
                        case _:
                            output_value = workflow_tuple[outputkeys]
                            if output_value == "":
                                output_value = None
                            OUTPUT_TUPLE.append(output_value)
                else:
                    MISSING_VALUES = None
                    match outputkeys:
                        case "vae":
                            if 'model_concept' in workflow_tuple and 'concept_data' in workflow_tuple and workflow_tuple['model_concept'] == 'Flux':
                                    MISSING_VALUES = workflow_tuple['concept_data']['flux_vae']
                            else:
                                if 'model_version' in workflow_tuple:
                                    if workflow_tuple['model_version'] == 'SDXL_2048':
                                        if 'vae_name_sdxl' in workflow_tuple:
                                            MISSING_VALUES = workflow_tuple['vae_name_sdxl']
                                    else:
                                        if 'vae_name_sd' in workflow_tuple:
                                            MISSING_VALUES = workflow_tuple['vae_name_sd']

                    OUTPUT_TUPLE.append(MISSING_VALUES)

        return OUTPUT_TUPLE

class PrimereMetaDistributorStage2:
    CATEGORY = TREE_INPUTS
    RETURN_TYPES = ("INT", "INT", "INT", "TUPLE")
    RETURN_NAMES = ("SEED", "WIDTH", "HEIGHT", "WORKFLOW_TUPLE")
    FUNCTION = "expand_meta_2"

    def __init__(self):
        wildcard_dir = os.path.join(PRIMERE_ROOT, 'wildcards')
        self._wildcard_manager = WildcardManager(wildcard_dir)
        self._parser_config = ParserConfig(
            variant_start = "{",
            variant_end = "}",
            wildcard_wrap = "__"
        )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": 0, "min": -1, "max": utility.MAX_SEED, "forceInput": True}),
                "width": ('INT', {"forceInput": True, "default": 512}),
                "height": ('INT', {"forceInput": True, "default": 512}),
                # "rnd_orientation": ("BOOLEAN", {"default": False}),

                "workflow_tuple": ("TUPLE", {"default": None}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "id": "UNIQUE_ID",
                "prompt": "PROMPT"
            },

        }
    def expand_meta_2(self, workflow_tuple, seed, width, height, prompt, **kwargs):
        PROCESSED_KEYS = ['setup_states']
        OUTPUT_TUPLE = []
        IMG_WIDTH = width
        IMG_HEIGHT = height
        EXT_SEED = seed
        prompt_state_setup = False

        if workflow_tuple is not None and type(workflow_tuple).__name__ == 'dict' and 'exif_status' in workflow_tuple and workflow_tuple['exif_status'] == 'SUCCEED':
            for outputkeys in PROCESSED_KEYS:
                if outputkeys in workflow_tuple:
                    match outputkeys:
                        case "setup_states":
                            RECYCLER_SETUP = workflow_tuple['setup_states']
                            if 'seed' in RECYCLER_SETUP:
                                if RECYCLER_SETUP['seed'] == True and 'seed' in workflow_tuple and workflow_tuple['seed'] > 1:
                                    OUTPUT_TUPLE.append(workflow_tuple['seed'])
                                    seed = workflow_tuple['seed']
                                else:
                                    OUTPUT_TUPLE.append(seed)
                            if 'image_size' in RECYCLER_SETUP:
                                if RECYCLER_SETUP['image_size'] == True and 'width' in workflow_tuple and 'height' in workflow_tuple:
                                    IMG_WIDTH = workflow_tuple['width']
                                    IMG_HEIGHT = workflow_tuple['height']
                            if 'prompt_state' in RECYCLER_SETUP:
                                prompt_state_setup = RECYCLER_SETUP['prompt_state']
        else:
            OUTPUT_TUPLE.append(seed)

        model_version = workflow_tuple['model_version']
        if 'model_shapes' in workflow_tuple and workflow_tuple['model_shapes'] is not None:
            wf_square_shape = workflow_tuple['model_shapes']['SD']
        else:
            wf_square_shape = 768
        match model_version:
            case 'SDXL_2048':
                if 'model_shapes' in workflow_tuple and workflow_tuple['model_shapes'] is not None:
                    wf_square_shape = workflow_tuple['model_shapes']['SDXL']
                else:
                    wf_square_shape = 1024

        if ('model_concept' in workflow_tuple and workflow_tuple['model_concept'] == 'Turbo'):
            if 'model_shapes' in workflow_tuple and workflow_tuple['model_shapes'] is not None:
                wf_square_shape = workflow_tuple['model_shapes']['TURBO']

        if IMG_WIDTH > IMG_HEIGHT:
            orientation = 'Horizontal'
        else:
            orientation = 'Vertical'
        LEGACY_DIMENSIONS = [IMG_WIDTH, IMG_HEIGHT]

        standard_name = 'STANDARD'
        if ('model_concept' in workflow_tuple and workflow_tuple['model_concept'] == 'Cascade'):
            standard_name = 'CASCADE'

        dimensions = utility.get_dimensions_by_shape(self, 'Square [1:1]', wf_square_shape, orientation, False, True, LEGACY_DIMENSIONS[0], LEGACY_DIMENSIONS[1], standard_name)
        LEGACY_DIMENSIONS = dimensions

        WORKFLOWDATA = kwargs['extra_pnginfo']['workflow']['nodes']
        rnd_orientation = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereResolution', 'rnd_orientation', prompt)

        if rnd_orientation == True:
            random.seed(EXT_SEED)
            random.shuffle(LEGACY_DIMENSIONS)
            # if (seed % 2) == 0:
            #    LEGACY_DIMENSIONS = [IMG_HEIGHT, IMG_WIDTH]

        OUTPUT_TUPLE.append(LEGACY_DIMENSIONS[0])
        OUTPUT_TUPLE.append(LEGACY_DIMENSIONS[1])

        workflow_tuple['seed'] = seed
        workflow_tuple['width'] = LEGACY_DIMENSIONS[0]
        workflow_tuple['height'] = LEGACY_DIMENSIONS[1]
        workflow_tuple['size_string'] = str(LEGACY_DIMENSIONS[0]) + 'x' + str(LEGACY_DIMENSIONS[1])

        def DictSort(element):
            if element in utility.WORKFLOW_SORT_LIST:
                return utility.WORKFLOW_SORT_LIST.index(element)
            else:
                return len(utility.WORKFLOW_SORT_LIST)

        if 'seed' in workflow_tuple and 'positive' in workflow_tuple:
            workflow_tuple['decoded_positive'] = utility.DynPromptDecoder(self, workflow_tuple['positive'], workflow_tuple['seed'])
        if 'seed' in workflow_tuple and 'negative' in workflow_tuple:
            workflow_tuple['decoded_negative'] = utility.DynPromptDecoder(self, workflow_tuple['negative'], workflow_tuple['seed'])
        if prompt_state_setup == True and 'decoded_positive' in workflow_tuple:
            workflow_tuple['prompt_state'] = 'Decoded'
        else:
            workflow_tuple['prompt_state'] = 'Dynamic'

        if workflow_tuple is not None and len(workflow_tuple) >= 1:
            workflow_tuple = dict(sorted(workflow_tuple.items(), key=lambda pair: DictSort(pair[0])))

        OUTPUT_TUPLE.append(workflow_tuple)
        return OUTPUT_TUPLE

class PrimereLoraStackMerger:
    RETURN_TYPES = ("LORA_STACK",)
    RETURN_NAMES = ("LORA_STACK",)
    FUNCTION = "lora_stack_merger"
    CATEGORY = TREE_INPUTS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lora_stack_1": ("LORA_STACK",),
                "lora_stack_2": ("LORA_STACK",),
            }
        }

    def lora_stack_merger(self, lora_stack_1, lora_stack_2):
        if lora_stack_1 is not None and lora_stack_2 is not None:
            return (lora_stack_1 + lora_stack_2, )
        else:
            return ([], )

class PrimereLoraKeywordMerger:
    RETURN_TYPES = ("MODEL_KEYWORD",)
    RETURN_NAMES = ("LORA_KEYWORD",)
    FUNCTION = "lora_keyword_merger"
    CATEGORY = TREE_INPUTS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "lora_keyword_SD": ("MODEL_KEYWORD", {"forceInput": True, "default": None}),
                "lora_keyword_SDXL": ("MODEL_KEYWORD", {"forceInput": True, "default": None}),
                "lora_keyword_tagloader": ("MODEL_KEYWORD", {"forceInput": True, "default": None}),
            },
        }

    def lora_keyword_merger(self, lora_keyword_SD = None, lora_keyword_SDXL = None, lora_keyword_tagloader = None):
        model_keyword = [None, None]

        if lora_keyword_SD is not None:
            mkw_list_1 = list(filter(None, lora_keyword_SD))
            if len(mkw_list_1) == 2:
                model_keyword_1 = mkw_list_1[0]
                placement = mkw_list_1[1]
                model_keyword = [model_keyword_1, placement]

        if lora_keyword_SDXL is not None:
            mkw_list_2 = list(filter(None, lora_keyword_SDXL))
            if len(mkw_list_2) == 2:
                model_keyword_2 = mkw_list_2[0]
                placement = mkw_list_2[1]
                model_keyword = [model_keyword_2, placement]

        if lora_keyword_tagloader is not None:
            mkw_list_3 = list(filter(None, lora_keyword_tagloader))
            if len(mkw_list_3) == 2:
                model_keyword_3 = mkw_list_3[0]
                placement = mkw_list_3[1]
                model_keyword = [model_keyword_3, placement]

        return (model_keyword,)

class PrimereEmbeddingKeywordMerger:
    RETURN_TYPES = ("EMBEDDING", "EMBEDDING",)
    RETURN_NAMES = ("EMBEDDING+", "EMBEDDING-")
    FUNCTION = "embedding_keyword_merger"
    CATEGORY = TREE_INPUTS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "embedding_pos_SD": ("EMBEDDING",),
                "embedding_pos_SDXL": ("EMBEDDING",),
                "embedding_neg_SD": ("EMBEDDING",),
                "embedding_neg_SDXL": ("EMBEDDING",),
            },
        }
    def embedding_keyword_merger(self, embedding_pos_SD, embedding_pos_SDXL, embedding_neg_SD, embedding_neg_SDXL):
        embedding_pos = []
        embedding_neg = []

        if embedding_pos_SD is not None:
            mkw_list_1 = list(filter(None, embedding_pos_SD))
            if len(mkw_list_1) == 2:
                model_keyword_1 = mkw_list_1[0]
                placement = mkw_list_1[1]
                embedding_pos.extend([model_keyword_1, placement])

        if embedding_pos_SDXL is not None:
            mkw_list_1 = list(filter(None, embedding_pos_SDXL))
            if len(mkw_list_1) == 2:
                model_keyword_1 = mkw_list_1[0]
                placement = mkw_list_1[1]
                embedding_pos.extend([model_keyword_1, placement])

        if embedding_neg_SD is not None:
            mkw_list_1 = list(filter(None, embedding_neg_SD))
            if len(mkw_list_1) == 2:
                model_keyword_1 = mkw_list_1[0]
                placement = mkw_list_1[1]
                embedding_neg.extend([model_keyword_1, placement])

        if embedding_neg_SDXL is not None:
            mkw_list_1 = list(filter(None, embedding_neg_SDXL))
            if len(mkw_list_1) == 2:
                model_keyword_1 = mkw_list_1[0]
                placement = mkw_list_1[1]
                embedding_neg.extend([model_keyword_1, placement])

        if (len(embedding_pos) == 0):
            embedding_pos = [None, None]
        if (len(embedding_neg) == 0):
            embedding_neg = [None, None]

        return (embedding_pos, embedding_neg,)

class PrimereLycorisStackMerger:
    RETURN_TYPES = ("LYCORIS_STACK",)
    RETURN_NAMES = ("LYCORIS_STACK",)
    FUNCTION = "lycoris_stack_merger"
    CATEGORY = TREE_INPUTS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lycoris_stack_1": ("LYCORIS_STACK",),
                "lycoris_stack_2": ("LYCORIS_STACK",),
            }
        }

    def lycoris_stack_merger(self, lycoris_stack_1, lycoris_stack_2):
        if lycoris_stack_1 is not None and lycoris_stack_2 is not None:
            return (lycoris_stack_1 + lycoris_stack_2, )
        else:
            return ([], )

class PrimereLycorisKeywordMerger:
    RETURN_TYPES = ("MODEL_KEYWORD",)
    RETURN_NAMES = ("LYCORIS_KEYWORD",)
    FUNCTION = "lycoris_keyword_merger"
    CATEGORY = TREE_INPUTS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "lycoris_keyword_SD": ("MODEL_KEYWORD", {"forceInput": True, "default": None}),
                "lycoris_keyword_SDXL": ("MODEL_KEYWORD", {"forceInput": True, "default": None}),
                "lycoris_keyword_tagloader": ("MODEL_KEYWORD", {"forceInput": True, "default": None}),
            },
        }

    def lycoris_keyword_merger(self, lycoris_keyword_SD = None, lycoris_keyword_SDXL = None, lycoris_keyword_tagloader = None):
        model_keyword = [None, None]

        if lycoris_keyword_SD is not None:
            mkw_list_1 = list(filter(None, lycoris_keyword_SD))
            if len(mkw_list_1) == 2:
                model_keyword_1 = mkw_list_1[0]
                placement = mkw_list_1[1]
                model_keyword = [model_keyword_1, placement]

        if lycoris_keyword_SDXL is not None:
            mkw_list_2 = list(filter(None, lycoris_keyword_SDXL))
            if len(mkw_list_2) == 2:
                model_keyword_2 = mkw_list_2[0]
                placement = mkw_list_2[1]
                model_keyword = [model_keyword_2, placement]

        if lycoris_keyword_tagloader is not None:
            mkw_list_3 = list(filter(None, lycoris_keyword_tagloader))
            if len(mkw_list_3) == 2:
                model_keyword_3 = mkw_list_3[0]
                placement = mkw_list_3[1]
                model_keyword = [model_keyword_3, placement]

        return (model_keyword,)

class PrimerePromptOrganizer:
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("PROMPT+", "PROMPT-", "SUBPATH", "MODEL", "ORIENTATION", "PREFERRED")
    FUNCTION = "prompt_organizer_toml"
    CATEGORY = TREE_INPUTS

    @ classmethod
    def INPUT_TYPES(cls):
        DEF_TOML_DIR = os.path.join(PRIMERE_ROOT, 'Toml')
        STYLE_FILE = os.path.join(DEF_TOML_DIR, "prompts.toml")
        STYLE_FILE_EXAMPLE = os.path.join(DEF_TOML_DIR, "prompts.example.toml")

        if Path(STYLE_FILE).is_file() == True:
            STYLE_SOURCE = STYLE_FILE
        else:
            STYLE_SOURCE = STYLE_FILE_EXAMPLE
        STYLE_RESULT = stylehandler.toml2node(STYLE_SOURCE, False, ['preferred_model', 'preferred_orientation'])

        additionalDict = {
            "use_subpath": ("BOOLEAN", {"default": False}),
            "use_model": ("BOOLEAN", {"default": False}),
            "use_orientation": ("BOOLEAN", {"default": False}),
        }

        MERGED_REQ = utility.merge_dict(additionalDict, STYLE_RESULT[0])
        INPUT_DICT_FINAL = {'required': MERGED_REQ}
        cls.STYLE_PROMPTS_POS = STYLE_RESULT[1]
        cls.STYLE_PROMPTS_NEG = STYLE_RESULT[2]
        cls.RAW_STYLE = STYLE_RESULT[3]

        cls.INPUT_DICT_RESULT = INPUT_DICT_FINAL
        return cls.INPUT_DICT_RESULT

    def prompt_organizer_toml(self, opt_pos_style = None, opt_neg_style = None, use_subpath = False, use_model = False, use_orientation = False, **kwargs):
        input_data = kwargs
        original = self
        style_text_result = StyleParser(opt_pos_style, opt_neg_style, input_data, original)

        preferred_subpath = None
        preferred_model = None
        preferred_orientation = None

        if use_subpath == True or use_model == True or use_orientation == True:
            for inputKey, inputValue in input_data.items():
                if inputValue != 'None':
                    DataKey = inputKey.upper()
                    if DataKey in self.RAW_STYLE:
                        DataSection = self.RAW_STYLE[DataKey]
                        ValueList = inputValue.split('::')
                        for DataSectionKey, DataSectionDict in DataSection.items():
                            SectionName = DataSectionDict['Name']
                            if SectionName == ValueList[-1]:
                                if DataSectionDict['preferred_subpath'] != '' and use_subpath == True:
                                    preferred_subpath = DataSectionDict['preferred_subpath']
                                if DataSectionDict['preferred_model'] != '' and use_model == True:
                                    preferred_model = DataSectionDict['preferred_model']
                                if DataSectionDict['preferred_orientation'] != '' and use_orientation == True:
                                    preferred_orientation = DataSectionDict['preferred_orientation']

        preferred = {'subpath': preferred_subpath, 'model': preferred_model, 'orientation': preferred_orientation}

        return (style_text_result[0], style_text_result[1], preferred_subpath, preferred_model, preferred_orientation, preferred)

class PrimerePromptOrganizerCSV:
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("PROMPT+", "PROMPT-", "SUBPATH", "MODEL", "ORIENTATION", "PREFERRED")
    FUNCTION = "prompt_organizer_csv"
    CATEGORY = TREE_INPUTS

    @ classmethod
    def INPUT_TYPES(cls):
        STYLE_DIR = os.path.join(PRIMERE_ROOT, 'stylecsv')
        STYLE_FILE = os.path.join(STYLE_DIR, "styles.csv")
        STYLE_FILE_EXAMPLE = os.path.join(STYLE_DIR, "styles.example.csv")
        if Path(STYLE_FILE).is_file() == True:
            STYLE_SOURCE = STYLE_FILE
        else:
            STYLE_SOURCE = STYLE_FILE_EXAMPLE
        cls.styles_csv = PrimereStyleLoader.load_styles_csv(STYLE_SOURCE)
        STYLE_RESULT = stylehandler.csv2node(cls.styles_csv)

        additionalDict = {
            "use_subpath": ("BOOLEAN", {"default": False}),
            "use_model": ("BOOLEAN", {"default": False}),
            "use_orientation": ("BOOLEAN", {"default": False}),
        }

        MERGED_REQ = utility.merge_dict(additionalDict, STYLE_RESULT)
        INPUT_DICT_FINAL = {'required': MERGED_REQ}
        return INPUT_DICT_FINAL

    def prompt_organizer_csv(self, use_subpath = False, use_model = False, use_orientation = False, **kwargs):
        input_data = kwargs
        styleResult = {}
        styleResult[0] = None
        styleResult[1] = None
        styleResult[2] = None
        styleResult[3] = None
        styleResult[4] = None
        styleResult[5] = None

        for inputKey, inputValue in input_data.items():
            if inputValue != 'None':
                styleResult = PrimereStyleLoader.load_csv(self, inputValue, use_subpath, use_model, use_orientation)
                break

        return (styleResult[0], styleResult[1], styleResult[2], styleResult[3], styleResult[4], styleResult[5])

class PrimereNetworkDataCollector:
    RETURN_TYPES = ("TUPLE",)
    RETURN_NAMES = ("NETWORK_DATA",)
    FUNCTION = "network_tuple_collector"
    CATEGORY = TREE_INPUTS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "optional": {
                "lora_sd": ("LORA_STACK", {"forceInput": True, "default": []}),
                "lora_sdxl": ("LORA_STACK", {"forceInput": True, "default": []}),

                "embedding_sd": ("EMBEDDING_STACK", {"forceInput": True, "default": []}),
                "embedding_sdxl": ("EMBEDDING_STACK", {"forceInput": True, "default": []}),

                "hypernetwork_sd": ("HYPERNETWORK_STACK", {"forceInput": True, "default": []}),
                "hypernetwork_sdxl": ("HYPERNETWORK_STACK", {"forceInput": True, "default": []}),

                "lycoris_sd": ("LYCORIS_STACK", {"forceInput": True, "default": []}),
                "lycoris_sdxl": ("LYCORIS_STACK", {"forceInput": True, "default": []}),
            },
        }

    def network_tuple_collector(self, **kwargs):
        return (kwargs,)

class PrimereMetaTupleCollector:
    RETURN_TYPES = ("TUPLE",)
    RETURN_NAMES = ("FINAL_WORKFLOW_TUPLE",)
    FUNCTION = "meta_tuple_collector"
    CATEGORY = TREE_INPUTS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "workflow_tuple": ("TUPLE", {"forceInput": True, "default": []}),
                "network_data": ("TUPLE", {"forceInput": True, "default": []}),
            },
            "optional": {
                "aesthetic_score": ("INT", {"forceInput": True, "default": 0}),
            },
        }

    def meta_tuple_collector(self, workflow_tuple, network_data, aesthetic_score = 0):
        if (type(aesthetic_score).__name__ == 'int'):
            aesthetic_score = str(aesthetic_score)
        if (not aesthetic_score.isdigit()) or (int(aesthetic_score) < 1):
            aesthetic_score = "*** Aesthetic scorer off ***"

        meta_output = workflow_tuple
        # meta_output["network_data"] = {}
        meta_output["network_data"] = network_data
        meta_output["aesthetic_score"] = aesthetic_score

        return (meta_output,)