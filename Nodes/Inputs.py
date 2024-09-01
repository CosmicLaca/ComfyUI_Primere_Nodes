from ..components.tree import TREE_INPUTS
from ..components.tree import PRIMERE_ROOT
from ..components.tree import TREE_DEPRECATED
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
from .modules import exif_data_checker
from ..components import utility
from pathlib import Path
import random
import string
from .modules.adv_encode import advanced_encode
from ..components import stylehandler
from .Styles import StyleParser
import nodes
from .modules.exif_data_checker import check_model_from_exif

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
    RETURN_TYPES = ("STRING", "STRING", "CONDITIONING", "CONDITIONING", "TUPLE")
    RETURN_NAMES = ("PROMPT+", "PROMPT-", "COND+", "COND-", "PROMPT_DATA")
    FUNCTION = "refiner_prompt"
    CATEGORY = TREE_INPUTS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "positive_refiner": ("STRING", {"default": "", "multiline": True}),
                "negative_refiner": ("STRING", {"default": "", "multiline": True}),
                "positive_refiner_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "negative_refiner_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "positive_original_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "negative_original_strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.01}),
                "clip": ("CLIP",),
                "seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff, "forceInput": True}),
                "token_normalization": (["none", "mean", "length", "length+mean"],),
                "weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),
            },
            "optional": {
                "positive_original": ("STRING", {"forceInput": True}),
                "negative_original": ("STRING", {"forceInput": True}),
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

    def refiner_prompt(self, extra_pnginfo, id, clip, seed, token_normalization, weight_interpretation, positive_refiner = "", negative_refiner = "", positive_original = None, negative_original = None, positive_refiner_strength = 1, negative_refiner_strength = 1, positive_original_strength = 1, negative_original_strength = 1):
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

        output_positive = rawResult[0].replace('\n', ' ')
        output_negative = rawResult[1].replace('\n', ' ')
        final_positive = ""
        final_negative = ""

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

        final_positive = utility.DynPromptDecoder(self, final_positive.strip(' ,;'), seed)
        final_negative = utility.DynPromptDecoder(self, final_negative.strip(' ,;'), seed)

        try:
            embeddings_final_pos, pooled_pos = advanced_encode(clip, final_positive, token_normalization, weight_interpretation, w_max=1.0, apply_to_pooled=True)
            embeddings_final_neg, pooled_neg = advanced_encode(clip, final_negative, token_normalization, weight_interpretation, w_max=1.0, apply_to_pooled=True)
        except Exception:
            tokens = clip.tokenize(final_positive)
            embeddings_final_pos, pooled_pos = clip.encode_from_tokens(tokens, return_pooled = True)

            tokens = clip.tokenize(final_negative)
            embeddings_final_neg, pooled_neg = clip.encode_from_tokens(tokens, return_pooled = True)

        prompt_tuple = {}
        prompt_tuple['final_positive'] = final_positive
        prompt_tuple['final_negative'] = final_negative
        prompt_tuple['clip'] = clip
        prompt_tuple['token_normalization'] = token_normalization
        prompt_tuple['weight_interpretation'] = weight_interpretation
        prompt_tuple['cond_positive'] = [[embeddings_final_pos, {"pooled_output": pooled_pos}]]
        prompt_tuple['cond_negative'] = [[embeddings_final_neg, {"pooled_output": pooled_neg}]]

        return final_positive, final_negative, [[embeddings_final_pos, {"pooled_output": pooled_pos}]], [[embeddings_final_neg, {"pooled_output": pooled_neg}]], prompt_tuple

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
                "seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff, "forceInput": True}),
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

class PrimereVAESelector:
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("VAE",)
    FUNCTION = "primere_vae_selector"
    CATEGORY = TREE_INPUTS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "vae_sd": ("VAE",),
                "vae_sdxl": ("VAE",),
                "vae_cascade": ("VAE",),
                "model_version": ("STRING", {"default": 'BaseModel_1024', "forceInput": True}),
                "model_concept": ("STRING", {"default": "Normal", "forceInput": True}),
            }
        }

    def primere_vae_selector(self, vae_sd, vae_sdxl, vae_cascade, model_version = "BaseModel_1024", model_concept = 'Normal'):
        match model_concept:
            case 'Cascade':
                return (vae_cascade,)
        match model_version:
            case 'SDXL_2048':
                return (vae_sdxl,)
        return (vae_sd,)

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

            if 'preferred' in workflow_tuple:
                prefred_settings = workflow_tuple['preferred']
                if len(prefred_settings) > 0:
                    for prefkey, prefval in prefred_settings.items():
                        if prefval is not None:
                            match prefkey:
                                case "model":
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
                    if model_version is None:
                        checkpointpaths = folder_paths.get_folder_paths("checkpoints")[0]
                        model_full_path = checkpointpaths + os.sep + workflow_tuple['model']
                        model_file = Path(model_full_path)
                        if model_file.is_file() == True:
                            LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, workflow_tuple['model'])
                            model_version = utility.getCheckpointVersion(LOADED_CHECKPOINT[0])
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
                    original_exif = readerResult.original
                    exif_data_count = len(workflow_tuple)
                    workflow_tuple['meta_source'] = reader.__class__.__name__
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

                is_sdxl = 0
                if 'model' in workflow_tuple:
                    modelname_only = Path((workflow_tuple['model'])).stem
                    model_version = utility.get_value_from_cache('model_version', modelname_only)
                    if model_version is None:
                        checkpointpaths = folder_paths.get_folder_paths("checkpoints")[0]
                        model_full_path = checkpointpaths + os.sep + workflow_tuple['model']
                        model_file = Path(model_full_path)
                        if model_file.is_file() == True:
                            LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, workflow_tuple['model'])
                            model_version = utility.getCheckpointVersion(LOADED_CHECKPOINT[0])
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
                                    LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, workflow_tuple['model'])
                                    model_version = utility.getCheckpointVersion(LOADED_CHECKPOINT[0])
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
            workflow_tuple['seed'] = random.randint(1, 0xffffffffffffffff)

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
                "seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff, "forceInput": True}),
                "width": ('INT', {"forceInput": True, "default": 512}),
                "height": ('INT', {"forceInput": True, "default": 512}),
                # "rnd_orientation": ("BOOLEAN", {"default": False}),

                "workflow_tuple": ("TUPLE", {"default": None}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "id": "UNIQUE_ID",
            },

        }
    def expand_meta_2(self, workflow_tuple, seed, width, height, **kwargs):
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
        rnd_orientation = utility.getDataFromWorkflow(WORKFLOWDATA, 'PrimereResolution', 4)

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


class PrimereMetaRead:
    CATEGORY = TREE_DEPRECATED
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "CHECKPOINT_NAME", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT", "INT", "INT", "FLOAT", "INT", "VAE_NAME", "VAE", "CLIP", "MODEL", "TUPLE")
    RETURN_NAMES = ("PROMPT+", "PROMPT-", "PROMPT L+", "PROMPT L-", "REFINER+", "REFINER-", "MODEL_NAME", "SAMPLER_NAME", "SCHEDULER_NAME", "SEED", "WIDTH", "HEIGHT", "CFG", "STEPS", "VAE_NAME", "VAE",  "CLIP", "MODEL", "METADATA")
    FUNCTION = "load_image_meta"

    def __init__(self):
        wildcard_dir = os.path.join(PRIMERE_ROOT, 'wildcards')
        self._wildcard_manager = WildcardManager(wildcard_dir)
        self._parser_config = ParserConfig(
            variant_start = "{",
            variant_end = "}",
            wildcard_wrap = "__"
        )
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        input_dir = folder_paths.get_input_directory()
        files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]

        return {
            "required": {
                "use_exif": ("BOOLEAN", {"default": True}),
                "use_decoded_dyn": ("BOOLEAN", {"default": False}),
                "use_model": ("BOOLEAN", {"default": True}),
                "model_hash_check": ("BOOLEAN", {"default": False}),
                "use_sampler": ("BOOLEAN", {"default": True}),
                "use_seed": ("BOOLEAN", {"default": True}),
                "use_size": ("BOOLEAN", {"default": True}),
                "recount_size": ("BOOLEAN", {"default": False}),
                "use_cfg_scale": ("BOOLEAN", {"default": True}),
                "use_steps": ("BOOLEAN", {"default": True}),
                "use_exif_vae": ("BOOLEAN", {"default": True}),
                "force_model_vae": ("BOOLEAN", {"default": False}),
                "image": (sorted(files),),
            },
            "optional": {
                "positive": ('STRING', {"forceInput": True, "default": ""}),
                "negative": ('STRING', {"forceInput": True, "default": ""}),
                "positive_l": ('STRING', {"forceInput": True, "default": ""}),
                "negative_l": ('STRING', {"forceInput": True, "default": ""}),
                "positive_r": ('STRING', {"forceInput": True, "default": ""}),
                "negative_r": ('STRING', {"forceInput": True, "default": ""}),
                "model_name": ('CHECKPOINT_NAME', {"forceInput": True, "default": ""}),
                "model_version": ("STRING", {"default": 'BaseModel_1024', "forceInput": True}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"forceInput": True, "default": "euler"}),
                "scheduler_name": (comfy.samplers.KSampler.SCHEDULERS, {"forceInput": True, "default": "normal"}),
                "seed": ('INT', {"forceInput": True, "default": 1}),
                "width": ('INT', {"forceInput": True, "default": 512}),
                "height": ('INT', {"forceInput": True, "default": 512}),
                "cfg_scale": ('FLOAT', {"forceInput": True, "default": 7}),
                "steps": ('INT', {"forceInput": True, "default": 12}),
                "vae_name_sd": ('VAE_NAME', {"forceInput": True, "default": ""}),
                "vae_name_sdxl": ('VAE_NAME', {"forceInput": True, "default": ""}),
                "model_concept": ("STRING", {"default": "Normal", "forceInput": True}),
                "concept_data": ("TUPLE", {"default": None, "forceInput": True}),
                "preferred_model": ("STRING", {"default": "", "forceInput": True}),
                "preferred_orientation": ("STRING", {"default": "", "forceInput": True}),
            },
        }

    def load_image_meta(self, use_exif, use_decoded_dyn, use_model, model_hash_check, use_sampler, use_seed, use_size, recount_size, use_cfg_scale, use_steps, use_exif_vae, force_model_vae, image,
                        positive="", negative="", positive_l="", negative_l="", positive_r="", negative_r="",
                        model_hash="", model_name="", model_version="BaseModel_1024", sampler_name="euler", scheduler_name="normal", seed=1, width=512, height=512, cfg_scale=7, steps=12, vae_name_sd="", vae_name_sdxl="", model_concept="Normal", concept_data = None, preferred_model="", preferred_orientation=""):

        if preferred_orientation == 'Random':
            if (seed % 2) == 0:
                preferred_orientation = "Horizontal"
            else:
                preferred_orientation = "Vertical"

        data_json = {}
        data_json['positive'] = positive.replace('ADDROW ', '').replace('ADDCOL ', '').replace('ADDCOMM ', '').replace('\n', ' ')
        data_json['negative'] = negative.replace('\n', ' ')
        data_json['positive_l'] = positive_l
        data_json['negative_l'] = negative_l
        data_json['positive_r'] = positive_r
        data_json['negative_r'] = negative_r
        data_json['model_hash'] = model_hash
        data_json['model_name'] = model_name
        data_json['sampler_name'] = sampler_name
        data_json['scheduler_name'] = scheduler_name
        data_json['seed'] = seed
        data_json['width'] = width
        data_json['height'] = height
        data_json['cfg_scale'] = cfg_scale
        data_json['steps'] = steps
        data_json['model_version'] = model_version
        data_json['model_concept'] = model_concept
        data_json['vae_name'] = vae_name_sd
        data_json['force_model_vae'] = force_model_vae
        data_json['preferred_model'] = preferred_model
        data_json['preferred_orientation'] = preferred_orientation
        LOADED_CHECKPOINT = []

        is_sdxl = 0
        match model_version:
            case 'SDXL_2048':
                is_sdxl = 1
        data_json['is_sdxl'] = is_sdxl

        if (is_sdxl == 1):
            data_json['vae_name'] = vae_name_sdxl
        else:
            data_json['vae_name'] = vae_name_sd

        if (data_json['vae_name'] == ""):
            data_json['vae_name'] = folder_paths.get_filename_list("vae")[0]

        if use_exif:
            image_path = folder_paths.get_annotated_filepath(image)
            if os.path.isfile(image_path):
                readerResult = ImageExifReader(image_path)

                if (type(readerResult.parser).__name__ == 'dict'):
                    print('Reader tool return empty, using node input')
                    if (force_model_vae == True):
                        LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, model_name)
                        realvae = LOADED_CHECKPOINT[2]
                    else:
                        LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, data_json['model_name'])
                        realvae = LOADED_CHECKPOINT[2]

                    return (positive, negative, positive_l, negative_l, positive_r, negative_r, model_name, sampler_name, scheduler_name, seed, width, height, cfg_scale, steps, data_json['vae_name'], realvae, LOADED_CHECKPOINT[1], LOADED_CHECKPOINT[0], data_json)

                reader = readerResult.parser

                if 'positive' in reader.parameter:
                    data_json['positive'] = reader.parameter["positive"].replace('ADDROW ', '').replace('ADDCOL ', '').replace('ADDCOMM ', '').replace('\n', ' ')
                else:
                    data_json['positive'] = ""

                if 'negative' in reader.parameter:
                    data_json['negative'] = reader.parameter["negative"].replace('\n', ' ')
                else:
                    data_json['negative'] = ""

                data_json['dynamic_positive'] = utility.DynPromptDecoder(self, data_json['positive'], seed)
                data_json['dynamic_negative'] = utility.DynPromptDecoder(self, data_json['negative'], seed)

                if (readerResult.tool == ''):
                    print('Reader tool return empty, using node input')
                    if (force_model_vae == True):
                        LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, model_name)
                        realvae = LOADED_CHECKPOINT[2]
                    else:
                        LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, data_json['model_name'])
                        realvae = LOADED_CHECKPOINT[2]

                    return (positive, negative, positive_l, negative_l, positive_r, negative_r, model_name, sampler_name, scheduler_name, seed, width, height, cfg_scale, steps, data_json['vae_name'], realvae, LOADED_CHECKPOINT[1], LOADED_CHECKPOINT[0], data_json)

                try:
                    if use_model == True:
                        if 'model_hash' in reader.parameter:
                            data_json['model_hash'] = reader.parameter["model_hash"]
                        else:
                            checkpointpaths = folder_paths.get_folder_paths("checkpoints")[0]
                            model_full_path = checkpointpaths + os.sep + model_name
                            if os.path.isfile(model_full_path):
                                data_json['model_hash'] = exif_data_checker.get_model_hash(model_full_path)
                            else:
                                data_json['model_hash'] = 'no_hash_data'

                        if 'model_name' in reader.parameter:
                            model_name_exif = reader.parameter["model_name"]
                            data_json['model_name'] = exif_data_checker.check_model_from_exif(data_json['model_hash'], model_name_exif, model_name, model_hash_check)
                        else:
                            data_json['model_name'] = folder_paths.get_filename_list("checkpoints")[0]

                    if (data_json['model_name'] != model_name):
                        is_sdxl = 0
                        modelname_only = Path((data_json['model_name'])).stem
                        model_version = utility.get_value_from_cache('model_version', modelname_only)
                        if model_version is None:
                            LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, data_json['model_name'])
                            model_version = utility.getCheckpointVersion(LOADED_CHECKPOINT[0])
                            utility.add_value_to_cache('model_version', modelname_only, model_version)

                        data_json['model_version'] = model_version
                        match model_version:
                            case 'SDXL_2048':
                                is_sdxl = 1
                        data_json['is_sdxl'] = is_sdxl

                    if use_sampler == True and data_json['model_concept'] == 'Normal' and (reader.parameter["cfg_scale"] >= 3 and reader.parameter["steps"] >= 9):
                        if 'sampler' in reader.parameter:
                            sampler_name_exif = reader.parameter["sampler"]
                            samplers = exif_data_checker.check_sampler_from_exif(sampler_name_exif.lower(), sampler_name, scheduler_name)
                            data_json['sampler_name'] = samplers['sampler']
                            data_json['scheduler_name'] = samplers['scheduler']
                        elif ('sampler_name' in reader.parameter and 'scheduler_name' in reader.parameter):
                            data_json['sampler_name'] = reader.parameter["sampler_name"]
                            data_json['scheduler_name'] = reader.parameter["scheduler_name"]

                    if use_seed == True:
                        if 'seed' in reader.parameter:
                            data_json['seed'] = reader.parameter["seed"]

                    if use_cfg_scale == True and data_json['model_concept'] == 'Normal' and reader.parameter["cfg_scale"] >= 3:
                        if 'cfg_scale' in reader.parameter:
                            data_json['cfg_scale'] = reader.parameter["cfg_scale"]

                    if use_steps == True and data_json['model_concept'] == 'Normal' and reader.parameter["steps"] >= 9:
                        if 'steps' in reader.parameter:
                            data_json['steps'] = reader.parameter["steps"]

                    if (is_sdxl == 1):
                        data_json['vae_name'] = vae_name_sdxl
                    else:
                        data_json['vae_name'] = vae_name_sd

                    if (data_json['vae_name'] == ""):
                        data_json['vae_name'] = folder_paths.get_filename_list("vae")[0]

                    if force_model_vae == True:
                        if len(LOADED_CHECKPOINT) == 3:
                            realvae = LOADED_CHECKPOINT[2]
                        else:
                            LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, data_json['model_name'])
                            realvae = LOADED_CHECKPOINT[2]
                    else:
                        if use_exif_vae == True:
                            if 'vae' in reader.parameter:
                                vae_name_exif = reader.parameter["vae"]
                                vae = exif_data_checker.check_vae_exif(vae_name_exif.lower(), data_json['vae_name'])
                                data_json['vae_name'] = vae

                        realvae = nodes.VAELoader.load_vae(self, data_json['vae_name'])[0]

                    if use_size == True:
                        if 'size_string' in reader.parameter or ('width' in reader.parameter and 'height' in reader.parameter):
                            data_json['width'] = reader.parameter["width"]
                            data_json['height'] = reader.parameter["height"]
                        if recount_size == True:
                            if (data_json['width'] > data_json['height']):
                                orientation = 'Horizontal'
                            else:
                                orientation = 'Vertical'

                            wf_square_shape = utility.get_square_shape(data_json['width'], data_json['height'])
                            image_sides = sorted([data_json['width'], data_json['height']])
                            custom_side_b = round((image_sides[1] / image_sides[0]), 4)
                            # dimensions = utility.calculate_dimensions(self, "Square [1:1]", orientation, True, model_version, True, 1, custom_side_b)
                            dimensions = utility.get_dimensions_by_shape(self, 'Square [1:1]', wf_square_shape, orientation, True, True, 1, custom_side_b, 'STANDARD')
                            data_json['width'] = dimensions[0]
                            data_json['height'] = dimensions[1]

                    if use_decoded_dyn == True:
                        if 'dynamic_positive' in reader.parameter:
                            data_json['positive'] = reader.parameter['dynamic_positive']
                            data_json['dynamic_positive'] = reader.parameter['dynamic_positive']
                        if 'dynamic_negative' in reader.parameter:
                            data_json['negative'] = reader.parameter['dynamic_negative']
                            data_json['dynamic_negative'] = reader.parameter['dynamic_negative']

                    if len(LOADED_CHECKPOINT) != 3:
                        LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, data_json['model_name'])

                    return (data_json['positive'], data_json['negative'], data_json['positive_l'], data_json['negative_l'], data_json['positive_r'], data_json['negative_r'], data_json['model_name'], data_json['sampler_name'], data_json['scheduler_name'], data_json['seed'], data_json['width'], data_json['height'], data_json['cfg_scale'], data_json['steps'], data_json['vae_name'], realvae, LOADED_CHECKPOINT[1], LOADED_CHECKPOINT[0], data_json)

                except ValueError as VE:
                    print(VE)
                    if (force_model_vae == True):
                        if len(LOADED_CHECKPOINT) == 3:
                            realvae = LOADED_CHECKPOINT[2]
                        else:
                            LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, data_json['model_name'])
                            realvae = LOADED_CHECKPOINT[2]
                    else:
                        realvae = nodes.VAELoader.load_vae(self, data_json['vae_name'])[0]

                    return (data_json['positive'], data_json['negative'], data_json['positive_l'], data_json['negative_l'], data_json['positive_r'], data_json['negative_r'], data_json['model_name'], data_json['sampler_name'], data_json['scheduler_name'], data_json['seed'], data_json['width'], data_json['height'], data_json['cfg_scale'], data_json['steps'], data_json['vae_name'], realvae, LOADED_CHECKPOINT[1], LOADED_CHECKPOINT[0], data_json)

            else:
                print('No source image loaded')
                if (force_model_vae == True):
                    if len(LOADED_CHECKPOINT) == 3:
                        realvae = LOADED_CHECKPOINT[2]
                    else:
                        LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, data_json['model_name'])
                        realvae = LOADED_CHECKPOINT[2]
                else:
                    realvae = nodes.VAELoader.load_vae(self, data_json['vae_name'])[0]

                return (data_json['positive'], data_json['negative'], data_json['positive_l'], data_json['negative_l'], data_json['positive_r'], data_json['negative_r'], data_json['model_name'], data_json['sampler_name'], data_json['scheduler_name'], data_json['seed'], data_json['width'], data_json['height'], data_json['cfg_scale'], data_json['steps'], data_json['vae_name'], realvae, LOADED_CHECKPOINT[1], LOADED_CHECKPOINT[0], data_json)

        else:
            print('Exif reader off')
            if preferred_model is not None and len(preferred_model.strip()) > 0:
                data_json['model_name'] = exif_data_checker.check_model_from_exif("no_hash_data", preferred_model, preferred_model, False)

                is_sdxl = 0
                modelname_only = Path(data_json['model_name']).stem
                model_version = utility.get_value_from_cache('model_version', modelname_only)
                if model_version is None:
                    LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, data_json['model_name'])
                    model_version = utility.getCheckpointVersion(LOADED_CHECKPOINT[0])
                    utility.add_value_to_cache('model_version', modelname_only, model_version)

                data_json['model_version'] = model_version
                match model_version:
                    case 'SDXL_2048':
                        is_sdxl = 1
                data_json['is_sdxl'] = is_sdxl

            if (force_model_vae == True):
                if len(LOADED_CHECKPOINT) == 3:
                    realvae = LOADED_CHECKPOINT[2]
                else:
                    LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, data_json['model_name'])
                    realvae = LOADED_CHECKPOINT[2]
            else:
                if len(LOADED_CHECKPOINT) != 3:
                    LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, data_json['model_name'])

                if (is_sdxl == 1):
                    data_json['vae_name'] = vae_name_sdxl
                else:
                    data_json['vae_name'] = vae_name_sd
                realvae = nodes.VAELoader.load_vae(self, data_json['vae_name'])[0]

            data_json['dynamic_positive'] = utility.DynPromptDecoder(self, data_json['positive'], seed)
            data_json['dynamic_negative'] = utility.DynPromptDecoder(self, data_json['negative'], seed)

            if preferred_orientation is not None and len(preferred_orientation.strip()) > 0:
                if preferred_orientation == 'Vertical' and (data_json['width'] > data_json['height']):
                    data_json['width'] = height
                    data_json['height'] = width
                if preferred_orientation == 'Horizontal' and (data_json['height'] > data_json['width']):
                    data_json['width'] = height
                    data_json['height'] = width

            LOADED_MODEL = LOADED_CHECKPOINT[0]

            if model_concept == 'Lightning':
                lightning_selector = 'SAFETENSOR'
                lightning_model_step = 8

                if concept_data is not None:
                    if 'lightning_selector' in concept_data:
                        lightning_selector = concept_data['lightning_selector']
                    if 'lightning_model_step' in concept_data:
                        lightning_model_step = concept_data['lightning_model_step']

                ModelConceptChanges = utility.ModelConceptNames(data_json['model_name'], model_concept, lightning_selector, lightning_model_step)
                data_json['model_name'] = ModelConceptChanges['ckpt_name']
                lora_name = ModelConceptChanges['lora_name']
                unet_name = ModelConceptChanges['unet_name']
                lightningModeValid = ModelConceptChanges['lightningModeValid']

                is_sdxl = 0
                modelname_only = Path(data_json['model_name']).stem
                model_version = utility.get_value_from_cache('model_version', modelname_only)
                if model_version is None:
                    LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, data_json['model_name'])
                    model_version = utility.getCheckpointVersion(LOADED_CHECKPOINT[0])
                    utility.add_value_to_cache('model_version', modelname_only, model_version)

                data_json['model_version'] = model_version
                match model_version:
                    case 'SDXL_2048':
                        is_sdxl = 1
                data_json['is_sdxl'] = is_sdxl

                if lightningModeValid == True:
                    LOADED_MODEL = utility.LightningConceptModel(self, model_concept, lightningModeValid, lightning_selector, lightning_model_step, LOADED_CHECKPOINT[0], lora_name, unet_name)

            return (data_json['positive'], data_json['negative'], data_json['positive_l'], data_json['negative_l'], data_json['positive_r'], data_json['negative_r'], data_json['model_name'], data_json['sampler_name'], data_json['scheduler_name'], data_json['seed'], data_json['width'], data_json['height'], data_json['cfg_scale'], data_json['steps'], data_json['vae_name'], realvae, LOADED_CHECKPOINT[1], LOADED_MODEL, data_json)

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
    FUNCTION = "prompt_organizer"
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

    def prompt_organizer(self, opt_pos_style = None, opt_neg_style = None, use_subpath = False, use_model = False, use_orientation = False, **kwargs):
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