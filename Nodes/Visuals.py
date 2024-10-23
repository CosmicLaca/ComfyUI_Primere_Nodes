from ..components.tree import TREE_VISUALS
from ..components.tree import PRIMERE_ROOT
import folder_paths
from ..components import utility
import os
from pathlib import Path
from ..utils import comfy_dir
from .modules import networkhandler
import random
import datetime
from ..Nodes.Inputs import PrimereStyleLoader
from ..components import stylehandler

class PrimereVisualCKPT:
    RETURN_TYPES = ("CHECKPOINT_NAME", "STRING")
    RETURN_NAMES = ("MODEL_NAME", "MODEL_VERSION")
    FUNCTION = "load_ckpt_visual_list"
    CATEGORY = TREE_VISUALS
    allModels = folder_paths.get_filename_list("checkpoints")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model": (cls.allModels,),
                "show_modal": ("BOOLEAN", {"default": True}),
                "show_hidden": ("BOOLEAN", {"default": True}),
                "preview_path": ("BOOLEAN", {"default": True, "label_on": "Primere legacy", "label_off": "Model path"}),
                "random_model": ("BOOLEAN", {"default": False, "label_on": "From selected path", "label_off": "OFF"}),
                "aescore_percent_min": ("INT", {"default": 550, "min": 0, "max": 800, "step": 50}),
                "aescore_percent_max": ("INT", {"default": 800, "min": 200, "max": 1000, "step": 50})
            },
            "optional": {
                "random_seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff}),
            },
            "hidden": {
                "subdir": ("checkpoints",),
                "sortbuttons": (['aScore', 'Name', 'Version', 'Path', 'Date', 'Symlink'],),
                "cache_key": ("model",),
            }
        }

    @classmethod
    def IS_CHANGED(self, **kwargs):
        if kwargs['random_model'] == True:
            return float('NaN')

    def load_ckpt_visual_list(self, base_model, show_hidden, show_modal, preview_path, aescore_percent_min, aescore_percent_max, random_model, random_seed = 0):
        def new_state_random():
            random.seed(datetime.datetime.now().timestamp())
            return random.randint(10, 0xffffffffffffffff)

        if random_model == True:
            fullSource = self.allModels
            slashIndex = base_model.find('\\')
            if slashIndex > 0:
                subdirType = base_model[0: slashIndex] + '\\'
                models_by_path = list(filter(lambda x: x.startswith(subdirType), fullSource))
                if random_seed is None or int(random_seed) <= 0:
                    random_seed = int(new_state_random())
                random.seed(random_seed)
                base_model = random.choice(models_by_path)

        modelname_only = Path(base_model).stem
        model_version = utility.get_value_from_cache('model_version', modelname_only)
        if model_version is None:
            model_version = utility.getModelType(base_model, 'checkpoints')
            utility.add_value_to_cache('model_version', modelname_only, model_version)

        return (base_model, model_version)

class PrimereVisualLORA:
    RETURN_TYPES = ("MODEL", "CLIP", "LORA_STACK", "MODEL_KEYWORD")
    RETURN_NAMES = ("MODEL", "CLIP", "LORA_STACK", "LORA_KEYWORD")
    FUNCTION = "visual_lora_stacker"
    CATEGORY = TREE_VISUALS
    LORASCOUNT = 6

    @classmethod
    def INPUT_TYPES(cls):
        LoraList = folder_paths.get_filename_list("loras")

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "model_version": ("STRING", {"default": 'SD1', "forceInput": True}),

                "stack_version": (["Any", "Auto"] + utility.SUPPORTED_MODELS, {"default": "Auto"}),
                "show_modal": ("BOOLEAN", {"default": True}),
                "show_hidden": ("BOOLEAN", {"default": True}),
                "auto_filter": ("BOOLEAN", {"default": False, "label_on": "Filter by version", "label_off": "OFF"}),
                "preview_path": ("BOOLEAN", {"default": True, "label_on": "Primere legacy", "label_off": "Model path"}),
                "randomize": ("BOOLEAN", {"default": False, "label_on": "One random input", "label_off": "OFF"}),
                "use_only_model_weight": ("BOOLEAN", {"default": True}),

                "use_lora_1": ("BOOLEAN", {"default": False}),
                "lora_1": (LoraList,),
                "lora_1_model_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_1_clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "use_lora_2": ("BOOLEAN", {"default": False}),
                "lora_2": (LoraList,),
                "lora_2_model_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_2_clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "use_lora_3": ("BOOLEAN", {"default": False}),
                "lora_3": (LoraList,),
                "lora_3_model_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_3_clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "use_lora_4": ("BOOLEAN", {"default": False}),
                "lora_4": (LoraList,),
                "lora_4_model_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_4_clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "use_lora_5": ("BOOLEAN", {"default": False}),
                "lora_5": (LoraList,),
                "lora_5_model_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_5_clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "use_lora_6": ("BOOLEAN", {"default": False}),
                "lora_6": (LoraList,),
                "lora_6_model_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lora_6_clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "use_lora_keyword": ("BOOLEAN", {"default": False}),
                "lora_keyword_placement": (["First", "Last"], {"default": "Last"}),
                "lora_keyword_selection": (["Select in order", "Random select"], {"default": "Select in order"}),
                "lora_keywords_num": ("INT", {"default": 1, "min": 1, "max": 50, "step": 1}),
                "lora_keyword_weight": ("FLOAT", {"default": 1.0, "min": 0, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "random_seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff}),
                "workflow_tuple": ("TUPLE", {"forceInput": True, "default": {}})
            },
            "hidden": {
                "subdir": ("loras",),
                "sortbuttons": (['Name', 'Version', 'Path', 'Date'],),
                "cache_key": ("lora",),
                "version_filter_input": ("stack_version",),
            }
        }

    def visual_lora_stacker(self, model, clip, use_only_model_weight, use_lora_keyword, lora_keyword_placement, lora_keyword_selection, lora_keywords_num, lora_keyword_weight,
                            workflow_tuple=None, stack_version ='Any', model_version ="SD1", **kwargs):

        model_keyword = [None, None]

        if workflow_tuple is not None and 'model_concept' in workflow_tuple and workflow_tuple['model_concept'] != stack_version and workflow_tuple['model_concept'] != 'Normal':
            return (model, clip, [], model_keyword)

        if model_version == 'SDXL_2048' and stack_version == 'SD':
            return (model, clip, [], model_keyword)

        if model_version != 'SDXL_2048' and stack_version == 'SDXL':
            return (model, clip, [], model_keyword)

        if workflow_tuple is not None and len(workflow_tuple) > 0 and 'setup_states' in workflow_tuple and 'lora_setup' in workflow_tuple['setup_states'] and 'exif_status' in workflow_tuple and workflow_tuple['exif_status'] == 'SUCCEED':
            if workflow_tuple['setup_states']['lora_setup'] == True:
                if 'network_data' in workflow_tuple:
                    loader = networkhandler.getNetworkLoader(workflow_tuple, 'lora', self.LORASCOUNT, True, stack_version)
                    if len(loader) > 0:
                        return networkhandler.LoraHandler(self, loader, model, clip, model_keyword, use_only_model_weight, lora_keywords_num, use_lora_keyword, lora_keyword_selection, lora_keyword_weight, lora_keyword_placement)
                    else:
                        return (model, clip, [], model_keyword)
                else:
                    return (model, clip, [], model_keyword)
            else:
                return (model, clip, [], model_keyword)

        return networkhandler.LoraHandler(self, kwargs, model, clip, model_keyword, use_only_model_weight, lora_keywords_num, use_lora_keyword, lora_keyword_selection, lora_keyword_weight, lora_keyword_placement)

class PrimereVisualEmbedding:
    RETURN_TYPES = ("EMBEDDING", "EMBEDDING", "EMBEDDING_STACK")
    RETURN_NAMES = ("EMBEDDING+", "EMBEDDING-", "EMBEDDING_STACK")
    FUNCTION = "primere_visual_embedding"
    CATEGORY = TREE_VISUALS
    EMBCOUNT = 6

    @classmethod
    def INPUT_TYPES(cls):
        EmbeddingList = folder_paths.get_filename_list("embeddings")
        return {
            "required": {
                "model_version": ("STRING", {"default": 'SD1', "forceInput": True}),
                "stack_version": (["Any", "Auto"] + utility.SUPPORTED_MODELS, {"default": "Auto"}),

                "show_modal": ("BOOLEAN", {"default": True}),
                "show_hidden": ("BOOLEAN", {"default": True}),
                "auto_filter": ("BOOLEAN", {"default": False, "label_on": "Filter by version", "label_off": "OFF"}),
                "preview_path": ("BOOLEAN", {"default": True, "label_on": "Primere legacy", "label_off": "Model path"}),
                "randomize": ("BOOLEAN", {"default": False, "label_on": "One random input", "label_off": "OFF"}),

                "use_embedding_1": ("BOOLEAN", {"default": False}),
                "embedding_1": (EmbeddingList,),
                "embedding_1_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, },),
                "is_negative_1": ("BOOLEAN", {"default": False}),

                "use_embedding_2": ("BOOLEAN", {"default": False}),
                "embedding_2": (EmbeddingList,),
                "embedding_2_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, },),
                "is_negative_2": ("BOOLEAN", {"default": False}),

                "use_embedding_3": ("BOOLEAN", {"default": False}),
                "embedding_3": (EmbeddingList,),
                "embedding_3_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, },),
                "is_negative_3": ("BOOLEAN", {"default": False}),

                "use_embedding_4": ("BOOLEAN", {"default": False}),
                "embedding_4": (EmbeddingList,),
                "embedding_4_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, },),
                "is_negative_4": ("BOOLEAN", {"default": False}),

                "use_embedding_5": ("BOOLEAN", {"default": False}),
                "embedding_5": (EmbeddingList,),
                "embedding_5_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, },),
                "is_negative_5": ("BOOLEAN", {"default": False}),

                "use_embedding_6": ("BOOLEAN", {"default": False}),
                "embedding_6": (EmbeddingList,),
                "embedding_6_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01, },),
                "is_negative_6": ("BOOLEAN", {"default": False}),

                "embedding_placement_pos": (["First", "Last"], {"default": "Last"}),
                "embedding_placement_neg": (["First", "Last"], {"default": "Last"}),
            },
            "optional": {
                "random_seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff}),
                "workflow_tuple": ("TUPLE", {"forceInput": True, "default": {}})
            },
            "hidden": {
                "subdir": ("embeddings",),
                "sortbuttons": (['Name', 'Version', 'Path', 'Date'],),
                "cache_key": ("embedding",),
                "version_filter_input": ("stack_version",),
            }
        }

    def primere_visual_embedding(self, embedding_placement_pos, embedding_placement_neg, workflow_tuple=None, stack_version='Any', model_version="SD1", **kwargs):
        if workflow_tuple is not None and 'model_concept' in workflow_tuple and workflow_tuple['model_concept'] != stack_version and workflow_tuple['model_concept'] != 'Normal':
            return ([None, None], [None, None], [])

        if model_version == 'SDXL_2048' and stack_version == 'SD':
            return ([None, None], [None, None], [])

        if model_version != 'SDXL_2048' and stack_version == 'SDXL':
            return ([None, None], [None, None], [])

        if workflow_tuple is not None and len(workflow_tuple) > 0 and 'setup_states' in workflow_tuple and 'embedding_setup' in workflow_tuple['setup_states'] and 'exif_status' in workflow_tuple and workflow_tuple['exif_status'] == 'SUCCEED':
            if workflow_tuple['setup_states']['embedding_setup'] == True:
                if 'network_data' in workflow_tuple:
                    loader = networkhandler.getNetworkLoader(workflow_tuple, 'embedding', self.EMBCOUNT, False, stack_version)
                    if len(loader) > 0:
                        return networkhandler.EmbeddingHandler(self, loader, embedding_placement_pos, embedding_placement_neg)
                    else:
                        return ([None, None], [None, None], [])
                else:
                    return ([None, None], [None, None], [])
            else:
                return ([None, None], [None, None], [])

        return networkhandler.EmbeddingHandler(self, kwargs, embedding_placement_pos, embedding_placement_neg)

class PrimereVisualHypernetwork:
    RETURN_TYPES = ("MODEL", "HYPERNETWORK_STACK")
    RETURN_NAMES = ("MODEL", "HYPERNETWORK_STACK")
    FUNCTION = "visual_hypernetwork"
    CATEGORY = TREE_VISUALS
    HNCOUNT = 6

    @classmethod
    def INPUT_TYPES(s):
        HypernetworkList = folder_paths.get_filename_list("hypernetworks")

        return {
            "required": {
                "model": ("MODEL",),
                "safe_load": ("BOOLEAN", {"default": True}),

                "show_modal": ("BOOLEAN", {"default": True}),
                "show_hidden": ("BOOLEAN", {"default": True}),
                "preview_path": ("BOOLEAN", {"default": True, "label_on": "Primere legacy", "label_off": "Model path"}),
                "randomize": ("BOOLEAN", {"default": False, "label_on": "One random input", "label_off": "OFF"}),

                "use_hypernetwork_1": ("BOOLEAN", {"default": False}),
                "hypernetwork_1": (HypernetworkList, ),
                "hypernetwork_1_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "use_hypernetwork_2": ("BOOLEAN", {"default": False}),
                "hypernetwork_2": (HypernetworkList,),
                "hypernetwork_2_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "use_hypernetwork_3": ("BOOLEAN", {"default": False}),
                "hypernetwork_3": (HypernetworkList,),
                "hypernetwork_3_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "use_hypernetwork_4": ("BOOLEAN", {"default": False}),
                "hypernetwork_4": (HypernetworkList,),
                "hypernetwork_4_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "use_hypernetwork_5": ("BOOLEAN", {"default": False}),
                "hypernetwork_5": (HypernetworkList,),
                "hypernetwork_5_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "use_hypernetwork_6": ("BOOLEAN", {"default": False}),
                "hypernetwork_6": (HypernetworkList,),
                "hypernetwork_6_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
            },
            "optional": {
                "random_seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff}),
                "workflow_tuple": ("TUPLE", {"forceInput": True, "default": {}})
            },
            "hidden": {
                "subdir": ("hypernetworks",),
                "sortbuttons": (['Name', 'Path', 'Date'],),
                "cache_key": ("embedding",),
            }
        }

    def visual_hypernetwork(self, model, model_version, workflow_tuple = None, stack_version = "Any", safe_load = True, **kwargs):
        if workflow_tuple is not None and 'model_concept' in workflow_tuple and workflow_tuple['model_concept'] != stack_version and workflow_tuple['model_concept'] != 'Normal':
            return (model, [],)

        if model_version == 'SDXL_2048' and stack_version == 'SD':
            return (model, [],)

        if model_version != 'SDXL_2048' and stack_version == 'SDXL':
            return (model, [],)

        if workflow_tuple is not None and len(workflow_tuple) > 0 and 'setup_states' in workflow_tuple and 'hypernetwork_setup' in workflow_tuple['setup_states'] and 'exif_status' in workflow_tuple and workflow_tuple['exif_status'] == 'SUCCEED':
            if workflow_tuple['setup_states']['hypernetwork_setup'] == True:
                loader = networkhandler.getNetworkLoader(workflow_tuple, 'hypernetwork', self.HNCOUNT, False, stack_version)
                if len(loader) > 0:
                    return networkhandler.HypernetworkHandler(self, loader, model, safe_load)
                else:
                    return (model, [],)
            else:
                return (model, [],)

        return networkhandler.HypernetworkHandler(self, kwargs, model, safe_load)

class PrimereVisualStyle:
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("PROMPT+", "PROMPT-", "SUBPATH", "MODEL", "ORIENTATION", "PREFERRED")
    FUNCTION = "load_visual_csv"
    CATEGORY = TREE_VISUALS

    @classmethod
    def INPUT_TYPES(cls):
        STYLE_DIR = os.path.join(PRIMERE_ROOT, 'stylecsv')
        STYLE_FILE = os.path.join(STYLE_DIR, "styles.csv")
        STYLE_FILE_EXAMPLE = os.path.join(STYLE_DIR, "styles.example.csv")
        if Path(STYLE_FILE).is_file() == True:
            STYLE_SOURCE = STYLE_FILE
        else:
            STYLE_SOURCE = STYLE_FILE_EXAMPLE
        cls.styles_csv = PrimereStyleLoader.load_styles_csv(STYLE_SOURCE)

        return {
            "required": {
                "styles": (sorted(list(cls.styles_csv['name'])),),
                "show_modal": ("BOOLEAN", {"default": True}),
                "show_hidden": ("BOOLEAN", {"default": True}),
                "use_subpath": ("BOOLEAN", {"default": False}),
                "use_model": ("BOOLEAN", {"default": False}),
                "use_orientation": ("BOOLEAN", {"default": False}),
                "random_prompt": ("BOOLEAN", {"default": False, "label_on": "From preferred path", "label_off": "OFF"}),
                "aescore_percent_min": ("INT", {"default": 550, "min": 0, "max": 800, "step": 50}),
                "aescore_percent_max": ("INT", {"default": 800, "min": 200, "max": 1000, "step": 50})
            },
            "optional": {
                "random_seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff}),
            },
            "hidden": {
                "subdir": ("styles",),
                "sortbuttons": (['aScore', 'Name', 'Path'],),
                "cache_key": ("styles",),
            }
        }

    @classmethod
    def IS_CHANGED(self, **kwargs):
        if kwargs['random_prompt'] == True:
            return float('NaN')

    def load_visual_csv(self, styles, show_modal, show_hidden, use_subpath, use_model, use_orientation, aescore_percent_min, aescore_percent_max, random_prompt, random_seed = 0):
        def new_state_random():
            random.seed(datetime.datetime.now().timestamp())
            return random.randint(10, 0xffffffffffffffff)

        styleKey = self.styles_csv['name'] == styles

        try:
            preferred_subpath = self.styles_csv[styleKey]['preferred_subpath'].values[0]
        except Exception:
            preferred_subpath = ''

        if random_prompt == True:
            if str(preferred_subpath) == "nan":
                resultsBySubpath = self.styles_csv[self.styles_csv['preferred_subpath'].isnull()]
            else:
                resultsBySubpath = self.styles_csv[self.styles_csv['preferred_subpath'] == preferred_subpath]

            if random_seed is None or int(random_seed) <= 0:
                random_seed = int(new_state_random())
            random.seed(random_seed)
            random_stylename = random.choice(list(resultsBySubpath['name']))
            styleKey = self.styles_csv['name'] == random_stylename

        else:
            styleKey = self.styles_csv['name'] == styles

        try:
            positive_prompt = self.styles_csv[styleKey]['prompt'].values[0]
        except Exception:
            positive_prompt = ''

        try:
            negative_prompt = self.styles_csv[styleKey]['negative_prompt'].values[0]
        except Exception:
            negative_prompt = ''

        try:
            preferred_model = self.styles_csv[styleKey]['preferred_model'].values[0]
        except Exception:
            preferred_model = ''

        try:
            preferred_orientation = self.styles_csv[styleKey]['preferred_orientation'].values[0]
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

class PrimereVisualPromptOrganizerCSV:
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("PROMPT+", "PROMPT-", "SUBPATH", "MODEL", "ORIENTATION", "PREFERRED")
    FUNCTION = "prompt_visual_organizer_csv"
    CATEGORY = TREE_VISUALS

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
            "show_modal": ("BOOLEAN", {"default": True}),
            "show_hidden": ("BOOLEAN", {"default": True}),
            "use_subpath": ("BOOLEAN", {"default": False}),
            "use_model": ("BOOLEAN", {"default": False}),
            "use_orientation": ("BOOLEAN", {"default": False}),
            "random_prompt": ("BOOLEAN", {"default": False, "label_on": "From preferred path", "label_off": "OFF"})
        }

        hiddenDict = {
            "subdir": ("styles",),
            "sortbuttons": (['aScore', 'Name'],),
            "cache_key": ("styles",)
        }

        optionalDict = {
            "random_seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff})
        }

        MERGED_REQ = utility.merge_dict(additionalDict, STYLE_RESULT)
        INPUT_DICT_FINAL = {'required': MERGED_REQ, 'optional': optionalDict, 'hidden': hiddenDict}
        return INPUT_DICT_FINAL

    def prompt_visual_organizer_csv(self, show_modal, show_hidden, random_prompt, use_subpath = False, use_model = False, use_orientation = False, random_seed = 0, **kwargs):
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

class PrimereVisualLYCORIS:
    RETURN_TYPES = ("MODEL", "CLIP", "LYCORIS_STACK", "MODEL_KEYWORD")
    RETURN_NAMES = ("MODEL", "CLIP", "LYCORIS_STACK", "LYCORIS_KEYWORD")
    FUNCTION = "primere_visual_lycoris_stacker"
    CATEGORY = TREE_VISUALS
    LYCOSCOUNT = 6

    @classmethod
    def INPUT_TYPES(cls):
        LYCO_DIR = os.path.join(comfy_dir, 'models', 'lycoris')
        folder_paths.add_model_folder_path("lycoris", LYCO_DIR)
        LyCORIS = folder_paths.get_filename_list("lycoris")
        LyCORISList = folder_paths.filter_files_extensions(LyCORIS, ['.ckpt', '.safetensors'])

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "model_version": ("STRING", {"default": 'SD1', "forceInput": True}),

                "stack_version": (["Any", "Auto"] + utility.SUPPORTED_MODELS, {"default": "Auto"}),
                "show_modal": ("BOOLEAN", {"default": True}),
                "show_hidden": ("BOOLEAN", {"default": True}),
                "auto_filter": ("BOOLEAN", {"default": False, "label_on": "Filter by version", "label_off": "OFF"}),
                "preview_path": ("BOOLEAN", {"default": True, "label_on": "Primere legacy", "label_off": "Model path"}),
                "randomize": ("BOOLEAN", {"default": False, "label_on": "One random input", "label_off": "OFF"}),
                "use_only_model_weight": ("BOOLEAN", {"default": True}),

                "use_lycoris_1": ("BOOLEAN", {"default": False}),
                "lycoris_1": (LyCORISList,),
                "lycoris_1_model_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lycoris_1_clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "use_lycoris_2": ("BOOLEAN", {"default": False}),
                "lycoris_2": (LyCORISList,),
                "lycoris_2_model_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lycoris_2_clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "use_lycoris_3": ("BOOLEAN", {"default": False}),
                "lycoris_3": (LyCORISList,),
                "lycoris_3_model_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lycoris_3_clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "use_lycoris_4": ("BOOLEAN", {"default": False}),
                "lycoris_4": (LyCORISList,),
                "lycoris_4_model_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lycoris_4_clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "use_lycoris_5": ("BOOLEAN", {"default": False}),
                "lycoris_5": (LyCORISList,),
                "lycoris_5_model_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lycoris_5_clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "use_lycoris_6": ("BOOLEAN", {"default": False}),
                "lycoris_6": (LyCORISList,),
                "lycoris_6_model_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
                "lycoris_6_clip_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

                "use_lycoris_keyword": ("BOOLEAN", {"default": False}),
                "lycoris_keyword_placement": (["First", "Last"], {"default": "Last"}),
                "lycoris_keyword_selection": (["Select in order", "Random select"], {"default": "Select in order"}),
                "lycoris_keywords_num": ("INT", {"default": 1, "min": 1, "max": 50, "step": 1}),
                "lycoris_keyword_weight": ("FLOAT", {"default": 1.0, "min": 0, "max": 10.0, "step": 0.1}),
            },
            "optional": {
                "random_seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff}),
                "workflow_tuple": ("TUPLE", {"forceInput": True, "default": {}})
            },
            "hidden": {
                "subdir": ("lycoris",),
                "sortbuttons": (['Name', 'Version', 'Path', 'Date'],),
                "cache_key": ("lycoris",),
                "version_filter_input": ("stack_version",),
            }
        }

    def primere_visual_lycoris_stacker(self, model, clip, use_only_model_weight, use_lycoris_keyword, lycoris_keyword_placement, lycoris_keyword_selection, lycoris_keywords_num, lycoris_keyword_weight, workflow_tuple=None, stack_version = 'Any', model_version = "SD1", **kwargs):
        model_keyword = [None, None]

        if workflow_tuple is not None and 'model_concept' in workflow_tuple and workflow_tuple['model_concept'] != stack_version and workflow_tuple['model_concept'] != 'Normal':
            return (model, clip, [], model_keyword)

        if model_version == 'SDXL_2048' and stack_version == 'SD':
            return (model, clip, [], model_keyword)

        if model_version != 'SDXL_2048' and stack_version == 'SDXL':
            return (model, clip, [], model_keyword)

        if workflow_tuple is not None and len(workflow_tuple) > 0 and 'setup_states' in workflow_tuple and 'lycoris_setup' in workflow_tuple['setup_states'] and 'exif_status' in workflow_tuple and workflow_tuple['exif_status'] == 'SUCCEED':
            if workflow_tuple['setup_states']['lycoris_setup'] == True:
                if 'network_data' in workflow_tuple:
                    loader = networkhandler.getNetworkLoader(workflow_tuple, 'lycoris', self.LYCOSCOUNT, True, stack_version)
                    if len(loader) > 0:
                        return networkhandler.LycorisHandler(self, loader, model, clip, model_keyword, use_only_model_weight, lycoris_keywords_num, use_lycoris_keyword, lycoris_keyword_selection, lycoris_keyword_weight, lycoris_keyword_placement)
                    else:
                        return (model, clip, [], model_keyword)
                else:
                    return (model, clip, [], model_keyword)
            else:
                return (model, clip, [], model_keyword)

        return networkhandler.LycorisHandler(self, kwargs, model, clip, model_keyword, use_only_model_weight, lycoris_keywords_num, use_lycoris_keyword, lycoris_keyword_selection, lycoris_keyword_weight, lycoris_keyword_placement)