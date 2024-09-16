import nodes
from ..components.tree import TREE_VISUALS
from ..components.tree import PRIMERE_ROOT
import folder_paths
from ..components import utility
import os
from pathlib import Path
import chardet
import pandas
import re
from ..utils import comfy_dir
from .modules import networkhandler

class PrimereVisualCKPT:
    RETURN_TYPES = ("CHECKPOINT_NAME", "STRING")
    RETURN_NAMES = ("MODEL_NAME", "MODEL_VERSION")
    FUNCTION = "load_ckpt_visual_list"
    CATEGORY = TREE_VISUALS
    model_versions = utility.get_category_from_cache('model_version')

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_model": (folder_paths.get_filename_list("checkpoints"),),
                "show_modal": ("BOOLEAN", {"default": True}),
                "show_hidden": ("BOOLEAN", {"default": True}),
            },
            "hidden": {
                "cached_model": (cls.model_versions,),
            }
        }

    def load_ckpt_visual_list(self, base_model, show_hidden, show_modal):
        modelname_only = Path(base_model).stem
        model_version = utility.get_value_from_cache('model_version', modelname_only)
        if model_version is None:
            LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, base_model)
            model_version = utility.getCheckpointVersion(LOADED_CHECKPOINT[0])
            utility.add_value_to_cache('model_version', modelname_only, model_version)

        return (base_model, model_version)

class PrimereVisualLORA:
    RETURN_TYPES = ("MODEL", "CLIP", "LORA_STACK", "MODEL_KEYWORD")
    RETURN_NAMES = ("MODEL", "CLIP", "LORA_STACK", "LORA_KEYWORD")
    FUNCTION = "visual_lora_stacker"
    CATEGORY = TREE_VISUALS
    LORASCOUNT = 6

    lora_versions = utility.get_category_from_cache('lora_version')

    @classmethod
    def INPUT_TYPES(cls):
        LoraList = folder_paths.get_filename_list("loras")

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "model_version": ("STRING", {"default": 'BaseModel_1024', "forceInput": True}),

                "stack_version": (["SD", "SDXL", "Flux", "Any"], {"default": "Any"}),
                "show_modal": ("BOOLEAN", {"default": True}),
                "show_hidden": ("BOOLEAN", {"default": True}),
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
                "workflow_tuple": ("TUPLE", {"forceInput": True, "default": []}),
            },
            "hidden": {
                "cached_lora": (cls.lora_versions,),
            }
        }

    def visual_lora_stacker(self, model, clip, use_only_model_weight, use_lora_keyword, lora_keyword_placement, lora_keyword_selection, lora_keywords_num, lora_keyword_weight, workflow_tuple, stack_version = 'Any', model_version = "BaseModel_1024", **kwargs):
        model_keyword = [None, None]
        if 'model_concept' in workflow_tuple and workflow_tuple['model_concept'] != stack_version and workflow_tuple['model_concept'] != 'Normal':
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

    embedding_versions = utility.get_category_from_cache('embedding_version')

    @classmethod
    def INPUT_TYPES(cls):
        EmbeddingList = folder_paths.get_filename_list("embeddings")
        return {
            "required": {
                "model_version": ("STRING", {"default": 'BaseModel_1024', "forceInput": True}),
                "stack_version": (["SD", "SDXL", "Flux", "Any"], {"default": "Any"}),

                "show_modal": ("BOOLEAN", {"default": True}),
                "show_hidden": ("BOOLEAN", {"default": True}),

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
                "workflow_tuple": ("TUPLE", {"forceInput": True, "default": []}),
            },
            "hidden": {
                "cached_embedding": (cls.embedding_versions,),
            }
        }

    def primere_visual_embedding(self, embedding_placement_pos, embedding_placement_neg, workflow_tuple, stack_version='Any', model_version="BaseModel_1024", **kwargs):
        if 'model_concept' in workflow_tuple and workflow_tuple['model_concept'] != stack_version and workflow_tuple['model_concept'] != 'Normal':
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
    EMBCOUNT = 6

    @classmethod
    def INPUT_TYPES(s):
        HypernetworkList = folder_paths.get_filename_list("hypernetworks")

        return {
            "required": {
                "model": ("MODEL",),
                "model_version": ("STRING", {"default": 'BaseModel_1024', "forceInput": True}),
                "safe_load": ("BOOLEAN", {"default": True}),
                "stack_version": (["SD", "SDXL", "Flux", "Any"], {"default": "Any"}),

                "show_modal": ("BOOLEAN", {"default": True}),
                "show_hidden": ("BOOLEAN", {"default": True}),

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
                "workflow_tuple": ("TUPLE", {"forceInput": True, "default": []}),
            },
        }

    def visual_hypernetwork(self, model, model_version, workflow_tuple, stack_version = "Any", safe_load = True, **kwargs):
        if 'model_concept' in workflow_tuple and workflow_tuple['model_concept'] != stack_version and workflow_tuple['model_concept'] != 'Normal':
            return (model, [],)

        if model_version == 'SDXL_2048' and stack_version == 'SD':
            return (model, [],)

        if model_version != 'SDXL_2048' and stack_version == 'SDXL':
            return (model, [],)

        if workflow_tuple is not None and len(workflow_tuple) > 0 and 'setup_states' in workflow_tuple and 'hypernetwork_setup' in workflow_tuple['setup_states'] and 'exif_status' in workflow_tuple and workflow_tuple['exif_status'] == 'SUCCEED':
            if workflow_tuple['setup_states']['hypernetwork_setup'] == True:
                loader = networkhandler.getNetworkLoader(workflow_tuple, 'hypernetwork', self.EMBCOUNT, False, stack_version)
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

        try:
            cls.styles_csv = cls.load_styles_csv(os.path.join(STYLE_DIR, "styles.csv"))
        except Exception:
            cls.styles_csv = cls.load_styles_csv(os.path.join(STYLE_DIR, "styles.example.csv"))

        return {
            "required": {
                "styles": (sorted(list(cls.styles_csv['name'])),),
                "show_modal": ("BOOLEAN", {"default": True}),
                "show_hidden": ("BOOLEAN", {"default": True}),
                "use_subpath": ("BOOLEAN", {"default": False}),
                "use_model": ("BOOLEAN", {"default": False}),
                "use_orientation": ("BOOLEAN", {"default": False}),
            },
        }

    def load_visual_csv(self, styles, show_modal, show_hidden, use_subpath, use_model, use_orientation):
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

class PrimereVisualLYCORIS:
    RETURN_TYPES = ("MODEL", "CLIP", "LYCORIS_STACK", "MODEL_KEYWORD")
    RETURN_NAMES = ("MODEL", "CLIP", "LYCORIS_STACK", "LYCORIS_KEYWORD")
    FUNCTION = "primere_visual_lycoris_stacker"
    CATEGORY = TREE_VISUALS
    LYCOSCOUNT = 6

    lyco_versions = utility.get_category_from_cache('lycoris_version')

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
                "model_version": ("STRING", {"default": 'BaseModel_1024', "forceInput": True}),

                "stack_version": (["SD", "SDXL", "Flux", "Any"], {"default": "Any"}),
                "show_modal": ("BOOLEAN", {"default": True}),
                "show_hidden": ("BOOLEAN", {"default": True}),
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
                "workflow_tuple": ("TUPLE", {"forceInput": True, "default": []}),
            },
            "hidden": {
                "cached_lyco": (cls.lyco_versions,),
            }
        }

    def primere_visual_lycoris_stacker(self, model, clip, use_only_model_weight, use_lycoris_keyword, lycoris_keyword_placement, lycoris_keyword_selection, lycoris_keywords_num, lycoris_keyword_weight, workflow_tuple, stack_version = 'Any', model_version = "BaseModel_1024", **kwargs):
        model_keyword = [None, None]

        if 'model_concept' in workflow_tuple and workflow_tuple['model_concept'] != stack_version and workflow_tuple['model_concept'] != 'Normal':
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