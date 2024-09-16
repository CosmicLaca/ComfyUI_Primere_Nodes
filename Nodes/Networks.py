from ..components.tree import TREE_NETWORKS
import folder_paths
import os
from ..utils import comfy_dir
from .modules import networkhandler

class PrimereLORA:
    RETURN_TYPES = ("MODEL", "CLIP", "LORA_STACK", "MODEL_KEYWORD")
    RETURN_NAMES = ("MODEL", "CLIP", "LORA_STACK", "LORA_KEYWORD")
    FUNCTION = "primere_lora_stacker"
    CATEGORY = TREE_NETWORKS
    LORASCOUNT = 6

    @classmethod
    def INPUT_TYPES(cls):
        LoraList = folder_paths.get_filename_list("loras")

        return {
            "required": {
                "model": ("MODEL",),
                "clip": ("CLIP",),
                "model_version": ("STRING", {"default": 'BaseModel_1024', "forceInput": True}),
                "stack_version": (["SD", "SDXL", "Flux", "Any"], {"default": "Any"}),
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
        }

    def primere_lora_stacker(self, model, clip, use_only_model_weight, use_lora_keyword, lora_keyword_placement, lora_keyword_selection, lora_keywords_num, lora_keyword_weight, workflow_tuple, stack_version = 'Any', model_version = "BaseModel_1024", **kwargs):
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

class PrimereEmbedding:
    RETURN_TYPES = ("EMBEDDING", "EMBEDDING", "EMBEDDING_STACK")
    RETURN_NAMES = ("EMBEDDING+", "EMBEDDING-", "EMBEDDING_STACK")
    FUNCTION = "primere_embedding"
    CATEGORY = TREE_NETWORKS
    EMBCOUNT = 6

    @classmethod
    def INPUT_TYPES(self):
        EmbeddingList = folder_paths.get_filename_list("embeddings")

        return {
            "required": {
                "model_version": ("STRING", {"default": 'BaseModel_1024', "forceInput": True}),
                "stack_version": (["SD", "SDXL", "Flux", "Any"], {"default": "Any"}),

                "use_embedding_1": ("BOOLEAN", {"default": False}),
                "embedding_1": (EmbeddingList,),
                "embedding_1_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01,},),
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
        }

    def primere_embedding(self, embedding_placement_pos, embedding_placement_neg, workflow_tuple, stack_version = 'Any', model_version = "BaseModel_1024", **kwargs):
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

class PrimereHypernetwork:
    RETURN_TYPES = ("MODEL", "HYPERNETWORK_STACK")
    RETURN_NAMES = ("MODEL", "HYPERNETWORK_STACK")
    FUNCTION = "primere_hypernetwork"
    CATEGORY = TREE_NETWORKS
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

    def primere_hypernetwork(self, model, model_version, workflow_tuple, stack_version = 'Any', safe_load = True, **kwargs):
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

class PrimereLYCORIS:
    RETURN_TYPES = ("MODEL", "CLIP", "LYCORIS_STACK", "MODEL_KEYWORD")
    RETURN_NAMES = ("MODEL", "CLIP", "LYCORIS_STACK", "LYCORIS_KEYWORD")
    FUNCTION = "primere_lycoris_stacker"
    CATEGORY = TREE_NETWORKS
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
                "model_version": ("STRING", {"default": 'BaseModel_1024', "forceInput": True}),
                "stack_version": (["SD", "SDXL", "Flux", "Any"], {"default": "Any"}),
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
        }

    def primere_lycoris_stacker(self, model, clip, use_only_model_weight, use_lycoris_keyword, lycoris_keyword_placement, lycoris_keyword_selection, lycoris_keywords_num, workflow_tuple, lycoris_keyword_weight, stack_version = 'Any', model_version = "BaseModel_1024", **kwargs):
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