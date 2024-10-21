import math
from ..components.tree import TREE_DASHBOARD
from ..components.tree import PRIMERE_ROOT
from ..components.tree import TREE_DEPRECATED
import comfy.samplers
import folder_paths
import nodes
import torch
import torch.nn.functional as F
from .modules.latent_noise import PowerLawNoise
import random
import os
import tomli
from .modules.adv_encode import advanced_encode
from nodes import MAX_RESOLUTION
from ..components import utility
from pathlib import Path
import re
import requests
from ..components import hypernetwork
from ..components import clipping
import comfy.sd
import comfy.model_detection
import comfy.utils
from ..utils import comfy_dir
from ..utils import here
import comfy_extras.nodes_model_advanced as nodes_model_advanced
import comfy_extras.nodes_upscale_model as nodes_upscale_model
from comfy import model_management
from datetime import datetime
from ..components.gguf import nodes as gguf_nodes
import comfy_extras.nodes_flux as nodes_flux
import comfy_extras.nodes_sd3 as nodes_sd3
from .modules import long_clip
from .modules import networkhandler
from .Networks import PrimereLORA
from .Networks import PrimereEmbedding
from .Networks import PrimereHypernetwork
from .Networks import PrimereLYCORIS
from diffusers import (UNet2DConditionModel, DPMSolverMultistepScheduler,  EulerDiscreteScheduler, EulerAncestralDiscreteScheduler, DEISMultistepScheduler, UniPCMultistepScheduler)
from ..components.kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from ..components.kolors.models.tokenization_chatglm import ChatGLMTokenizer
from ..components.kolors.models.modeling_chatglm import ChatGLMModel, ChatGLMConfig
import gc
from ..components.hunyuan.conf import hydit_conf
from ..components.hunyuan.loader import load_hydit
from ..components.hunyuan.utils.dtype import string_to_dtype
from ..components.hunyuan.tenc import load_clip, load_t5

class PrimereSamplersSteps:
    CATEGORY = TREE_DASHBOARD
    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT", "FLOAT")
    RETURN_NAMES = ("SAMPLER_NAME", "SCHEDULER_NAME", "STEPS", "CFG")
    FUNCTION = "get_sampler_step"

    kolors_schedulers = ["EulerDiscreteScheduler", "EulerAncestralDiscreteScheduler", "DPMSolverMultistepScheduler", "DPMSolverMultistepScheduler_SDE_karras", "UniPCMultistepScheduler", "DEISMultistepScheduler"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler_name": (cls.kolors_schedulers + comfy.samplers.KSampler.SCHEDULERS,),
                "steps": ("INT", {"default": 12, "min": 1, "max": 1000, "step": 1}),
                "cfg": ("FLOAT", {"default": 7, "min": 0.1, "max": 100, "step": 0.01}),
            }
        }

    def get_sampler_step(self, sampler_name, scheduler_name, steps = 12, cfg = 7):
        return sampler_name, scheduler_name, steps, round(cfg, 2)

class PrimereVAE:
    RETURN_TYPES = ("VAE_NAME",)
    RETURN_NAMES = ("VAE_NAME",)
    FUNCTION = "load_vae_list"
    CATEGORY = TREE_DASHBOARD

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_model": (folder_paths.get_filename_list("vae"),)
            },
        }

    def load_vae_list(self, vae_model):
        return vae_model,

class PrimereCKPT:
    RETURN_TYPES = ("CHECKPOINT_NAME", "STRING", ['None'] + folder_paths.get_filename_list("checkpoints"),)
    RETURN_NAMES = ("MODEL_NAME", "MODEL_VERSION", "MODEL")
    FUNCTION = "load_ckpt_list"
    CATEGORY = TREE_DASHBOARD

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "base_model": (folder_paths.get_filename_list("checkpoints"),),
            },
        }

    def load_ckpt_list(self, base_model):
        model_version = utility.getModelType(base_model, 'checkpoints')
        return (base_model, model_version, base_model)

class PrimereVAELoader:
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("VAE",)
    FUNCTION = "load_primere_vae"
    CATEGORY = TREE_DASHBOARD

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": ("VAE_NAME",),
                "baked_vae": ("VAE",)
            },
        }

    def load_primere_vae(self, vae_name, baked_vae,):
        if (vae_name == 'Baked VAE'):
            return (baked_vae,)

        if (vae_name == 'External VAE'):
            vae_name = folder_paths.get_filename_list("vae")[0]

        return nodes.VAELoader.load_vae(self, vae_name)

class PrimereModelConceptSelector:
    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT", "FLOAT",
                    "STRING", "CLIP_SELECTION", "VAE_SELECTION", "VAE_NAME",
                    "FLOAT",
                    "STRING", "INT", "FLOAT",
                    "STRING", "STRING", "STRING", "STRING",
                    "STRING", "INT", "FLOAT",
                    "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "FLOAT", "STRING",  "STRING",
                    "FLUX_HYPER_LORA", "STRING", "INT", "FLOAT",
                    "STRING", "STRING", "STRING",
                    "STRING", "STRING", "STRING", "STRING", "SD3_HYPER_LORA", "INT", "FLOAT",
                    "STRING"
                    )
    RETURN_NAMES = ("SAMPLER_NAME", "SCHEDULER_NAME", "STEPS", "CFG",
                    "MODEL_CONCEPT", "CLIP_SELECTION", "VAE_SELECTION", "VAE_NAME",
                    "STRENGTH_LCM_LORA_MODEL",
                    "LIGHTNING_SELECTOR", "LIGHTNING_MODEL_STEP", "STRENGTH_LIGHTNING_LORA_MODEL",
                    "CASCADE_STAGE_A", "CASCADE_STAGE_B", "CASCADE_STAGE_C", "CASCADE_CLIP",
                    "HYPER-SD_SELECTOR", "HYPER-SD_MODEL_STEP", "STRENGTH_HYPERSD_LORA_MODEL",
                    "FLUX_SELECTOR", "FLUX_DIFFUSION_MODEL", "FLUX_WEIGHT_TYPE", "FLUX_GGUF_MODEL", "FLUX_CLIP_T5XXL", "FLUX_CLIP_L", "FLUX_CLIP_GUIDANCE", "FLUX_VAE", "FLUX_SAMPLER",
                    "USE_FLUX_HYPER_LORA", "FLUX_HYPER_LORA_TYPE", "FLUX_HYPER_LORA_STEP", "FLUX_HYPER_LORA_STRENGTH",
                    "HUNYUAN_CLIP_T5XXL", "HUNYUAN_CLIP_L", "HUNYUAN_VAE",
                    "SD3_CLIP_G", "SD3_CLIP_L", "SD3_CLIP_T5XXL", "SD3_UNET_VAE", "USE_SD3_HYPER_LORA", "SD3_HYPER_LORA_STEP", "SD3_HYPER_LORA_STRENGTH",
                    "KOLORS_PRECISION"
                    )
    FUNCTION = "select_model_concept"
    CATEGORY = TREE_DASHBOARD

    UNETLIST = folder_paths.get_filename_list("unet")
    DIFFUSIONLIST = folder_paths.get_filename_list("diffusion_models")
    GGUFLIST = folder_paths.get_filename_list("unet_gguf")
    VAELIST = folder_paths.get_filename_list("vae")
    CLIPLIST = folder_paths.get_filename_list("clip")
    CLIPLIST += folder_paths.get_filename_list("clip_gguf")
    CLIPLIST += folder_paths.get_filename_list("t5")
    CONCEPT_LIST = utility.SUPPORTED_MODELS[0:14]

    SAMPLER_INPUTS = {'model_version': ("STRING", {"forceInput": True, "default": "SD1"})}

    for concept in CONCEPT_LIST:
        concept_string = concept.lower()
        SAMPLER_INPUTS[concept_string + '_sampler_name'] = (comfy.samplers.KSampler.SAMPLERS, {"forceInput": True, "default": "euler"})
        SAMPLER_INPUTS[concept_string + '_scheduler_name'] = (comfy.samplers.KSampler.SCHEDULERS, {"forceInput": True, "default": "normal"})
        SAMPLER_INPUTS[concept_string + '_steps'] = ('INT', {"forceInput": True, "default": 12})
        SAMPLER_INPUTS[concept_string + '_cfg_scale'] = ('FLOAT', {"forceInput": True, "default": 7})

    INPUT_DICT = {
        "required": {
            "default_sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"default": "euler"}),
            "default_scheduler_name": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
            "default_cfg_scale": ("FLOAT", {"default": 7, "min": 0.1, "max": 100, "step": 0.01}),
            "default_steps": ("INT", {"default": 12, "min": 1, "max": 1000, "step": 1}),

            "sd_vae": (["None"] + VAELIST,),
            "sdxl_vae": (["None"] + VAELIST,),

            "model_concept": (["Auto"] + CONCEPT_LIST, {"default": "Auto"}),
            "clip_selection": ("BOOLEAN", {"default": True, "label_on": "Use baked if exist", "label_off": "Always use custom"}),
            "vae_selection": ("BOOLEAN", {"default": True, "label_on": "Use baked if exist", "label_off": "Always use custom"}),

            "strength_lcm_lora_model": ("FLOAT", {"default": 1.000, "min": -20.000, "max": 20.000, "step": 0.001}),

            "lightning_selector": (["UNET", "LORA", "SAFETENSOR"], {"default": "LORA"}),
            "lightning_model_step": ([1, 2, 4, 8], {"default": 8}),
            "lightning_sampler": ("BOOLEAN", {"default": False, "label_on": "Set by model", "label_off": "Custom (external)"}),
            "strength_lightning_lora_model": ("FLOAT", {"default": 1.000, "min": -20.000, "max": 20.000, "step": 0.001}),

            "hypersd_selector": (["UNET", "LORA"], {"default": "LORA"}),
            "hypersd_model_step": ([1, 2, 4, 8, 12], {"default": 8}),
            "hypersd_sampler": ("BOOLEAN", {"default": False, "label_on": "Set by model", "label_off": "Custom (external)"}),
            "strength_hypersd_lora_model": ("FLOAT", {"default": 1.000, "min": -20.000, "max": 20.000, "step": 0.001}),

            "cascade_stage_a": (["None"] + VAELIST,),
            "cascade_stage_b": (["None"] + UNETLIST,),
            "cascade_stage_c": (["None"] + UNETLIST,),
            "cascade_clip": (["None"] + CLIPLIST,),

            # "playground_sigma_max": ("FLOAT", {"default": 120.0, "min": 0.0, "max": 1000.0, "step": 0.001, "round": False}),
            # "playground_sigma_min": ("FLOAT", {"default": 0.002, "min": 0.0, "max": 1000.0, "step": 0.001, "round": False}),

            "flux_selector": (["DIFFUSION", "GGUF", "SAFETENSOR"], {"default": "DIFFUSION"}),
            "flux_diffusion": (["None"] + DIFFUSIONLIST,),
            "flux_weight_dtype": (["default", "fp8_e4m3fn", "fp8_e5m2"],),
            "flux_gguf": (["None"] + GGUFLIST,),
            "flux_clip_t5xxl": (["None"] + CLIPLIST,),
            "flux_clip_l": (["None"] + CLIPLIST,),
            "flux_clip_guidance": ('FLOAT', {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
            "flux_vae": (["None"] + VAELIST,),
            "flux_sampler": (["custom_advanced", "ksampler"], {"default": "ksampler"}),
            "use_flux_hyper_lora": ("BOOLEAN", {"default": False, "label_on": "Use hyper Lora", "label_off": "Ignore Lora"}),
            "flux_hyper_lora_type": (["FLUX.1-dev", "FLUX.1-dev-fp16"], {"default": "FLUX.1-dev"}),
            "flux_hyper_lora_step": ([8, 16], {"default": 16}),
            "flux_hyper_lora_strength": ("FLOAT", {"default": 0.125, "min": -20.000, "max": 20.000, "step": 0.001}),

            "hunyuan_clip_t5xxl": (["None"] + CLIPLIST,),
            "hunyuan_clip_l": (["None"] + CLIPLIST,),
            "hunyuan_vae": (["None"] + VAELIST,),

            "sd3_clip_g": (["None"] + CLIPLIST,),
            "sd3_clip_l": (["None"] + CLIPLIST,),
            "sd3_clip_t5xxl": (["None"] + CLIPLIST,),
            "sd3_unet_vae": (["None"] + VAELIST,),
            "use_sd3_hyper_lora": ("BOOLEAN", {"default": False, "label_on": "Use hyper Lora", "label_off": "Ignore Lora"}),
            "sd3_hyper_lora_step": ([4, 8, 16], {"default": 8}),
            "sd3_hyper_lora_strength": ("FLOAT", {"default": 0.125, "min": -20.000, "max": 20.000, "step": 0.001}),

            "kolors_precision": ([ 'fp16', 'quant8', 'quant4'], {"default": "quant8"}),
        },
        "optional": SAMPLER_INPUTS
    }

    @classmethod
    def INPUT_TYPES(cls):
        return cls.INPUT_DICT

    def select_model_concept(self, cascade_stage_a, cascade_stage_b, cascade_stage_c, cascade_clip,
                             flux_diffusion, flux_weight_dtype, flux_gguf, flux_clip_t5xxl, flux_clip_l, flux_vae,
                             hunyuan_clip_t5xxl, hunyuan_clip_l, hunyuan_vae,
                             sd3_clip_g, sd3_clip_l, sd3_clip_t5xxl, sd3_unet_vae,
                             use_sd3_hyper_lora = False, sd3_hyper_lora_step = 8, sd3_hyper_lora_strength = 0.125,
                             kolors_precision = 'quant8',
                             model_version = None,
                             default_sampler_name = 'euler', default_scheduler_name = 'normal', default_cfg_scale = 7, default_steps = 12,
                             sd_vae = "None", sdxl_vae = "None",
                             model_concept = 'Auto',
                             clip_selection = True, vae_selection = True,
                             strength_lcm_lora_model = 1,
                             lightning_selector = "LORA", lightning_model_step = 8, lightning_sampler = False,
                             strength_lightning_lora_model = 1,
                             hypersd_selector = "LORA", hypersd_model_step = 8, hypersd_sampler = False,
                             strength_hypersd_lora_model = 1,
                             flux_sampler = 'ksampler', flux_selector = "DIFFUSION", flux_clip_guidance = 3.5,
                             use_flux_hyper_lora = False, flux_hyper_lora_type = 'FLUX.1-dev', flux_hyper_lora_step = 8, flux_hyper_lora_strength = 0.125,
                             **kwargs
                             ):

        sampler_name = default_sampler_name
        scheduler_name = default_scheduler_name
        steps = default_steps
        cfg_scale = default_cfg_scale

        if model_concept == 'Auto' and model_version == 'Lightning':
            lightning_selector = 'CUSTOM'
        if model_concept == 'Auto' and model_version == 'Hyper':
            hypersd_selector = 'LORA'

        if model_concept == 'Auto' and model_version is not None:
            model_concept = model_version

        vae = 'Baked'
        if vae_selection == True:
            vae = 'Baked'
        else:
            if model_version == "SD1" and sd_vae != "None":
                vae = sd_vae
            if (model_version == "SDXL" or model_version == "KwaiKolors") and sdxl_vae != "None":
                vae = sdxl_vae

        sampler_name_input = model_concept.lower() + '_sampler_name'
        scheduler_name_input = model_concept.lower() + '_scheduler_name'
        steps_input = model_concept.lower() + '_steps'
        cfg_scale_input = model_concept.lower() + '_cfg_scale'

        if sampler_name_input in kwargs:
            sampler_name = kwargs[sampler_name_input]
        if scheduler_name_input in kwargs:
            scheduler_name = kwargs[scheduler_name_input]
        if steps_input in kwargs:
            steps = kwargs[steps_input]
        if cfg_scale_input in kwargs:
            cfg_scale = kwargs[cfg_scale_input]

        match model_concept:
            case 'Lightning':
                if lightning_sampler == True:
                    sampler_name = 'euler'
                    scheduler_name = 'sgm_uniform'
                    steps = lightning_model_step
                    cfg_scale = 1

            case 'Hyper':
                if hypersd_sampler == True and hypersd_selector == 'UNET':
                    sampler_name = 'lcm'
                    steps = 1
                    cfg_scale = 1
                elif hypersd_sampler == True and hypersd_selector != 'UNET':
                    sampler_name = 'euler'
                    scheduler_name = 'sgm_uniform'
                    steps = hypersd_model_step
                    cfg_scale = 1

        if model_concept != 'LCM':
            strength_lcm_lora_model = None

        if model_concept != 'Lightning':
            lightning_selector = None
            lightning_model_step = None
            strength_lightning_lora_model = None

        if model_concept != 'Hyper':
            hypersd_selector = None
            hypersd_model_step = None
            strength_hypersd_lora_model = None

        if model_concept != 'StableCascade':
            cascade_stage_a = None
            cascade_stage_b = None
            cascade_stage_c = None
            cascade_clip = None

        if model_concept != 'Flux':
            flux_selector = None
            flux_diffusion = None
            flux_weight_dtype = None
            flux_gguf = None
            flux_clip_t5xxl = None
            flux_clip_l = None
            flux_clip_guidance = None
            flux_vae = None
            flux_sampler = None
            use_flux_hyper_lora = None
            flux_hyper_lora_type = None
            flux_hyper_lora_step = None
            flux_hyper_lora_strength = None

        if model_concept != 'Hunyuan':
            hunyuan_clip_t5xxl = None
            hunyuan_clip_l = None
            hunyuan_vae = None

        if model_concept != 'KwaiKolors':
            kolors_precision = None

        if model_concept != 'SD3':
            sd3_clip_g = None
            sd3_clip_l = None
            sd3_clip_t5xxl = None
            sd3_unet_vae = None
            use_sd3_hyper_lora = None
            sd3_hyper_lora_step = None
            sd3_hyper_lora_strength = None

        if model_concept == 'Flux' and use_flux_hyper_lora == True:
            steps = flux_hyper_lora_step

        if model_concept == 'SD3' and use_sd3_hyper_lora == True:
            steps = sd3_hyper_lora_step

        return (sampler_name, scheduler_name, steps, round(cfg_scale, 2),
                model_concept,
                clip_selection, vae_selection, vae,
                strength_lcm_lora_model,
                lightning_selector, lightning_model_step, strength_lightning_lora_model,
                cascade_stage_a, cascade_stage_b, cascade_stage_c, cascade_clip,
                hypersd_selector, hypersd_model_step, strength_hypersd_lora_model,
                flux_selector, flux_diffusion, flux_weight_dtype, flux_gguf, flux_clip_t5xxl, flux_clip_l, flux_clip_guidance, flux_vae, flux_sampler,
                use_flux_hyper_lora, flux_hyper_lora_type, flux_hyper_lora_step, flux_hyper_lora_strength,
                hunyuan_clip_t5xxl, hunyuan_clip_l, hunyuan_vae,
                sd3_clip_g, sd3_clip_l, sd3_clip_t5xxl, sd3_unet_vae, use_sd3_hyper_lora, sd3_hyper_lora_step, sd3_hyper_lora_strength,
                kolors_precision
                )

class PrimereConceptDataTuple:
    RETURN_TYPES = ("TUPLE",)
    RETURN_NAMES = ("CONCEPT_DATA",)
    FUNCTION = "load_concept_collector"
    CATEGORY = TREE_DASHBOARD

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "clip_selection": ("CLIP_SELECTION", {"default": True, "forceInput": True}),
                "vae_selection": ("VAE_SELECTION", {"default": True, "forceInput": True}),
                "vae_name": ("VAE_NAME", {"default": "Baked", "forceInput": True}),

                "strength_lcm_lora_model": ("FLOAT", {"default": 1, "forceInput": True}),

                "lightning_selector": ("STRING", {"default": "SAFETENSOR", "forceInput": True}),
                "lightning_model_step": ("INT", {"default": 8, "forceInput": True}),
                "strength_lightning_lora_model": ("FLOAT", {"default": 1, "forceInput": True}),

                "cascade_stage_a": ("STRING", {"forceInput": True}),
                "cascade_stage_b": ("STRING", {"forceInput": True}),
                "cascade_stage_c": ("STRING", {"forceInput": True}),
                "cascade_clip": ("STRING", {"forceInput": True}),

                "hypersd_selector": ("STRING", {"default": "LORA", "forceInput": True}),
                "hypersd_model_step": ("INT", {"default": 8, "forceInput": True}),
                "strength_hypersd_lora_model": ("FLOAT", {"default": 1, "forceInput": True}),

                "flux_selector": ("STRING", {"default": "DIFFUSION", "forceInput": True}),
                "flux_diffusion": ("STRING", {"forceInput": True}),
                "flux_weight_dtype": ("STRING", {"forceInput": True}),
                "flux_gguf": ("STRING", {"forceInput": True}),
                "flux_clip_t5xxl": ("STRING", {"forceInput": True}),
                "flux_clip_l": ("STRING", {"forceInput": True}),
                "flux_clip_guidance": ("FLOAT", {"default": 3.5, "forceInput": True}),
                "flux_vae": ("STRING", {"forceInput": True}),
                "flux_sampler": ("STRING", {"forceInput": True}),
                "use_flux_hyper_lora": ("FLUX_HYPER_LORA", {"forceInput": True}),
                "flux_hyper_lora_type": ("STRING", {"forceInput": True}),
                "flux_hyper_lora_step": ("INT", {"forceInput": True}),
                "flux_hyper_lora_strength": ("FLOAT", {"default": 0.125, "forceInput": True}),

                "hunyuan_clip_t5xxl": ("STRING", {"forceInput": True}),
                "hunyuan_clip_l": ("STRING", {"forceInput": True}),
                "hunyuan_vae": ("STRING", {"forceInput": True}),

                "sd3_clip_g": ("STRING", {"forceInput": True}),
                "sd3_clip_l": ("STRING", {"forceInput": True}),
                "sd3_clip_t5xxl": ("STRING", {"forceInput": True}),
                "sd3_unet_vae": ("STRING", {"forceInput": True}),
                "use_sd3_hyper_lora": ("SD3_HYPER_LORA", {"forceInput": True}),
                "sd3_hyper_lora_step": ("INT", {"default": 8, "forceInput": True}),
                "sd3_hyper_lora_strength": ("FLOAT", {"default": 0.125, "forceInput": True}),

                "kolors_precision": ("STRING", {"forceInput": True}),
            },
        }

    def load_concept_collector(self, **kwargs):
        return (kwargs,)

class PrimereCKPTLoader:
    RETURN_TYPES = ("MODEL", "CLIP", "VAE", "STRING",)
    RETURN_NAMES = ("MODEL", "CLIP", "VAE", "MODEL_VERSION")
    FUNCTION = "load_primere_ckpt"
    CATEGORY = TREE_DASHBOARD

    def __init__(self):
        self.loaded_lora = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "ckpt_name": ("CHECKPOINT_NAME",),
                "use_yaml": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "model_concept": ("STRING", {"forceInput": True}),
                "concept_data": ("TUPLE", {"default": None, "forceInput": True}),
                "loaded_model": ('MODEL', {"forceInput": True, "default": None}),
                "loaded_clip": ('CLIP', {"forceInput": True, "default": None}),
                "loaded_vae": ('VAE', {"forceInput": True, "default": None}),
            },
        }

    def load_primere_ckpt(self, ckpt_name, use_yaml,
                          model_concept = None, concept_data = None,
                          clip_selection = True, vae_selection = True, vae_name = "Baked",
                          strength_lcm_lora_model = 1,
                          lightning_selector = 'LORA', lightning_model_step = 8,
                          strength_lightning_lora_model = 1,
                          hypersd_selector = 'LORA', hypersd_model_step = 8,
                          strength_hypersd_lora_model = 1,
                          cascade_stage_a = None, cascade_stage_b = None, cascade_stage_c = None, cascade_clip = None,
                          loaded_model = None, loaded_clip = None, loaded_vae = None,
                          flux_selector = 'DIFFUSION', flux_diffusion = None, flux_weight_dtype = None, flux_gguf = None, flux_clip_t5xxl = None, flux_clip_l = None, flux_clip_guidance = None, flux_vae = None,
                          use_flux_hyper_lora = False, flux_hyper_lora_type = 'FLUX.1-dev-fp16', flux_hyper_lora_step = 8, flux_hyper_lora_strength = 0.125,
                          hunyuan_clip_t5xxl = None, hunyuan_clip_l = None, hunyuan_vae = None,
                          sd3_clip_g = None, sd3_clip_l = None, sd3_clip_t5xxl = None, sd3_unet_vae = None, use_sd3_hyper_lora = False, sd3_hyper_lora_step = 8, sd3_hyper_lora_strength = 0.125,
                          kolors_precision = 'quant8'
                          ):

        playground_sigma_max = 120
        playground_sigma_min = 0.002

        print('----------- CONCEPT CHECK ---------------------')
        print(model_concept)
        print('unload')

        comfy.model_management.unload_all_models()
        comfy.model_management.cleanup_models()
        comfy.model_management.soft_empty_cache()

        if concept_data is not None:
            if 'clip_selection' in concept_data:
                clip_selection = concept_data['clip_selection']
            if 'vae_selection' in concept_data:
                vae_selection = concept_data['vae_selection']
            if 'vae_name' in concept_data:
                vae_name = concept_data['vae_name']

            if 'strength_lcm_lora_model' in concept_data:
                strength_lcm_lora_model = concept_data['strength_lcm_lora_model']

            if 'lightning_selector' in concept_data:
                lightning_selector = concept_data['lightning_selector']
            if 'lightning_model_step' in concept_data:
                lightning_model_step = concept_data['lightning_model_step']
            if 'strength_lightning_lora_model' in concept_data:
                strength_lightning_lora_model = concept_data['strength_lightning_lora_model']

            if 'hypersd_selector' in concept_data:
                hypersd_selector = concept_data['hypersd_selector']
            if 'hypersd_model_step' in concept_data:
                hypersd_model_step = concept_data['hypersd_model_step']
            if 'strength_hypersd_lora_model' in concept_data:
                strength_hypersd_lora_model = concept_data['strength_hypersd_lora_model']

            if 'cascade_stage_a' in concept_data:
                cascade_stage_a = concept_data['cascade_stage_a']
            if 'cascade_stage_b' in concept_data:
                cascade_stage_b = concept_data['cascade_stage_b']
            if 'cascade_stage_c' in concept_data:
                cascade_stage_c = concept_data['cascade_stage_c']
            if 'cascade_clip' in concept_data:
                cascade_clip = concept_data['cascade_clip']

            if 'flux_selector' in concept_data:
                flux_selector = concept_data['flux_selector']
            if 'flux_diffusion' in concept_data:
                flux_diffusion = concept_data['flux_diffusion']
            if 'flux_weight_dtype' in concept_data:
                flux_weight_dtype = concept_data['flux_weight_dtype']
            if 'flux_gguf' in concept_data:
                flux_gguf = concept_data['flux_gguf']
            if 'flux_clip_t5xxl' in concept_data:
                flux_clip_t5xxl = concept_data['flux_clip_t5xxl']
            if 'flux_clip_l' in concept_data:
                flux_clip_l = concept_data['flux_clip_l']
            if 'flux_clip_guidance' in concept_data:
                flux_clip_guidance = concept_data['flux_clip_guidance']
            if 'flux_vae' in concept_data:
                flux_vae = concept_data['flux_vae']
            if 'use_flux_hyper_lora' in concept_data:
                use_flux_hyper_lora = concept_data['use_flux_hyper_lora']
            if 'flux_hyper_lora_type' in concept_data:
                flux_hyper_lora_type = concept_data['flux_hyper_lora_type']
            if 'flux_hyper_lora_step' in concept_data:
                flux_hyper_lora_step = concept_data['flux_hyper_lora_step']
            if 'flux_hyper_lora_strength' in concept_data:
                flux_hyper_lora_strength = concept_data['flux_hyper_lora_strength']

            if 'hunyuan_clip_t5xxl' in concept_data:
                hunyuan_clip_t5xxl = concept_data['hunyuan_clip_t5xxl']
            if 'hunyuan_clip_l' in concept_data:
                hunyuan_clip_l = concept_data['hunyuan_clip_l']
            if 'hunyuan_vae' in concept_data:
                hunyuan_vae = concept_data['hunyuan_vae']

            # if 'flux_sampler' in concept_data:
            #    flux_sampler = concept_data['flux_sampler']
            if 'sd3_clip_g' in concept_data:
                sd3_clip_g = concept_data['sd3_clip_g']
            if 'sd3_clip_l' in concept_data:
                sd3_clip_l = concept_data['sd3_clip_l']
            if 'sd3_clip_t5xxl' in concept_data:
                sd3_clip_t5xxl = concept_data['sd3_clip_t5xxl']
            if 'sd3_unet_vae' in concept_data:
                sd3_unet_vae = concept_data['sd3_unet_vae']
            if 'use_sd3_hyper_lora' in concept_data:
                use_sd3_hyper_lora = concept_data['use_sd3_hyper_lora']
            if 'sd3_hyper_lora_step' in concept_data:
                sd3_hyper_lora_step = concept_data['sd3_hyper_lora_step']
            if 'sd3_hyper_lora_strength' in concept_data:
                sd3_hyper_lora_strength = concept_data['sd3_hyper_lora_strength']

            if 'kolors_precision' in concept_data:
                kolors_precision = concept_data['kolors_precision']

        modelname_only = Path(ckpt_name).stem
        MODEL_VERSION_ORIGINAL = utility.get_value_from_cache('model_version', modelname_only)
        if MODEL_VERSION_ORIGINAL is None:
            MODEL_VERSION_ORIGINAL = utility.getModelType(ckpt_name, 'checkpoints')
            utility.add_value_to_cache('model_version', ckpt_name, MODEL_VERSION_ORIGINAL)

        if model_concept == "LCM" or (model_concept == 'Lightning' and lightning_selector == 'LORA') or (model_concept == 'Hyper' and hypersd_selector == 'LORA'):
            print('1')
            MODEL_VERSION = MODEL_VERSION_ORIGINAL
            print('2')
        else:
            if model_concept is not None:
                MODEL_VERSION = model_concept
            else:
                MODEL_VERSION = MODEL_VERSION_ORIGINAL

        print(ckpt_name)
        print(MODEL_VERSION)
        print(MODEL_VERSION_ORIGINAL)
        print(model_concept)

        def lcm(self, model, zsnr = False):
            print('def lcm ok')
            m = model.clone()
            print(ckpt_name)
            # sampling_base = comfy.model_sampling.ModelSamplingDiscrete
            sampling_type = nodes_model_advanced.LCM
            sampling_base = utility.ModelSamplingDiscreteLCM
            class ModelSamplingAdvanced(sampling_base, sampling_type):
                pass
            model_sampling = ModelSamplingAdvanced()
            if zsnr:
                model_sampling.set_sigmas(nodes_model_advanced.rescale_zero_terminal_snr_sigmas(model_sampling.sigmas))
            m.add_object_patch("model_sampling", model_sampling)
            return m

        match model_concept:
            case 'Hunyuan':
                print('---Hunyuan---')

                print(hunyuan_vae)
                print(ckpt_name)
                HUNYUAN_VAE = nodes.VAELoader.load_vae(self, hunyuan_vae)[0]
                try:
                    LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, ckpt_name)
                    print('sima ckpt')
                    print(len(LOADED_CHECKPOINT))
                    print(LOADED_CHECKPOINT)
                    HUNYUAN_MODEL = LOADED_CHECKPOINT[0]
                    CLIP = LOADED_CHECKPOINT[1]
                    print('HY clip out')
                    T5 = None
                except Exception:
                    print('hunyuan ckpt')
                    model = 'G/2-1.2'
                    ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
                    model_conf = hydit_conf[model]
                    HUNYUAN_MODEL = load_hydit(model_path = ckpt_path, model_conf = model_conf)

                    dtype = string_to_dtype('FP32', "text_encoder")
                    device = 'GPU'
                    CLIP = load_clip(
                        model_path=folder_paths.get_full_path("clip", hunyuan_clip_l),
                        device=device,
                        dtype=dtype,
                    )
                    T5 = load_t5(
                        model_path=folder_paths.get_full_path("t5", hunyuan_clip_t5xxl),
                        device=device,
                        dtype=dtype,
                    )

                HUNYUAN_CLIP = {'clip': CLIP, 't5': T5}
                print('hy model loaded...')
                return (HUNYUAN_MODEL,) + (HUNYUAN_CLIP,) + (HUNYUAN_VAE,) + (MODEL_VERSION,)

            case 'KwaiKolors':
                print('---KwaiKolors---')
                precision = kolors_precision
                model_name = Path(ckpt_name).stem

                if MODEL_VERSION == MODEL_VERSION_ORIGINAL:
                    fullpathFile = folder_paths.get_full_path('checkpoints', ckpt_name)
                    print('KOLORS symlink check.....')
                    print(fullpathFile)
                    is_link = os.path.islink(str(fullpathFile))
                    if is_link == True:
                        print('DE! symlink')
                        model_name = Path(fullpathFile).stem

                print(model_name)
                device = model_management.get_torch_device()
                offload_device = model_management.unet_offload_device()
                dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}['fp16']
                pbar = comfy.utils.ProgressBar(4)
                model_path = os.path.join(folder_paths.models_dir, "diffusers", model_name)
                pbar.update(1)
                scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder='scheduler')
                print("Load KOLORS UNET...")
                unet = UNet2DConditionModel.from_pretrained(model_path, subfolder='unet', variant="fp16", revision=None, low_cpu_mem_usage=True).to(dtype).eval()
                pipeline = StableDiffusionXLPipeline(unet=unet, scheduler=scheduler)
                KOLORS_MODEL = {'pipeline': pipeline, 'dtype': dtype}

                pbar = comfy.utils.ProgressBar(2)
                text_encoder_path = os.path.join(model_path, "text_encoder")
                pbar.update(1)
                print("Load KOLORS TEXT_ENCODER...")
                text_encoder = ChatGLMModel.from_pretrained(text_encoder_path, torch_dtype=torch.float16)
                if precision == 'quant8':
                    text_encoder.quantize(8)
                elif precision == 'quant4':
                    text_encoder.quantize(4)
                tokenizer = ChatGLMTokenizer.from_pretrained(text_encoder_path)
                pbar.update(1)
                CHATGLM3_MODEL = {'text_encoder': text_encoder, 'tokenizer': tokenizer}

                print("Load KOLORS VAE...")
                print(vae_name)
                if vae_name != "Baked":
                    OUTPUT_VAE = nodes.VAELoader.load_vae(self, vae_name)[0]
                else:
                    vae_list = folder_paths.get_filename_list("vae")
                    allLSDXLvae = list(filter(lambda a: 'sdxl_'.casefold() in a.casefold(), vae_list))
                    print(allLSDXLvae[0])
                    OUTPUT_VAE = nodes.VAELoader.load_vae(self, allLSDXLvae[0])[0]

                print("KOROLS loading OK...")
                # exit()

                return (KOLORS_MODEL,) + (CHATGLM3_MODEL,) + (OUTPUT_VAE,) + (MODEL_VERSION,)

            case 'StableCascade':
                if cascade_stage_a is not None and cascade_stage_b is not None and cascade_stage_c is not None and cascade_clip is not None:
                    print('---StableCascade---')
                    OUTPUT_CLIP_CAS = nodes.CLIPLoader.load_clip(self, cascade_clip, 'stable_cascade')[0]
                    OUTPUT_VAE_CAS = nodes.VAELoader.load_vae(self, cascade_stage_a)[0]
                    if MODEL_VERSION == MODEL_VERSION_ORIGINAL:
                        fullpathFile = folder_paths.get_full_path('checkpoints', ckpt_name)
                        print('symlink check.....')
                        print(fullpathFile)
                        is_link = os.path.islink(str(fullpathFile))
                        if is_link == False:
                            MODEL_C_CAS = nodes.UNETLoader.load_unet(self, ckpt_name, 'default')[0]
                        else:
                            print('CKPT cascade yeah symlink...')
                            File_link = Path(str(fullpathFile)).resolve()
                            print(File_link)
                            linkName_U = str(folder_paths.folder_names_and_paths["diffusion_models"][0][0])
                            print('U folder')
                            print(linkName_U)
                            linkName_D = str(folder_paths.folder_names_and_paths["diffusion_models"][0][1])
                            print('D folder')
                            print(linkName_D)
                            linkedFileName = str(File_link).replace(linkName_U + '\\', '').replace(linkName_D + '\\', '')
                            print(linkedFileName)
                            MODEL_C_CAS = nodes.UNETLoader.load_unet(self, linkedFileName, 'default')[0]
                    else:
                        MODEL_C_CAS = nodes.UNETLoader.load_unet(self, cascade_stage_c, 'default')[0]
                    MODEL_B_CAS = nodes.UNETLoader.load_unet(self, cascade_stage_b, 'default')[0]

                    OUTPUT_MODEL_CAS = [MODEL_B_CAS, MODEL_C_CAS]
                    return (OUTPUT_MODEL_CAS,) + (OUTPUT_CLIP_CAS,) + (OUTPUT_VAE_CAS,) + (MODEL_VERSION,)

            case 'Flux':
                if flux_selector is not None and flux_diffusion is not None and flux_weight_dtype is not None and flux_gguf is not None and flux_clip_t5xxl is not None and flux_clip_l is not None and flux_clip_guidance is not None and flux_vae is not None:
                    print('Flux')
                    print(flux_selector)
                    print(MODEL_VERSION)
                    print(MODEL_VERSION_ORIGINAL)
                    print(model_concept)
                    if MODEL_VERSION == MODEL_VERSION_ORIGINAL:
                        flux_selector = 'SAFETENSOR'

                    match flux_selector:
                        case 'DIFFUSION':
                            MODEL_DIFFUSION = nodes.UNETLoader.load_unet(self, flux_diffusion, flux_weight_dtype)[0]
                            DUAL_CLIP = nodes.DualCLIPLoader.load_clip(self, flux_clip_t5xxl, flux_clip_l, 'flux')[0]
                            FLUX_VAE = nodes.VAELoader.load_vae(self, flux_vae)[0]
                            # return (MODEL_DIFFUSION,) + (DUAL_CLIP,) + (FLUX_VAE,) + (MODEL_VERSION,)

                        case 'GGUF':
                            MODEL_DIFFUSION = gguf_nodes.UnetLoaderGGUF.load_unet(self, flux_gguf)[0]
                            DUAL_CLIP = gguf_nodes.DualCLIPLoaderGGUF.load_clip(self, flux_clip_t5xxl, flux_clip_l, 'flux')[0]
                            FLUX_VAE = nodes.VAELoader.load_vae(self, flux_vae)[0]
                            # return (MODEL_GGUF,) + (CLIP_GGUF,) + (FLUX_VAE,) + (MODEL_VERSION,)

                        case 'SAFETENSOR':
                            fullpathFile = folder_paths.get_full_path('checkpoints', ckpt_name)
                            print('FLUX symlink check.....')
                            print(fullpathFile)
                            is_link = os.path.islink(str(fullpathFile))
                            if is_link == False:
                                print('NEM symlink')
                                MODEL_DIFFUSION = nodes.CheckpointLoaderSimple.load_checkpoint(self, ckpt_name)[0]
                                DUAL_CLIP = nodes.DualCLIPLoader.load_clip(self, flux_clip_t5xxl, flux_clip_l, 'flux')[0]
                                FLUX_VAE = nodes.VAELoader.load_vae(self, flux_vae)[0]
                            else:
                                print('DE! symlink')
                                File_link = Path(str(fullpathFile)).resolve()
                                print(File_link)
                                linkName_U = str(folder_paths.folder_names_and_paths["diffusion_models"][0][0])
                                print('U folder')
                                print(linkName_U)
                                linkName_D = str(folder_paths.folder_names_and_paths["diffusion_models"][0][1])
                                print('D folder')
                                print(linkName_D)
                                linkedFileName = str(File_link).replace(linkName_U + '\\', '').replace(linkName_D + '\\', '')
                                print(linkedFileName)
                                if 'diffusion_models' in str(File_link):
                                    print('GGUF loader')
                                    MODEL_DIFFUSION = gguf_nodes.UnetLoaderGGUF.load_unet(self, linkedFileName)[0]
                                    DUAL_CLIP = gguf_nodes.DualCLIPLoaderGGUF.load_clip(self, flux_clip_t5xxl, flux_clip_l, 'flux')[0]
                                    FLUX_VAE = nodes.VAELoader.load_vae(self, flux_vae)[0]
                                elif 'unet' in str(File_link):
                                    print('UNET loader')
                                    MODEL_DIFFUSION = nodes.UNETLoader.load_unet(self, linkedFileName, flux_weight_dtype)[0]
                                    DUAL_CLIP = nodes.DualCLIPLoader.load_clip(self, flux_clip_t5xxl, flux_clip_l, 'flux')[0]
                                    FLUX_VAE = nodes.VAELoader.load_vae(self, flux_vae)[0]
                                else:
                                    MODEL_DIFFUSION = nodes.CheckpointLoaderSimple.load_checkpoint(self, linkedFileName)[0]
                                    DUAL_CLIP = nodes.DualCLIPLoader.load_clip(self, flux_clip_t5xxl, flux_clip_l, 'flux')[0]
                                    FLUX_VAE = nodes.VAELoader.load_vae(self, flux_vae)[0]

                    if use_flux_hyper_lora == True:
                        print('Kell flux lora....')
                        FLUX_DEV_LORA8 = 'https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-FLUX.1-dev-8steps-lora.safetensors?download=true'
                        FLUX_DEV_FP16_LORA8 = 'https://huggingface.co/nakodanei/Hyper-FLUX.1-dev-8steps-lora-fp16/resolve/main/Hyper-FLUX.1-dev-8steps-lora-fp16.safetensors?download=true'
                        FLUX_DEV_LORA16 = 'https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-FLUX.1-dev-16steps-lora.safetensors?download=true'

                        DOWNLOADED_FLUX_DEV_LORA8 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Hyper-FLUX.1-dev-8steps-lora-fp16.safetensors')
                        DOWNLOADED_FLUX_DEV_FP16_LORA8 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Hyper-FLUX.1-dev-8steps-lora.safetensors')
                        DOWNLOADED_FLUX_DEV_LORA16 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Hyper-FLUX.1-dev-16steps-lora.safetensors')

                        utility.fileDownloader(DOWNLOADED_FLUX_DEV_LORA8, FLUX_DEV_LORA8)
                        utility.fileDownloader(DOWNLOADED_FLUX_DEV_FP16_LORA8, FLUX_DEV_FP16_LORA8)
                        utility.fileDownloader(DOWNLOADED_FLUX_DEV_LORA16, FLUX_DEV_LORA16)

                        print('fluxlorapath test...')
                        downloaded_filelist_filtered = utility.getDownloadedFiles()
                        print(downloaded_filelist_filtered)
                        allHyperFluxLoras = list(filter(lambda a: 'hyper-flux'.casefold() in a.casefold(), downloaded_filelist_filtered))
                        print(allHyperFluxLoras)
                        finalLoras = list(filter(lambda a: str(flux_hyper_lora_step) + 'step'.casefold() in a.casefold() and '-fp16'.casefold() not in a.casefold(), allHyperFluxLoras))
                        print(finalLoras)
                        if flux_hyper_lora_type == 'FLUX.1-dev-fp16':
                            finalLoras_pre = list(filter(lambda a: str(flux_hyper_lora_step) + 'step'.casefold() in a.casefold() and '-fp16'.casefold() in a.casefold(), allHyperFluxLoras))
                            if len(finalLoras_pre) > 0:
                                finalLoras = finalLoras_pre
                        print(finalLoras)

                        LORA_FILE = finalLoras[0]
                        print(LORA_FILE)
                        FULL_LORA_PATH = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', LORA_FILE)
                        print(FULL_LORA_PATH)

                        if FULL_LORA_PATH is not None and os.path.exists(FULL_LORA_PATH) == True:
                            print('Lora path ok')
                            if flux_hyper_lora_strength != 0:
                                lora = None
                                if self.loaded_lora is not None:
                                    if self.loaded_lora[0] == FULL_LORA_PATH:
                                        lora = self.loaded_lora[1]
                                    else:
                                        temp = self.loaded_lora
                                        self.loaded_lora = None
                                        del temp

                                if lora is None:
                                    lora = comfy.utils.load_torch_file(FULL_LORA_PATH, safe_load = True)
                                    self.loaded_lora = (FULL_LORA_PATH, lora)

                                MODEL_DIFFUSION = comfy.sd.load_lora_for_models(MODEL_DIFFUSION, None, lora, flux_hyper_lora_strength, 0)[0]
                                print('Flux lora loaded...')

                    return (MODEL_DIFFUSION,) + (DUAL_CLIP,) + (FLUX_VAE,) + (MODEL_VERSION,)

            case 'Hyper':
                if hypersd_selector == 'UNET':
                    print('hyper-unet')
                    ModelConceptChanges = utility.ModelConceptNames(ckpt_name, model_concept, lightning_selector, lightning_model_step, hypersd_selector, hypersd_model_step, 'SDXL')
                    print(ModelConceptChanges)
                    lora_name = ModelConceptChanges['lora_name']
                    unet_name = ModelConceptChanges['unet_name']
                    hyperModeValid = ModelConceptChanges['hyperModeValid']
                    OUTPUT_MODEL = utility.BDanceConceptHelper(self, model_concept, hyperModeValid, hypersd_selector, hypersd_model_step, None, lora_name, unet_name, None)
                    return (OUTPUT_MODEL[0],) + (OUTPUT_MODEL[1],) + (OUTPUT_MODEL[2],) + (MODEL_VERSION,)

        path = Path(ckpt_name)
        ModelName = path.stem
        ModelConfigPath = path.parent.joinpath(ModelName + '.yaml')
        ModelConfigFullPath = Path(folder_paths.models_dir).joinpath('checkpoints').joinpath(ModelConfigPath)
        print('7')
        print(loaded_model)
        print(loaded_clip)
        print(loaded_vae)

        LOADED_CHECKPOINT = []
        if loaded_model is not None and loaded_clip is not None and loaded_vae is not None:
            print('7.0')
            LOADED_CHECKPOINT.insert(0, loaded_model)
            LOADED_CHECKPOINT.insert(1, loaded_clip)
            LOADED_CHECKPOINT.insert(2, loaded_vae)
        else:
            print(ckpt_name)
            if os.path.isfile(ModelConfigFullPath) and use_yaml == True:
                print('7.1')
                print(ModelConfigFullPath)
                print(ckpt_name)
                ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
                print(ckpt_path)
                print('7.2')
                print(ModelName + '.yaml file found and loading...')
                try:
                    LOADED_CHECKPOINT = comfy.sd.load_checkpoint(ModelConfigFullPath, ckpt_path, True, True, None, None, None)
                except Exception:
                    LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, ckpt_name)
            else:
                print('7.3')
                print(ckpt_name)
                try:
                    print('load ckpt')
                    LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, ckpt_name)
                except Exception:
                    print('load unet')
                    fullpathFile = folder_paths.get_full_path('checkpoints', ckpt_name)
                    print('CKPT UNET symlink check.....')
                    print(fullpathFile)
                    is_link = os.path.islink(str(fullpathFile))
                    if is_link == False:
                        LOADED_CHECKPOINT = nodes.UNETLoader.load_unet(self, ckpt_name, 'default')
                    else:
                        print('yeah CKTP symlink...')
                        File_link = Path(str(fullpathFile)).resolve()
                        linkName_U = str(folder_paths.folder_names_and_paths["diffusion_models"][0][0])
                        linkName_D = str(folder_paths.folder_names_and_paths["diffusion_models"][0][1])
                        linkedFileName = str(File_link).replace(linkName_U + '\\', '').replace(linkName_D + '\\', '')
                        LOADED_CHECKPOINT = nodes.UNETLoader.load_unet(self, linkedFileName, 'default')

                print('7.3.1')

        print('8')
        OUTPUT_MODEL = LOADED_CHECKPOINT[0]
        if model_concept == 'SD3':
            print(len(LOADED_CHECKPOINT))
            # print(type(LOADED_CHECKPOINT[1]).__name__)
            print('SD3 triple  clipping...')
            print(sd3_clip_g)
            print(sd3_clip_l)
            print(sd3_clip_t5xxl)
            print(clip_selection)
            print(list(LOADED_CHECKPOINT))
            # print(1 not in list(LOADED_CHECKPOINT))

            if len(LOADED_CHECKPOINT) < 2 or (len(LOADED_CHECKPOINT) >= 2 and type(LOADED_CHECKPOINT[1]).__name__ != 'CLIP') or clip_selection == False:
                print('clip models loading...')
                OUTPUT_CLIP = nodes_sd3.TripleCLIPLoader.load_clip(self, sd3_clip_g, sd3_clip_l, sd3_clip_t5xxl)[0]
            else:
                print('clip models __NOT__ loading...')
                OUTPUT_CLIP = LOADED_CHECKPOINT[1]

            print('vaecheck...')
            # print(type(LOADED_CHECKPOINT[2]).__name__)
            if len(LOADED_CHECKPOINT) == 3 and type(LOADED_CHECKPOINT[2]).__name__ == 'VAE':
                OUTPUT_VAE = LOADED_CHECKPOINT[2]
            else:
                print('custom vae:')
                print(sd3_unet_vae)
                OUTPUT_VAE = nodes.VAELoader.load_vae(self, sd3_unet_vae)[0]

            if use_sd3_hyper_lora == True:
                print('Kell SD3 lora....')
                # sd3_hyper_lora_step
                # sd3_hyper_lora_strength

                SD3_LORA4 = 'https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SD3-4steps-CFG-lora.safetensors?download=true'
                SD3_LORA8 = 'https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SD3-8steps-CFG-lora.safetensors?download=true'
                SD3_LORA16 = 'https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SD3-16steps-CFG-lora.safetensors?download=true'

                DOWNLOADED_SD3_LORA4 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Hyper-SD3-4steps-CFG-lora.safetensors')
                DOWNLOADED_SD3_LORA8 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Hyper-SD3-8steps-CFG-lora.safetensors')
                DOWNLOADED_SD3_LORA16 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Hyper-SD3-16steps-CFG-lora.safetensors')

                utility.fileDownloader(DOWNLOADED_SD3_LORA4, SD3_LORA4)
                utility.fileDownloader(DOWNLOADED_SD3_LORA8, SD3_LORA8)
                utility.fileDownloader(DOWNLOADED_SD3_LORA16, SD3_LORA16)
                print('sd3 lorapath test...')
                downloaded_filelist_filtered = utility.getDownloadedFiles()
                print(downloaded_filelist_filtered)
                allHyperSD3Loras = list(filter(lambda a: 'hyper-sd3'.casefold() in a.casefold(), downloaded_filelist_filtered))
                print(allHyperSD3Loras)
                finalLoras = list(filter(lambda a: str(sd3_hyper_lora_step) + 'step'.casefold() in a.casefold(), allHyperSD3Loras))
                print(finalLoras)
                LORA_FILE = finalLoras[0]
                print(LORA_FILE)
                FULL_LORA_PATH = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', LORA_FILE)
                print(FULL_LORA_PATH)

                if FULL_LORA_PATH is not None and os.path.exists(FULL_LORA_PATH) == True:
                    print('Lora path ok')
                    if sd3_hyper_lora_strength != 0:
                        lora = None
                        if self.loaded_lora is not None:
                            if self.loaded_lora[0] == FULL_LORA_PATH:
                                lora = self.loaded_lora[1]
                            else:
                                temp = self.loaded_lora
                                self.loaded_lora = None
                                del temp

                        if lora is None:
                            lora = comfy.utils.load_torch_file(FULL_LORA_PATH, safe_load=True)
                            self.loaded_lora = (FULL_LORA_PATH, lora)

                        OUTPUT_MODEL = comfy.sd.load_lora_for_models(OUTPUT_MODEL, None, lora, sd3_hyper_lora_strength, 0)[0]
                        print('SD3 lora loaded...')

            return (OUTPUT_MODEL,) + (OUTPUT_CLIP,) + (OUTPUT_VAE,) + (MODEL_VERSION,)
        else:
            OUTPUT_CLIP = LOADED_CHECKPOINT[1]
        print('9')

        print(model_concept)
        print(MODEL_VERSION)

        if model_concept == 'Lightning' and lightning_selector == 'LORA' and MODEL_VERSION != 'SDXL':
            model_concept = MODEL_VERSION

        match model_concept:
            case 'Hyper' | 'Lightning':
                HYPER_LIGHTNING_ORIGINAL_VERSION = utility.getModelType(ckpt_name, 'checkpoints')
                print(HYPER_LIGHTNING_ORIGINAL_VERSION)

                print('Hyper Ligntning loras check....')
                if lightning_selector == 'LORA':
                    Lightning_SDXL_2 = 'https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_2step_lora.safetensors?download=true'
                    DOWNLOADED_Lightning_SDXL_2 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'sdxl_lightning_2step_lora.safetensors')

                    Lightning_SDXL_4 = 'https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_4step_lora.safetensors?download=true'
                    DOWNLOADED_Lightning_SDXL_4 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'sdxl_lightning_4step_lora.safetensors')

                    Lightning_SDXL_8 = 'https://huggingface.co/ByteDance/SDXL-Lightning/resolve/main/sdxl_lightning_8step_lora.safetensors?download=true'
                    DOWNLOADED_Lightning_SDXL_8 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'sdxl_lightning_8step_lora.safetensors')

                    utility.fileDownloader(DOWNLOADED_Lightning_SDXL_2, Lightning_SDXL_2)
                    utility.fileDownloader(DOWNLOADED_Lightning_SDXL_4, Lightning_SDXL_4)
                    utility.fileDownloader(DOWNLOADED_Lightning_SDXL_8, Lightning_SDXL_8)

                if hypersd_selector == 'LORA':
                    Hyper_SD_1 = 'https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SD15-1step-lora.safetensors?download=true'
                    DOWNLOADED_Hyper_SD_1 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Hyper-SD15-1step-lora.safetensors')
                    utility.fileDownloader(DOWNLOADED_Hyper_SD_1, Hyper_SD_1)

                    Hyper_SD_2 = 'https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SD15-2steps-lora.safetensors?download=true'
                    DOWNLOADED_Hyper_SD_2 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Hyper-SD15-2steps-lora.safetensors')
                    utility.fileDownloader(DOWNLOADED_Hyper_SD_2, Hyper_SD_2)

                    Hyper_SD_4 = 'https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SD15-4steps-lora.safetensors?download=true'
                    DOWNLOADED_Hyper_SD_4 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Hyper-SD15-4steps-lora.safetensors')
                    utility.fileDownloader(DOWNLOADED_Hyper_SD_4, Hyper_SD_4)

                    Hyper_SD_8 = 'https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SD15-8steps-lora.safetensors?download=true'
                    DOWNLOADED_Hyper_SD_8 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Hyper-SD15-8steps-lora.safetensors')
                    utility.fileDownloader(DOWNLOADED_Hyper_SD_8, Hyper_SD_8)

                    Hyper_SD_12 = 'https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SD15-12steps-CFG-lora.safetensors?download=true'
                    DOWNLOADED_Hyper_SD_12 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Hyper-SD15-12steps-CFG-lora.safetensors')
                    utility.fileDownloader(DOWNLOADED_Hyper_SD_12, Hyper_SD_12)

                    Hyper_SDXL_1 = 'https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SDXL-1step-lora.safetensors?download=true'
                    DOWNLOADED_Hyper_SDXL_1 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Hyper-SDXL-1step-lora.safetensors')
                    utility.fileDownloader(DOWNLOADED_Hyper_SDXL_1, Hyper_SDXL_1)

                    Hyper_SDXL_2 = 'https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SDXL-2steps-lora.safetensors?download=true'
                    DOWNLOADED_Hyper_SDXL_2 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Hyper-SDXL-2steps-lora.safetensors')
                    utility.fileDownloader(DOWNLOADED_Hyper_SDXL_2, Hyper_SDXL_2)

                    Hyper_SDXL_4 = 'https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SDXL-4steps-lora.safetensors?download=true'
                    DOWNLOADED_Hyper_SDXL_4 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Hyper-SDXL-4steps-lora.safetensors')
                    utility.fileDownloader(DOWNLOADED_Hyper_SDXL_4, Hyper_SDXL_4)

                    Hyper_SDXL_8 = 'https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SDXL-8steps-lora.safetensors?download=true'
                    DOWNLOADED_Hyper_SDXL_8 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Hyper-SDXL-8steps-lora.safetensors')
                    utility.fileDownloader(DOWNLOADED_Hyper_SDXL_8, Hyper_SDXL_8)

                    Hyper_SDXL_12 = 'https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SDXL-12steps-CFG-lora.safetensors?download=true'
                    DOWNLOADED_Hyper_SDXL_12 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Hyper-SDXL-12steps-CFG-lora.safetensors')
                    utility.fileDownloader(DOWNLOADED_Hyper_SDXL_12, Hyper_SDXL_12)

                print('5')
                print('Hyper or Lightning')
                print(MODEL_VERSION)
                ModelConceptChanges = utility.ModelConceptNames(ckpt_name, model_concept, lightning_selector, lightning_model_step, hypersd_selector, hypersd_model_step, HYPER_LIGHTNING_ORIGINAL_VERSION)
                print('6')
                print(ModelConceptChanges)
                ckpt_name = ModelConceptChanges['ckpt_name']
                if ModelConceptChanges['lora_name'] is not None:
                    lora_name = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', ModelConceptChanges['lora_name'])
                else:
                    lora_name = None
                unet_name = ModelConceptChanges['unet_name']
                lightningModeValid = ModelConceptChanges['lightningModeValid']
                hyperModeValid = ModelConceptChanges['hyperModeValid']

                if lightningModeValid == True and loaded_model is None:
                    print('Lightning end')
                    OUTPUT_MODEL = utility.BDanceConceptHelper(self, model_concept, lightningModeValid, lightning_selector, lightning_model_step, OUTPUT_MODEL, lora_name, unet_name, ckpt_name, strength_lightning_lora_model)
                    print("Lightning lora loaded...")

                if hyperModeValid == True and loaded_model is None:
                    print('Hyper end')
                    print(strength_hypersd_lora_model)
                    print(ckpt_name)
                    print(lora_name)
                    OUTPUT_MODEL = utility.BDanceConceptHelper(self, model_concept, hyperModeValid, hypersd_selector, hypersd_model_step, OUTPUT_MODEL, lora_name, unet_name, ckpt_name, strength_hypersd_lora_model)
                    print(type(OUTPUT_MODEL).__name__)
                    vae_selection = True
                    print("Hyper lora loaded...")

            case 'LCM':
                if MODEL_VERSION == 'SD1' or MODEL_VERSION == 'SDXL':
                    SDXL_LORA = 'https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors?download=true'
                    SD_LORA = 'https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors?download=true'

                    DOWNLOADED_SD_LORA = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'lcm_lora_sd.safetensors')
                    DOWNLOADED_SDXL_LORA = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'lcm_lora_sdxl.safetensors')

                    utility.fileDownloader(DOWNLOADED_SD_LORA, SD_LORA)
                    utility.fileDownloader(DOWNLOADED_SDXL_LORA, SDXL_LORA)

                    print(MODEL_VERSION)
                    LORA_PATH = None
                    if MODEL_VERSION == 'SDXL':
                        LORA_PATH = DOWNLOADED_SDXL_LORA
                    elif MODEL_VERSION == 'SD1':
                        LORA_PATH = DOWNLOADED_SD_LORA
                    print(LORA_PATH)

                    if LORA_PATH is not None and os.path.exists(LORA_PATH) == True:
                        print('Lora path ok')
                        if strength_lcm_lora_model != 0:
                            lora = None
                            if self.loaded_lora is not None:
                                if self.loaded_lora[0] == LORA_PATH:
                                    lora = self.loaded_lora[1]
                                else:
                                    temp = self.loaded_lora
                                    self.loaded_lora = None
                                    del temp

                            if lora is None:
                                lora = comfy.utils.load_torch_file(LORA_PATH, safe_load = True)
                                self.loaded_lora = (LORA_PATH, lora)

                            print(LORA_PATH)
                            MODEL_LORA = comfy.sd.load_lora_for_models(OUTPUT_MODEL, None, lora, strength_lcm_lora_model, 0)[0]

                            OUTPUT_MODEL = lcm(self, MODEL_LORA, False)
                            # OUTPUT_CLIP = CLIP_LORA

            case 'Playground':
                print('Playground end')
                OUTPUT_MODEL = nodes_model_advanced.ModelSamplingContinuousEDM.patch(self, OUTPUT_MODEL, 'edm_playground_v2.5', playground_sigma_max, playground_sigma_min)[0]

        #  vae_selection = True, vae_name = "Baked",
        print("VAE selection")
        if len(LOADED_CHECKPOINT) < 3 or (len(LOADED_CHECKPOINT) == 3 and type(LOADED_CHECKPOINT[2]).__name__ != 'VAE') or vae_selection == False:
            print("custom VAE:")
            print(vae_name)
            print(MODEL_VERSION)
            if vae_name != "Baked":
                OUTPUT_VAE = nodes.VAELoader.load_vae(self, vae_name)[0]
            else:
                vae_list = folder_paths.get_filename_list("vae")
                print(vae_list)
                if MODEL_VERSION == 'SD1':
                    allLSD1vae = list(filter(lambda a: 'sdxl'.casefold() not in a.casefold() and 'flux'.casefold() not in a.casefold() and 'hunyuan'.casefold() not in a.casefold() and 'stage'.casefold() not in a.casefold(), vae_list))
                    print(allLSD1vae[0])
                    OUTPUT_VAE = nodes.VAELoader.load_vae(self, allLSD1vae[0])[0]
                else:
                    allLSDXLvae = list(filter(lambda a: 'sdxl_'.casefold() in a.casefold(), vae_list))
                    print(allLSDXLvae[0])
                    OUTPUT_VAE = nodes.VAELoader.load_vae(self, allLSDXLvae[0])[0]
        else:
            print("baked VAE:")
            OUTPUT_VAE = LOADED_CHECKPOINT[2]

        return (OUTPUT_MODEL,) + (OUTPUT_CLIP,) + (OUTPUT_VAE,) + (MODEL_VERSION,)

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

class PrimerePromptSwitch:
    any_typ = AnyType("*")

    RETURN_TYPES = (any_typ, any_typ, "INT", "TUPLE")
    RETURN_NAMES = ("PROMPT+", "PROMPT-", "SELECTED_INDEX", "PREFERRED")
    FUNCTION = "promptswitch"
    CATEGORY = TREE_DASHBOARD

    @classmethod
    def INPUT_TYPES(cls):
        any_typ = AnyType("*")

        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1}),
            },
            "optional": {
                "prompt_pos_1": (any_typ,),
                "prompt_neg_1": (any_typ,),
                "preferred_1": (any_typ,),
            },
        }

    def promptswitch(self, *args, **kwargs):
        selected_index = int(kwargs['select'])
        input_namep = f"prompt_pos_{selected_index}"
        input_namen = f"prompt_neg_{selected_index}"
        input_preferred = f"preferred_{selected_index}"

        if input_namep in kwargs:
            return (kwargs[input_namep], kwargs[input_namen], selected_index, kwargs[input_preferred])
        else:
            print(f"PrimerePromptSwitch: invalid select index (ignored)")
            return (None, None, selected_index, None)


class PrimereSeed:
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("SEED",)
    FUNCTION = "seed"
    CATEGORY = TREE_DASHBOARD

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed": ("INT", {"default": -1, "min": -1125899906842624, "max": 1125899906842624}),
            }
        }

    def seed(self, seed = -1):
      return (seed,)

class PrimereFractalLatent:
    RETURN_TYPES = ("LATENT", "IMAGE", "TUPLE")
    RETURN_NAMES = ("LATENTS", "PREVIEWS", "WORKFLOW_TUPLE")
    FUNCTION = "primere_latent_noise"
    CATEGORY = TREE_DASHBOARD

    pln = PowerLawNoise('cpu')
    INPUT_DICT = {
        "required": {
            "width": ("INT", {"default": 512, "max": 8192, "min": 64, "forceInput": True}),
            "height": ("INT", {"default": 512, "max": 8192, "min": 64, "forceInput": True}),
            "rand_noise_type": ("BOOLEAN", {"default": False}),
            "noise_type": (pln.get_noise_types(),),
            "rand_alpha_exponent": ("BOOLEAN", {"default": True}),
            "alpha_exponent": ("FLOAT", {"default": 1.0, "max": 12.0, "min": -12.0, "step": 0.001}),
            "alpha_exp_rand_min": ("FLOAT", {"default": 0.5, "max": 12.0, "min": -12.0, "step": 0.001}),
            "alpha_exp_rand_max": ("FLOAT", {"default": 1.5, "max": 12.0, "min": -12.0, "step": 0.001}),
            "rand_modulator": ("BOOLEAN", {"default": True}),
            "modulator": ("FLOAT", {"default": 1.0, "max": 2.0, "min": 0.1, "step": 0.01}),
            "modulator_rand_min": ("FLOAT", {"default": 0.8, "max": 2.0, "min": 0.1, "step": 0.01}),
            "modulator_rand_max": ("FLOAT", {"default": 1.4, "max": 2.0, "min": 0.1, "step": 0.01}),
            "noise_seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff, "forceInput": True}),
            "rand_device": ("BOOLEAN", {"default": False}),
            "device": (["cpu", "cuda"],),
            "expand_random_limits": ("BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF"}),
        },
        "optional": {
            "optional_vae": ("VAE",),
            "workflow_tuple": ("TUPLE", {"default": None}),
        }
    }

    @classmethod
    def INPUT_TYPES(cls):
        return cls.INPUT_DICT

    @classmethod
    def IS_CHANGED(self, **kwargs):
        if kwargs['expand_random_limits'] == True or kwargs['rand_noise_type'] == True or kwargs['rand_device'] == True or kwargs['rand_alpha_exponent'] == True or kwargs['rand_modulator'] == True:
            return float('NaN')

    def primere_latent_noise(self, width, height, rand_noise_type, noise_type, rand_alpha_exponent, alpha_exponent, alpha_exp_rand_min, alpha_exp_rand_max, rand_modulator, modulator, modulator_rand_min, modulator_rand_max, noise_seed, rand_device, device, optional_vae = None, workflow_tuple = None, expand_random_limits = False):
        if workflow_tuple is not None and len(workflow_tuple) > 0 and 'exif_status' in workflow_tuple and workflow_tuple['exif_status'] == 'SUCCEED':
            if 'latent_data' in workflow_tuple and len(workflow_tuple['latent_data']) > 0 and 'setup_states' in workflow_tuple and 'latent_setup' in workflow_tuple['setup_states']:
                if workflow_tuple['setup_states']['latent_setup'] == True:
                    expand_random_limits = False
                    rand_device = False
                    rand_alpha_exponent = False
                    rand_modulator = False
                    rand_noise_type = False
                    noise_type = workflow_tuple['latent_data']['noise_type']
                    device = workflow_tuple['latent_data']['device']
                    alpha_exponent = workflow_tuple['latent_data']['alpha_exponent']
                    modulator = workflow_tuple['latent_data']['modulator']

        if expand_random_limits == True:
            rand_device = True
            rand_alpha_exponent = True
            rand_modulator = True
            rand_noise_type = True
            alpha_exp_rand_min = -12.00
            alpha_exp_rand_max = 7.00
            modulator_rand_min = 0.10
            modulator_rand_max = 2.00

        if rand_noise_type == True:
            pln = PowerLawNoise(device)
            noise_type = random.choice(pln.get_noise_types())

        if rand_device == True:
            device = random.choice(["cpu", "cuda"])

        if expand_random_limits == True and (noise_type == 'white' or noise_type == 'violet'):
            alpha_exp_rand_min = 0.00

        power_law = PowerLawNoise(device = device)

        if rand_alpha_exponent == True:
            alpha_exponent = round(random.uniform(alpha_exp_rand_min, alpha_exp_rand_max), 3)

        if rand_modulator == True:
            modulator = round(random.uniform(modulator_rand_min, modulator_rand_max), 2)

        tensors = power_law(1, width, height, scale = 1, alpha = alpha_exponent, modulator = modulator, noise_type = noise_type, seed = noise_seed)
        alpha_channel = torch.ones((1, height, width, 1), dtype = tensors.dtype, device = "cpu")
        tensors = torch.cat((tensors, alpha_channel), dim = 3)

        if optional_vae is None:
            latents = tensors.permute(0, 3, 1, 2)
            latents = F.interpolate(latents, size=((height // 8), (width // 8)), mode = 'nearest-exact')
            return {'samples': latents}, tensors, workflow_tuple

        encoder = nodes.VAEEncode()
        latents = []
        for tensor in tensors:
            tensor = tensor.unsqueeze(0)
            latents.append(encoder.encode(optional_vae, tensor)[0]['samples'])
        latents = torch.cat(latents)

        if workflow_tuple is not None:
            workflow_tuple['latent_data'] = {}
            workflow_tuple['latent_data']['noise_type'] = noise_type
            workflow_tuple['latent_data']['alpha_exponent'] = alpha_exponent
            workflow_tuple['latent_data']['modulator'] = modulator
            workflow_tuple['latent_data']['device'] = device

        return {'samples': latents}, tensors, workflow_tuple

class PrimereCLIP:
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING", "STRING", "STRING", "TUPLE")
    RETURN_NAMES = ("COND+", "COND-", "PROMPT+", "PROMPT-", "PROMPT L+", "PROMPT L-", "WORKFLOW_TUPLE")
    FUNCTION = "clip_encode"
    CATEGORY = TREE_DASHBOARD

    @staticmethod
    def get_default_neg(toml_path: str):
        with open(toml_path, "rb") as f:
            style_def_neg = tomli.load(f)
        return style_def_neg
    @ classmethod
    def INPUT_TYPES(cls):
        DEF_TOML_DIR = os.path.join(PRIMERE_ROOT, 'Toml')
        cls.default_neg = cls.get_default_neg(os.path.join(DEF_TOML_DIR, "default_neg.toml"))
        cls.default_pos = cls.get_default_neg(os.path.join(DEF_TOML_DIR, "default_pos.toml"))
        CLIPLIST = folder_paths.get_filename_list("clip")
        CLIPLIST += folder_paths.get_filename_list("clip_gguf")
        CLIPLIST += folder_paths.get_filename_list("t5")
        cls.CLIPLIST = CLIPLIST

        return {
            "required": {
                "clip": ("CLIP", {"forceInput": True}),
                "model_version": ("STRING", {"default": 'SD1', "forceInput": True}),
                "positive_prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": ("STRING", {"forceInput": True}),
                "clip_mode": ("BOOLEAN", {"default": True, "label_on": "CLIP", "label_off": "Long-CLIP"}),
                "clip_model": (['Default'] + sorted(cls.CLIPLIST),),
                "longclip_model": (['Default'] + sorted(cls.CLIPLIST),),
                "last_layer": ("INT", {"default": 0, "min": -24, "max": 0, "step": 1}),
                "negative_strength": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.01}),
                "use_int_style": ("BOOLEAN", {"default": False}),
                "int_style_pos": (['None'] + sorted(list(cls.default_pos.keys())),),
                "int_style_pos_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "int_style_neg": (['None'] + sorted(list(cls.default_neg.keys())),),
                "int_style_neg_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "adv_encode": ("BOOLEAN", {"default": False}),
                "token_normalization": (["none", "mean", "length", "length+mean"],),
                "weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),
            },
            "optional": {
                # "clip_raw": ("CLIP", {"forceInput": True}),
                "model_concept": ("STRING", {"default": "Normal", "forceInput": True}),
                "model_keywords": ("MODEL_KEYWORD", {"forceInput": True}),
                "lora_keywords": ("MODEL_KEYWORD", {"forceInput": True}),
                "lycoris_keywords": ("MODEL_KEYWORD", {"forceInput": True}),
                "embedding_pos": ("EMBEDDING", {"forceInput": True}),
                "embedding_neg": ("EMBEDDING", {"forceInput": True}),

                "opt_pos_prompt": ("STRING", {"forceInput": True}),
                "opt_pos_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "opt_neg_prompt": ("STRING", {"forceInput": True}),
                "opt_neg_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),

                "style_position": ("BOOLEAN", {"default": False, "label_on": "Top", "label_off": "Bottom"}),
                "style_pos_prompt": ("STRING", {"forceInput": True}),
                "style_pos_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "style_neg_prompt": ("STRING", {"forceInput": True}),
                "style_neg_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),

                "sdxl_positive_l": ("STRING", {"forceInput": True}),
                "sdxl_negative_l": ("STRING", {"forceInput": True}),
                "copy_prompt_to_l": ("BOOLEAN", {"default": True}),
                "sdxl_l_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION, "forceInput": True}),
                "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION, "forceInput": True}),
                "workflow_tuple": ("TUPLE", {"default": None}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT"
            }
        }

    def clip_encode(self, clip, clip_mode, last_layer, negative_strength, int_style_pos_strength, int_style_neg_strength, opt_pos_strength, opt_neg_strength, style_pos_strength, style_neg_strength, int_style_pos, int_style_neg, adv_encode, token_normalization, weight_interpretation, sdxl_l_strength, extra_pnginfo, prompt, copy_prompt_to_l = True, width = 1024, height = 1024, positive_prompt = "", negative_prompt = "", clip_model = 'Default', longclip_model = 'Default', model_keywords = None, lora_keywords = None, lycoris_keywords = None, embedding_pos = None, embedding_neg = None, opt_pos_prompt = "", opt_neg_prompt = "", style_position = False, style_neg_prompt = "", style_pos_prompt = "", sdxl_positive_l = "", sdxl_negative_l = "", use_int_style = False, model_version = "SD1", model_concept = "Normal", workflow_tuple = None):
        print('--------- CLIP CONCEPT TEST ---------------------')

        if workflow_tuple is not None and len(workflow_tuple) > 0 and 'exif_status' in workflow_tuple and workflow_tuple['exif_status'] == 'SUCCEED':
            if 'prompt_encoder' in workflow_tuple and len(workflow_tuple['prompt_encoder']) > 0 and 'setup_states' in workflow_tuple and 'clip_encoder_setup' in workflow_tuple['setup_states']:
                if workflow_tuple['setup_states']['clip_encoder_setup'] == True:
                    clip_mode = workflow_tuple['prompt_encoder']['clip_mode']
                    last_layer = workflow_tuple['prompt_encoder']['last_layer']
                    negative_strength = workflow_tuple['prompt_encoder']['negative_strength']
                    adv_encode = workflow_tuple['prompt_encoder']['adv_encode']
                    token_normalization = workflow_tuple['prompt_encoder']['token_normalization']
                    weight_interpretation = workflow_tuple['prompt_encoder']['weight_interpretation']
                    copy_prompt_to_l = workflow_tuple['prompt_encoder']['copy_prompt_to_l']
                    sdxl_l_strength = workflow_tuple['prompt_encoder']['sdxl_l_strength']
                    clip_model = workflow_tuple['prompt_encoder']['clip_model']
                    longclip_model = workflow_tuple['prompt_encoder']['longclip_model']
                    if 'use_int_style' in workflow_tuple['prompt_encoder']:
                        use_int_style = workflow_tuple['prompt_encoder']['use_int_style']
                    if 'int_style_pos' in workflow_tuple['prompt_encoder']:
                        int_style_pos = workflow_tuple['prompt_encoder']['int_style_pos']
                    if 'int_style_pos_strength' in workflow_tuple['prompt_encoder']:
                        int_style_pos_strength = workflow_tuple['prompt_encoder']['int_style_pos_strength']
                    if 'int_style_neg' in workflow_tuple['prompt_encoder']:
                        int_style_neg = workflow_tuple['prompt_encoder']['int_style_neg']
                    if 'int_style_neg_strength' in workflow_tuple['prompt_encoder']:
                        int_style_neg_strength = workflow_tuple['prompt_encoder']['int_style_neg_strength']
                if workflow_tuple['setup_states']['clip_optional_prompts'] == True:
                    opt_pos_prompt = workflow_tuple['prompt_encoder']['opt_pos_prompt']
                    opt_pos_strength = workflow_tuple['prompt_encoder']['opt_pos_strength']
                    opt_neg_prompt = workflow_tuple['prompt_encoder']['opt_neg_prompt']
                    opt_neg_strength = workflow_tuple['prompt_encoder']['opt_neg_strength']
                if workflow_tuple['setup_states']['clip_style_prompts'] == True:
                    style_pos_prompt = workflow_tuple['prompt_encoder']['style_pos_prompt']
                    style_pos_strength = workflow_tuple['prompt_encoder']['style_pos_strength']
                    style_neg_prompt = workflow_tuple['prompt_encoder']['style_neg_prompt']
                    style_neg_strength = workflow_tuple['prompt_encoder']['style_neg_strength']
                    style_position = workflow_tuple['prompt_encoder']['style_position']
                if workflow_tuple['setup_states']['clip_additional_keywords'] == True:
                    model_keywords = workflow_tuple['prompt_encoder']['model_keywords']
                    lora_keywords = workflow_tuple['prompt_encoder']['lora_keywords']
                    lycoris_keywords = workflow_tuple['prompt_encoder']['lycoris_keywords']
                    embedding_pos = workflow_tuple['prompt_encoder']['embedding_pos']
                    embedding_neg = workflow_tuple['prompt_encoder']['embedding_neg']

        if workflow_tuple is None:
            workflow_tuple = {}

        print(model_concept)
        if model_concept == 'Hunyuan' or model_concept == 'KwaiKolors' or model_concept == 'SD3' or model_concept == 'Playground' or model_concept == 'StableCascade' or model_concept == 'Turbo' or model_concept == 'Flux' or model_concept == 'Lightning':
            model_version = 'SDXL'
            clip_model = 'Default'

        is_sdxl = 0
        match model_version:
            case 'SDXL':
                is_sdxl = 1

        print(is_sdxl)
        print(model_version)

        additional_positive = int_style_pos
        additional_negative = int_style_neg
        if int_style_pos == 'None' or use_int_style == False:
            additional_positive = None
        if int_style_neg == 'None' or use_int_style == False:
            additional_negative = None

        if use_int_style == True:
            if int_style_pos != 'None':
                additional_positive = self.default_pos[int_style_pos]['positive'].strip(' ,;')
            if int_style_neg != 'None':
                additional_negative = self.default_neg[int_style_neg]['negative'].strip(' ,;')

        additional_positive = f'({additional_positive}:{int_style_pos_strength:.2f})' if additional_positive is not None and additional_positive != '' else ''
        additional_negative = f'({additional_negative}:{int_style_neg_strength:.2f})' if additional_negative is not None and additional_negative != '' else ''

        negative_prompt = f'({negative_prompt}:{negative_strength:.2f})' if negative_prompt is not None and negative_prompt.strip(' ,;') != '' else ''

        opt_pos_prompt = f'({opt_pos_prompt}:{opt_pos_strength:.2f})' if opt_pos_prompt is not None and opt_pos_prompt.strip(' ,;') != '' else ''
        opt_neg_prompt = f'({opt_neg_prompt}:{opt_neg_strength:.2f})' if opt_neg_prompt is not None and opt_neg_prompt.strip(' ,;') != '' else ''

        if style_pos_strength != 1:
            style_pos_prompt = f'({style_pos_prompt}:{style_pos_strength:.2f})' if style_pos_prompt is not None and style_pos_prompt.strip(' ,;') != '' else ''
        else:
            style_pos_prompt = f'{style_pos_prompt}' if style_pos_prompt is not None and style_pos_prompt.strip(' ,;') != '' else ''

        if style_neg_prompt != 1:
            style_neg_prompt = f'({style_neg_prompt}:{style_neg_strength:.2f})' if style_neg_prompt is not None and style_neg_prompt.strip(' ,;') != '' else ''
        else:
            style_neg_prompt = f'{style_neg_prompt}' if style_neg_prompt is not None and style_neg_prompt.strip(' ,;') != '' else ''

        if (style_pos_prompt is not None and style_pos_prompt != '') or (style_neg_prompt is not None and style_neg_prompt != '') or model_concept != "Normal":
            copy_prompt_to_l = False

        if copy_prompt_to_l == True:
            sdxl_positive_l = positive_prompt
            sdxl_negative_l = negative_prompt

        if sdxl_l_strength != 1:
            sdxl_positive_l = f'({sdxl_positive_l}:{sdxl_l_strength:.2f})'.replace(":1.00", "") if sdxl_positive_l is not None and sdxl_positive_l.strip(' ,;') != '' else ''
            sdxl_negative_l = f'({sdxl_negative_l}:{sdxl_l_strength:.2f})'.replace(":1.00", "") if sdxl_negative_l is not None and sdxl_negative_l.strip(' ,;') != '' else ''
        else:
            sdxl_positive_l = f'{sdxl_positive_l}'.replace(":1.00", "") if sdxl_positive_l is not None and sdxl_positive_l.strip(' ,;') != '' else ''
            sdxl_negative_l = f'{sdxl_negative_l}'.replace(":1.00", "") if sdxl_negative_l is not None and sdxl_negative_l.strip(' ,;') != '' else ''

        if (style_pos_prompt.startswith('((') and style_pos_prompt.endswith('))')):
            style_pos_prompt = '(' + style_pos_prompt.strip('()') + ')'

        if (style_neg_prompt.startswith('((') and style_neg_prompt.endswith('))')):
            style_neg_prompt = '(' + style_neg_prompt.strip('()') + ')'

        if style_position == False:
            positive_text = f'{positive_prompt}, {opt_pos_prompt}, {style_pos_prompt}, {additional_positive}'.strip(' ,;').replace(", , ", ", ").replace(", , ", ", ").replace(":1.00", "")
            negative_text = f'{negative_prompt}, {opt_neg_prompt}, {style_neg_prompt}, {additional_negative}'.strip(' ,;').replace(", , ", ", ").replace(", , ", ", ").replace(":1.00", "")
        else:
            positive_text = f'{style_pos_prompt}, {opt_pos_prompt}, {positive_prompt}, {additional_positive}'.strip(' ,;').replace(", , ", ", ").replace(", , ", ", ").replace(":1.00", "")
            negative_text = f'{style_neg_prompt}, {opt_neg_prompt}, {negative_prompt}, {additional_negative}'.strip(' ,;').replace(", , ", ", ").replace(", , ", ", ").replace(":1.00", "")

        if model_keywords is not None:
            mkw_list = list(filter(None, model_keywords))
            if len(mkw_list) == 2:
                model_keyword = mkw_list[0]
                mplacement = mkw_list[1]
                if (mplacement == 'First'):
                    positive_text = model_keyword + ', ' + positive_text
                else:
                    positive_text = positive_text + ', ' + model_keyword

        if lora_keywords is not None:
            lkw_list = list(filter(None, lora_keywords))
            if len(lkw_list) == 2:
                lora_keyword = lkw_list[0]
                lplacement = lkw_list[1]
                if (lplacement == 'First'):
                    positive_text = lora_keyword + ', ' + positive_text
                else:
                    positive_text = positive_text + ', ' + lora_keyword

        if lycoris_keywords is not None:
            lykw_list = list(filter(None, lycoris_keywords))
            if len(lykw_list) == 2:
                lyco_keyword = lykw_list[0]
                lyplacement = lykw_list[1]
                if (lyplacement == 'First'):
                    positive_text = lyco_keyword + ', ' + positive_text
                else:
                    positive_text = positive_text + ', ' + lyco_keyword

        if embedding_pos is not None:
            embp_list = list(filter(None, embedding_pos))
            if len(embp_list) == 2:
                embp_keyword = embp_list[0]
                embp_placement = embp_list[1]
                if (embp_placement == 'First'):
                    positive_text = embp_keyword + ', ' + positive_text
                else:
                    positive_text = positive_text + ', ' + embp_keyword

        if embedding_neg is not None:
            embn_list = list(filter(None, embedding_neg))
            if len(embn_list) == 2:
                embn_keyword = embn_list[0]
                embn_placement = embn_list[1]
                if (embn_placement == 'First'):
                    negative_text = embn_keyword + ', ' + negative_text
                else:
                    negative_text = negative_text + ', ' + embn_keyword

        if model_concept == 'KwaiKolors' or model_version == 'SD1' or model_concept == 'StableCascade' or model_concept == 'Lightning':
            adv_encode = False

        if model_concept == 'Flux':
            adv_encode = False
            clip_model = 'Default'
            # clip_mode = True
            last_layer = 0


        WORKFLOWDATA = extra_pnginfo['workflow']['nodes']
        CONCEPT_SELECTOR = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereModelConceptSelector', 'model_concept', prompt)
        print(CONCEPT_SELECTOR)
        print(model_concept)

        if CONCEPT_SELECTOR == 'Flux' and (model_concept == 'Flux' and model_concept is not None):
            adv_encode = False
            clip_model = 'Default'
            # clip_mode = True
            last_layer = 0

        if model_concept == 'Hunyuan':
            last_layer = 0
            print('Clip encoding Hunyuan start:')
            if clip['t5'] is not None:
                print('HY T5 cllipping...')
                CLIPDIT = clip['clip']
                CLIPT5 = clip['t5']

                positive_text = utility.clear_hunyuan(positive_text, 0)
                negative_text = utility.clear_hunyuan(negative_text, 0)

                pos_out = clipping.HunyuanClipping(self, positive_text, "", CLIPDIT, CLIPT5)
                neg_out = clipping.HunyuanClipping(self, negative_text, "", CLIPDIT, CLIPT5)
                print('clipping end 2 - sampling T5 OUT')
                return (pos_out[0], neg_out[0], positive_text, negative_text, "", "", workflow_tuple)
            else:
                print('Standard cllipping...')
                clip = clip['clip']

                positive_text = utility.clear_hunyuan(positive_text, 512)
                negative_text = utility.clear_hunyuan(negative_text, 512)

                '''tokens_pos = clip.tokenize(positive_text)
                out_pos = clip.encode_from_tokens(tokens_pos, return_pooled = True, return_dict=True)

                tokens_neg = clip.tokenize(negative_text)
                out_neg = clip.encode_from_tokens(tokens_neg, return_pooled = True, return_dict=True)

                cond_pos = out_pos.pop("cond")
                cond_neg = out_neg.pop("cond")
                print('clipping end 2 - sampling OUT')
                return ([[cond_pos, out_pos]], [[cond_neg, out_neg]], positive_text, negative_text, "", "", workflow_tuple)'''


        if model_concept == 'KwaiKolors':
            print('Clip encoding KWAI start:')
            device = model_management.get_torch_device()
            offload_device = model_management.unet_offload_device()
            model_management.unload_all_models()
            model_management.soft_empty_cache()
            tokenizer = clip['tokenizer']
            text_encoder = clip['text_encoder']
            text_encoder.to(device)

            positive_text = utility.clear_cascade(positive_text)
            negative_text = utility.clear_cascade(negative_text)

            text_inputs = tokenizer(positive_text, padding="max_length", max_length=256, truncation=True, return_tensors="pt",).to(device)
            output = text_encoder(input_ids=text_inputs['input_ids'], attention_mask=text_inputs['attention_mask'], position_ids=text_inputs['position_ids'], output_hidden_states=True)
            prompt_embeds = output.hidden_states[-2].permute(1, 0, 2).clone() # [batch_size, 77, 4096]
            text_proj = output.hidden_states[-1][-1, :, :].clone() # [batch_size, 4096]
            bs_embed, seq_len, _ = prompt_embeds.shape

            num_images_per_prompt = 1
            batch_size = 1
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

            uncond_tokens = []
            uncond_tokens = [""] * batch_size
            uncond_tokens = [negative_text]
            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(uncond_tokens, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt", ).to(device)
            output = text_encoder(input_ids=uncond_input['input_ids'], attention_mask=uncond_input['attention_mask'], position_ids=uncond_input['position_ids'], output_hidden_states=True)
            print('clip out 2:')
            negative_prompt_embeds = output.hidden_states[-2].permute(1, 0, 2).clone() # [batch_size, 77, 4096]
            negative_text_proj = output.hidden_states[-1][-1, :, :].clone() # [batch_size, 4096]
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoder.dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)
            print('guidance end...')

            bs_embed = text_proj.shape[0]
            text_proj = text_proj.repeat(1, num_images_per_prompt).view(bs_embed * num_images_per_prompt, -1)
            negative_text_proj = negative_text_proj.repeat(1, num_images_per_prompt).view(bs_embed * num_images_per_prompt, -1)
            text_encoder.to(offload_device)
            model_management.soft_empty_cache()
            gc.collect()
            kolors_embeds = {
                'prompt_embeds': prompt_embeds,
                'negative_prompt_embeds': negative_prompt_embeds,
                'pooled_prompt_embeds': text_proj,
                'negative_pooled_prompt_embeds': negative_text_proj
            }
            print('Kolors clipping end...')
            return (kolors_embeds, None, positive_text, negative_text, "", "", workflow_tuple)

        print('*************')
        print(clip_mode)
        if clip_mode == False:
            if longclip_model == 'Default':
                longclip_model = 'longclip-L.pt'

            LONGCLIPL_PATH = os.path.join(comfy_dir, 'models', 'clip')
            if os.path.exists(LONGCLIPL_PATH) == False:
                Path(LONGCLIPL_PATH).mkdir(parents=True, exist_ok=True)
            clipFiles = folder_paths.get_filename_list("clip")

            if longclip_model not in clipFiles and longclip_model == 'Default':
                FileUrl = 'https://huggingface.co/BeichenZhang/LongCLIP-L/resolve/main/longclip-L.pt?download=true'
                FullFilePath = os.path.join(LONGCLIPL_PATH, 'longclip-L.pt')
                ModelDownload = utility.downloader(FileUrl, FullFilePath)
                if (ModelDownload == True):
                    clipFiles = folder_paths.get_filename_list("clip")

            if longclip_model in clipFiles:
                if model_concept == 'Normal' and (CONCEPT_SELECTOR == 'Normal' or CONCEPT_SELECTOR is None):
                    if (is_sdxl == 0):
                        clip = long_clip.SDLongClip.sd_longclip(self, longclip_model)[0]
                        adv_encode = False
                    else:
                        clip = long_clip.SDXLLongClip.sdxl_longclip(self, longclip_model, clip)[0]
                if model_concept == 'Flux' and (CONCEPT_SELECTOR == 'Flux' or CONCEPT_SELECTOR is None):
                    clip = long_clip.FluxLongClip.flux_longclip(self, longclip_model, clip)[0]

        if (last_layer < 0):
            clip = nodes.CLIPSetLastLayer.set_last_layer(self, clip, last_layer)[0]

        if workflow_tuple is not None:
            workflow_tuple['prompt_encoder'] = {}
            # workflow_tuple['prompt_encoder']['positive_prompt'] = positive_prompt
            # workflow_tuple['prompt_encoder']['negative_prompt'] = negative_prompt
            workflow_tuple['prompt_encoder']['clip_model'] = clip_model
            workflow_tuple['prompt_encoder']['longclip_model'] = longclip_model
            workflow_tuple['prompt_encoder']['clip_mode'] = clip_mode
            workflow_tuple['prompt_encoder']['last_layer'] = last_layer
            workflow_tuple['prompt_encoder']['negative_strength'] = negative_strength
            workflow_tuple['prompt_encoder']['use_int_style'] = use_int_style
            if use_int_style == True and int_style_pos != 'None':
                workflow_tuple['prompt_encoder']['int_style_pos'] = int_style_pos
                workflow_tuple['prompt_encoder']['int_style_pos_strength'] = int_style_pos_strength
            if use_int_style == True and int_style_neg != 'None':
                workflow_tuple['prompt_encoder']['int_style_neg'] = int_style_neg
                workflow_tuple['prompt_encoder']['int_style_neg_strength'] = int_style_neg_strength
            workflow_tuple['prompt_encoder']['adv_encode'] = adv_encode
            workflow_tuple['prompt_encoder']['token_normalization'] = token_normalization
            workflow_tuple['prompt_encoder']['weight_interpretation'] = weight_interpretation

            workflow_tuple['prompt_encoder']['model_keywords'] = model_keywords
            workflow_tuple['prompt_encoder']['lora_keywords'] = lora_keywords
            workflow_tuple['prompt_encoder']['lycoris_keywords'] = lycoris_keywords
            workflow_tuple['prompt_encoder']['embedding_pos'] = embedding_pos
            workflow_tuple['prompt_encoder']['embedding_neg'] = embedding_neg

            workflow_tuple['prompt_encoder']['opt_pos_prompt'] = opt_pos_prompt
            workflow_tuple['prompt_encoder']['opt_pos_strength'] = opt_pos_strength
            workflow_tuple['prompt_encoder']['opt_neg_prompt'] = opt_neg_prompt
            workflow_tuple['prompt_encoder']['opt_neg_strength'] = opt_neg_strength

            workflow_tuple['prompt_encoder']['style_position'] = style_position
            workflow_tuple['prompt_encoder']['style_pos_prompt'] = style_pos_prompt
            workflow_tuple['prompt_encoder']['style_pos_strength'] = style_pos_strength
            workflow_tuple['prompt_encoder']['style_neg_prompt'] = style_neg_prompt
            workflow_tuple['prompt_encoder']['style_neg_strength'] = style_neg_strength

            if copy_prompt_to_l == False:
                workflow_tuple['prompt_encoder']['sdxl_positive_l'] = sdxl_positive_l
                workflow_tuple['prompt_encoder']['sdxl_negative_l'] = sdxl_negative_l
            workflow_tuple['prompt_encoder']['copy_prompt_to_l'] = copy_prompt_to_l
            workflow_tuple['prompt_encoder']['sdxl_l_strength'] = sdxl_l_strength

        if model_concept == 'StableCascade':
            print('---Cascade')
            positive_text = utility.clear_cascade(positive_text)
            negative_text = utility.clear_cascade(negative_text)

        print(clip_model)
        if clip_model != 'Default' and clip_mode == True:
            if is_sdxl == 1:
                # adv_encode = False
                clip_model_g = 'clip_g.safetensors'
                clip_g_path = folder_paths.get_full_path("clip", clip_model_g)
                if clip_g_path is not None:
                    if model_concept == 'Flux':
                        concept_type = 'flux'
                    else:
                        concept_type = 'sdxl'
                    clip = nodes.DualCLIPLoader.load_clip(self, clip_model, clip_model_g, concept_type)[0]
            else:
                clip_path = folder_paths.get_full_path("clip", clip_model)
                if clip_path is not None:
                    if model_concept == 'StableCascade':
                        concept_type = 'stable_cascade'
                    else:
                        concept_type = 'stable_diffusion'
                    clip = nodes.CLIPLoader.load_clip(self, clip_model, concept_type)[0]

        print(adv_encode)
        if adv_encode == True:
            tokens_p = clip.tokenize(positive_text)
            tokens_n = clip.tokenize(negative_text)
            if is_sdxl == 0 or 'l' not in tokens_p or 'g' not in tokens_p or 'l' not in tokens_n or 'g' not in tokens_n:
                embeddings_final_pos, pooled_pos = advanced_encode(clip, positive_text, token_normalization, weight_interpretation, w_max = 1.0, apply_to_pooled = True)
                embeddings_final_neg, pooled_neg = advanced_encode(clip, negative_text, token_normalization, weight_interpretation, w_max = 1.0, apply_to_pooled = True)

                return ([[embeddings_final_pos, {"pooled_output": pooled_pos}]], [[embeddings_final_neg, {"pooled_output": pooled_neg}]], positive_text, negative_text, "", "", workflow_tuple)
            else:
                # tokens_p = clip.tokenize(positive_text)
                if 'l' in clip.tokenize(sdxl_positive_l):
                    tokens_p["l"] = clip.tokenize(sdxl_positive_l)["l"]
                    if len(tokens_p["l"]) != len(tokens_p["g"]):
                        empty = clip.tokenize("")
                        while len(tokens_p["l"]) < len(tokens_p["g"]):
                            tokens_p["l"] += empty["l"]
                        while len(tokens_p["l"]) > len(tokens_p["g"]):
                            tokens_p["g"] += empty["g"]

                # tokens_n = clip.tokenize(negative_text)
                if 'l' in clip.tokenize(sdxl_negative_l):
                    tokens_n["l"] = clip.tokenize(sdxl_negative_l)["l"]

                    if len(tokens_n["l"]) != len(tokens_n["g"]):
                        empty = clip.tokenize("")
                        while len(tokens_n["l"]) < len(tokens_n["g"]):
                            tokens_n["l"] += empty["l"]
                        while len(tokens_n["l"]) > len(tokens_n["g"]):
                            tokens_n["g"] += empty["g"]

                cond_p, pooled_p = clip.encode_from_tokens(tokens_p, return_pooled = True)
                cond_n, pooled_n = clip.encode_from_tokens(tokens_n, return_pooled = True)

                return ([[cond_p, {"pooled_output": pooled_p, "width": width, "height": height, "crop_w": 0, "crop_h": 0, "target_width": width, "target_height": height}]], [[cond_n, {"pooled_output": pooled_n, "width": width, "height": height, "crop_w": 0, "crop_h": 0, "target_width": width, "target_height": height}]], positive_text, negative_text, sdxl_positive_l, sdxl_negative_l, workflow_tuple)

        else:
            if clip_mode == True:
                if model_concept == 'Flux':
                    WORKFLOWDATA = extra_pnginfo['workflow']['nodes']
                    FLUX_SAMPLER = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereModelConceptSelector', 'flux_sampler', prompt)
                    if FLUX_SAMPLER == 'ksampler':
                        FLUX_GUIDANCE = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereModelConceptSelector', 'flux_clip_guidance', prompt)
                        if FLUX_GUIDANCE is None:
                            FLUX_GUIDANCE = 1.7
                        CONDITIONING_POS = nodes_flux.CLIPTextEncodeFlux.encode(self, clip, positive_text, positive_text, FLUX_GUIDANCE)[0]
                        if workflow_tuple is not None and 'cfg' in workflow_tuple and int(workflow_tuple['cfg']) < 1.2:
                            CONDITIONING_NEG = CONDITIONING_POS
                        else:
                            CONDITIONING_NEG = nodes_flux.CLIPTextEncodeFlux.encode(self, clip, negative_text, negative_text, FLUX_GUIDANCE)[0]
                        return (CONDITIONING_POS, CONDITIONING_NEG, positive_text, negative_text, "", "", workflow_tuple)

            if model_concept == 'StableCascade':
                positive_text = utility.clear_cascade(positive_text)
                negative_text = utility.clear_cascade(negative_text)

            print('clipping end 1 ...')

            tokens_pos = clip.tokenize(positive_text)
            # cond_pos, pooled_pos = clip.encode_from_tokens(tokens, return_pooled = True)
            out_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True, return_dict=True)

            tokens_neg = clip.tokenize(negative_text)
            # cond_neg, pooled_neg = clip.encode_from_tokens(tokens, return_pooled = True)
            out_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True, return_dict=True)

            cond_pos = out_pos.pop("cond")
            cond_neg = out_neg.pop("cond")

            print('clipping end 2 - sampling')
            # return ([[cond_pos, {"pooled_output": pooled_pos}]], [[cond_neg, {"pooled_output": pooled_neg}]], positive_text, negative_text, "", "", workflow_tuple)
            return ([[cond_pos, out_pos]], [[cond_neg, out_neg]], positive_text, negative_text, "", "", workflow_tuple)

class PrimereResolution:
    RETURN_TYPES = ("INT", "INT", "INT")
    RETURN_NAMES = ("WIDTH", "HEIGHT", "SQUARE_SHAPE")
    FUNCTION = "calculate_imagesize"
    CATEGORY = TREE_DASHBOARD

    @staticmethod
    def get_ratios(toml_path: str):
        with open(toml_path, "rb") as f:
            image_ratios = tomli.load(f)
        return image_ratios

    @ classmethod
    def INPUT_TYPES(cls):
        DEF_TOML_DIR = os.path.join(PRIMERE_ROOT, 'Toml')
        cls.sd_ratios = cls.get_ratios(os.path.join(DEF_TOML_DIR, "resolution_ratios.toml"))

        namelist = {}
        for sd_ratio_key in cls.sd_ratios:
            rationName = cls.sd_ratios[sd_ratio_key]['name']
            namelist[rationName] = sd_ratio_key

        cls.ratioNames = namelist

        return {
            "required": {
                "ratio": (list(namelist.keys()),),
                "resolution": ("BOOLEAN", {"default": True, "label_on": "Auto", "label_off": "Manual"}),
                "sd1_res": (utility.VALID_SHAPES, {"default": utility.VALID_SHAPES[1]}),
                "sdxl_res": (utility.VALID_SHAPES, {"default": utility.VALID_SHAPES[2]}),
                "turbo_res": (utility.VALID_SHAPES, {"default": utility.VALID_SHAPES[0]}),
                "rnd_orientation": ("BOOLEAN", {"default": False}),
                "orientation": (["Horizontal", "Vertical"], {"default": "Horizontal"}),
                "round_to_standard": ("BOOLEAN", {"default": False}),

                "calculate_by_custom": ("BOOLEAN", {"default": False}),
                "custom_side_a": ("FLOAT", {"default": 1.6, "min": 1.0, "max": 100.0, "step": 0.05}),
                "custom_side_b": ("FLOAT", {"default": 2.8, "min": 1.0, "max": 100.0, "step": 0.05}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff, "forceInput": True}),
                "model_version": ("STRING", {"default": 'SD1', "forceInput": True}),
                "model_concept": ("STRING", {"default": "Auto", "forceInput": True}),
            }
        }

    def calculate_imagesize(self, ratio: str, resolution: bool, rnd_orientation: bool, orientation: str, round_to_standard: bool, calculate_by_custom: bool, custom_side_a: float, custom_side_b: float, seed: int = 0, model_version: str = "SD1", model_concept: str = 'Auto', **kwargs):
        square_shape = kwargs['sdxl_res']

        if seed < 1:
            seed = random.randint(0, 9)

        if rnd_orientation == True:
            if (seed % 2) == 0:
                orientation = "Horizontal"
            else:
                orientation = "Vertical"

        if resolution == True:
            square_shape = utility.getResolutionByType(model_version)
        else:
            input_string = model_version.lower() + '_res'
            if input_string in kwargs:
                square_shape = int(kwargs[input_string])

        if square_shape is None:
            square_shape = kwargs['sdxl_res']

        standard = 'STANDARD'
        if model_version == 'StableCascade':
            standard = 'CASCADE'

        dimensions = utility.get_dimensions_by_shape(self, ratio, square_shape, orientation, round_to_standard, calculate_by_custom, custom_side_a, custom_side_b, standard)
        dimension_x = dimensions[0]
        dimension_y = dimensions[1]

        # MODEL_SHAPES = {'SD1': sd1_res, 'SDXL': sdxl_res, 'TURBO': turbo_res}
        return dimension_x, dimension_y, square_shape

class PrimereResolutionMultiplierMPX:
    RETURN_TYPES = ("INT", "INT", "FLOAT", "IMAGE")
    RETURN_NAMES = ("WIDTH", "HEIGHT", "UPSCALE_RATIO", "IMAGE")
    FUNCTION = "multiply_imagesize_mpx"
    CATEGORY = TREE_DASHBOARD
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_multiplier": ("BOOLEAN", {"default": True}),
                "width": ('INT', {"forceInput": True, "default": 512}),
                "height": ('INT', {"forceInput": True, "default": 512}),
                "upscale_to_mpx": ("FLOAT", {"default": 12.00, "min": 0.01, "max": 48.00, "step": 0.01}),
            },
            "optional": {
                "triggered_prescale": ("BOOLEAN", {"default": False}),
                "image": ("IMAGE", {"forceInput": True}),
                "area_trigger_mpx": ("FLOAT", {"default": 0.60, "min": 0.01, "max": round(pow(utility.MAX_RESOLUTION, 2) / 1000000, 2), "step": 0.01}),
                "area_target_mpx": ("FLOAT", {"default": 1.05, "min": 0.25, "max": round(pow(utility.MAX_RESOLUTION, 2) / 1000000, 2), "step": 0.01}),
                "upscale_model": (['None'] + folder_paths.get_filename_list("upscale_models"), {"default": 'None'}),
                "upscale_method": (cls.upscale_methods, {"default": 'bicubic'}),
            }
        }

    def multiply_imagesize_mpx(self, width: int, height: int, use_multiplier: bool, upscale_to_mpx: int, triggered_prescale = False, image = None, area_trigger_mpx = 0.60, area_target_mpx = 1.05, upscale_model = 'None', upscale_method = 'bicubic'):
        if use_multiplier == False or upscale_to_mpx < 0.01:
            return (width, height, 1, image)

        if image is not None:
            width = image.shape[2]
            height = image.shape[1]

        if triggered_prescale == True and use_multiplier == True:
            area_trigger = area_trigger_mpx * (1000 * 1000)
            area_target = area_target_mpx * (1024 * 1024)
            area = width * height
            if area_trigger >= area:
                sourceMPXTrigger = area
                differenceTrigger = area_target / sourceMPXTrigger
                squareDiffTrigger = math.sqrt(differenceTrigger)
                if image is not None:
                    if upscale_model == 'None':
                        prescaledImage = nodes.ImageScaleBy.upscale(self, image, upscale_method, squareDiffTrigger)[0]
                    else:
                        loaded_upscale_model = nodes_upscale_model.UpscaleModelLoader.load_model(self, upscale_model)[0]
                        prescaledImage = nodes_upscale_model.ImageUpscaleWithModel.upscale(self, loaded_upscale_model, image)[0]

                    image = prescaledImage
                    width = prescaledImage.shape[2]
                    height = prescaledImage.shape[1]
                    newArea = width * height

                    if newArea > area_target:
                        differenceTrigger = area_target / newArea
                        squareDiffTrigger = math.sqrt(differenceTrigger)
                        prescaledImage = nodes.ImageScaleBy.upscale(self, image, upscale_method, squareDiffTrigger)[0]
                        image = prescaledImage
                        width = prescaledImage.shape[2]
                        height = prescaledImage.shape[1]

        sourceMPX = (width * height) / (1024 * 1024)
        difference = upscale_to_mpx / sourceMPX
        squareDiff = math.sqrt(difference)
        dimension_x = round(width * squareDiff)
        dimension_y = round(height * squareDiff)
        ratio = round(squareDiff, 2)

        return (dimension_x, dimension_y, ratio, image)

class PrimereResolutionCoordinatorMPX:
    RETURN_TYPES = ("INT", "INT", "INT", "INT", "IMAGE", "INT", "INT", "IMAGE", "INT", "INT")
    RETURN_NAMES = ("REF_WIDTH", "REF_HEIGHT", "SLAVE_WIDTH", "SLAVE_HEIGHT", "RESIZED_REFERENCE", "REF_WIDTH_RES", "REF_HEIGHT_RES", "RESIZED_SLAVE", "SLAVE_WIDTH_RES", "SLAVE_HEIGHT_RES")
    FUNCTION = "imagesize_coordinator"
    CATEGORY = TREE_DASHBOARD
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_resizer": ("BOOLEAN", {"default": True}),
                "reference_image": ("IMAGE", {"forceInput": True}),
                "slave_image": ("IMAGE", {"forceInput": True}),
                "resize_to_mpx": ("FLOAT", {"default": 1.00, "min": 0.01, "max": 48.00, "step": 0.01}),
                "keep_slave_ratio": ("BOOLEAN", {"default": False}),
                "upscale_model": (['None'] + folder_paths.get_filename_list("upscale_models"), {"default": 'None'}),
                "upscale_method": (cls.upscale_methods, {"default": 'lanczos'}),
            }
        }

    def imagesize_coordinator(self, use_resizer, reference_image, slave_image, resize_to_mpx, keep_slave_ratio, upscale_model, upscale_method):
        ref_width = 0
        ref_height = 0
        slave_width = 0
        slave_height = 0

        if reference_image is not None and slave_image is not None:
            ref_width = reference_image.shape[2]
            ref_height = reference_image.shape[1]
            slave_width = slave_image.shape[2]
            slave_height = slave_image.shape[1]

        if use_resizer == True:
            referenceMPX = (ref_width * ref_height) / (1024 * 1024)
            referenceDifference = resize_to_mpx / referenceMPX
            ref_squareDiff = math.sqrt(referenceDifference)
            ref_width_resized = round(ref_width * ref_squareDiff)
            ref_height_resized = round(ref_height * ref_squareDiff)
            slaveMPX = (slave_width * slave_height) / (1024 * 1024)
            slaveDifference = resize_to_mpx / slaveMPX
            slave_squareDiff = math.sqrt(slaveDifference)

            if keep_slave_ratio == True:
                slave_width_resized = round(slave_width * slave_squareDiff)
                slave_height_resized = round(slave_height * slave_squareDiff)
            else:
                slave_width_resized = ref_width_resized
                slave_height_resized = ref_height_resized

            if upscale_model == 'None':
                reference_image = nodes.ImageScaleBy.upscale(self, reference_image, upscale_method, ref_squareDiff)[0]
                if keep_slave_ratio == True:
                    slave_image = nodes.ImageScaleBy.upscale(self, slave_image, upscale_method, slave_squareDiff)[0]
                else:
                    slave_image = nodes.ImageScale.upscale(self, slave_image, upscale_method, slave_width_resized, slave_height_resized, "disabled")[0]
            else:
                loaded_upscale_model = nodes_upscale_model.UpscaleModelLoader.load_model(self, upscale_model)[0]

                reference_image_model = nodes_upscale_model.ImageUpscaleWithModel.upscale(self, loaded_upscale_model, reference_image)[0]
                reference_image = nodes.ImageScale.upscale(self, reference_image_model, upscale_method, ref_width_resized, ref_height_resized, "disabled")[0]

                if keep_slave_ratio == True:
                    slave_image_model = nodes_upscale_model.ImageUpscaleWithModel.upscale(self, loaded_upscale_model, slave_image)[0]
                    slave_image = nodes.ImageScaleBy.upscale(self, slave_image_model, upscale_method, slave_squareDiff)[0]

                slave_image = nodes.ImageScale.upscale(self, slave_image, upscale_method, slave_width_resized, slave_height_resized, "disabled")[0]

        else:
            ref_width_resized = ref_width
            ref_height_resized = ref_height
            slave_width_resized = slave_width
            slave_height_resized = slave_height

        return (ref_width, ref_height, slave_width, slave_height, reference_image, ref_width_resized, ref_height_resized, slave_image, slave_width_resized, slave_height_resized)

class PrimereClearPrompt:
  RETURN_TYPES = ("STRING", "STRING")
  RETURN_NAMES = ("PROMPT+", "PROMPT-")
  FUNCTION = "clean_prompt"
  CATEGORY = TREE_DASHBOARD

  @classmethod
  def INPUT_TYPES(cls):
      return {
          "required": {
              "model_version": ("STRING", {"default": 'BaseModel_1024', "forceInput": True}),
              "positive_prompt": ("STRING", {"forceInput": True}),
              "negative_prompt": ("STRING", {"forceInput": True}),
              "remove_only_if_sdxl": ("BOOLEAN", {"default": False}),
              "remove_comfy_embedding": ("BOOLEAN", {"default": False}),
              "remove_a1111_embedding": ("BOOLEAN", {"default": False}),
              "remove_lora": ("BOOLEAN", {"default": False}),
              "remove_lycoris": ("BOOLEAN", {"default": False}),
              "remove_hypernetwork": ("BOOLEAN", {"default": False}),
          },
          "optional": {
              "model_concept": ("STRING", {"default": "Normal", "forceInput": True}),
          }
      }

  def clean_prompt(self, positive_prompt, negative_prompt, remove_comfy_embedding, remove_a1111_embedding, remove_lora, remove_lycoris, remove_hypernetwork, remove_only_if_sdxl, model_version = 'BaseModel_1024', model_concept = "Normal"):
      NETWORK_START = []

      is_sdxl = 0
      match model_version:
          case 'SDXL_2048':
              is_sdxl = 1

      if remove_only_if_sdxl == True and is_sdxl == 0:
          return (positive_prompt, negative_prompt,)

      if remove_comfy_embedding == True:
          NETWORK_START.append('embedding:')

      if remove_lora == True:
          NETWORK_START.append('<lora:')

      if remove_lycoris == True:
          NETWORK_START.append('<lyco:')

      if remove_hypernetwork == True:
          NETWORK_START.append('<hypernet:')

      if remove_a1111_embedding == True:
          positive_prompt = positive_prompt.replace('embedding:', '')
          negative_prompt = negative_prompt.replace('embedding:', '')
          EMBEDDINGS = folder_paths.get_filename_list("embeddings")
          for embeddings_path in EMBEDDINGS:
              path = Path(embeddings_path)
              embedding_name = path.stem
              positive_prompt = re.sub("(\(" + embedding_name + ":d+\.d+\))|(\(" + embedding_name + ":d+\))|(" + embedding_name + ":d+\.d+)|(" + embedding_name + ":d+)|(" + embedding_name + ":)|(\(" + embedding_name + "\))|(" + embedding_name + ")", "", positive_prompt)
              negative_prompt = re.sub("(\(" + embedding_name + ":d+\.d+\))|(\(" + embedding_name + ":d+\))|(" + embedding_name + ":d+\.d+)|(" + embedding_name + ":d+)|(" + embedding_name + ":)|(\(" + embedding_name + "\))|(" + embedding_name + ")", "", negative_prompt)
              positive_prompt = re.sub(r'(, )\1+', r', ', positive_prompt).strip(', ').replace(' ,', ',')
              negative_prompt = re.sub(r'(, )\1+', r', ', negative_prompt).strip(', ').replace(' ,', ',')

      if len(NETWORK_START) > 0:
         NETWORK_END = ['\n', '>', ' ', ',', '}', ')', '|'] + NETWORK_START
         positive_prompt = utility.clear_prompt(NETWORK_START, NETWORK_END, positive_prompt)
         negative_prompt = utility.clear_prompt(NETWORK_START, NETWORK_END, negative_prompt)

      if model_concept == 'StableCascade':
          positive_prompt = utility.clear_cascade(positive_prompt)
          negative_prompt = utility.clear_cascade(negative_prompt)

      return (positive_prompt, negative_prompt,)

class PrimereNetworkTagLoader:
  RETURN_TYPES = ("MODEL", "CLIP", "LORA_STACK", "LYCORIS_STACK", "HYPERNETWORK_STACK", "MODEL_KEYWORD", "MODEL_KEYWORD")
  RETURN_NAMES = ("MODEL", "CLIP", "LORA_STACK", "LYCORIS_STACK", "HYPERNETWORK_STACK", "LORA_KEYWORD", "LYCORIS_KEYWORD")
  FUNCTION = "load_networks"
  CATEGORY = TREE_DASHBOARD
  LORASCOUNT = PrimereLORA.LORASCOUNT
  EMBCOUNT = PrimereEmbedding.EMBCOUNT
  HNCOUNT = PrimereHypernetwork.HNCOUNT
  LYCOSCOUNT = PrimereLYCORIS.LYCOSCOUNT

  @classmethod
  def INPUT_TYPES(s):
      return {
          "required": {
              "model": ("MODEL",),
              "clip": ("CLIP",),
              "positive_prompt": ("STRING", {"forceInput": True}),
              "process_lora": ("BOOLEAN", {"default": True}),
              "process_lycoris": ("BOOLEAN", {"default": True}),
              "process_hypernetwork": ("BOOLEAN", {"default": True}),
              "hypernetwork_safe_load": ("BOOLEAN", {"default": True}),
              "copy_weight_to_clip": ("BOOLEAN", {"default": False}),
              "lora_clip_custom_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),
              "lycoris_clip_custom_weight": ("FLOAT", {"default": 1.0, "min": -10.0, "max": 10.0, "step": 0.01}),

              "use_lora_keyword": ("BOOLEAN", {"default": False}),
              "lora_keyword_placement": (["First", "Last"], {"default": "Last"}),
              "lora_keyword_selection": (["Select in order", "Random select"], {"default": "Select in order"}),
              "lora_keywords_num": ("INT", {"default": 1, "min": 1, "max": 50, "step": 1}),
              "lora_keyword_weight": ("FLOAT", {"default": 1.0, "min": 0, "max": 10.0, "step": 0.1}),

              "use_lycoris_keyword": ("BOOLEAN", {"default": False}),
              "lycoris_keyword_placement": (["First", "Last"], {"default": "Last"}),
              "lycoris_keyword_selection": (["Select in order", "Random select"], {"default": "Select in order"}),
              "lycoris_keywords_num": ("INT", {"default": 1, "min": 1, "max": 50, "step": 1}),
              "lycoris_keyword_weight": ("FLOAT", {"default": 1.0, "min": 0, "max": 10.0, "step": 0.1}),
          },
          "optional": {
              "workflow_tuple": ("TUPLE", {"default": None}),
          }
      }

  def load_networks(self, model, clip, positive_prompt, process_lora, process_lycoris, process_hypernetwork, copy_weight_to_clip, lora_clip_custom_weight, lycoris_clip_custom_weight, use_lora_keyword, use_lycoris_keyword, lora_keyword_placement, lycoris_keyword_placement, lora_keyword_selection, lycoris_keyword_selection, lora_keywords_num, lycoris_keywords_num, lora_keyword_weight, lycoris_keyword_weight, hypernetwork_safe_load = True, workflow_tuple = None):
      if workflow_tuple is not None and len(workflow_tuple) > 0 and 'setup_states' in workflow_tuple and 'exif_status' in workflow_tuple and workflow_tuple['exif_status'] == 'SUCCEED':
          concept = 'Normal'
          stack_version = 'ANY'
          if 'model_concept' in workflow_tuple:
              concept = workflow_tuple['model_concept']
          if 'model_version' in workflow_tuple:
            if concept == 'Normal' and workflow_tuple['model_version'] == 'SDXL_2048':
                stack_version = 'SDXL'
            else:
                stack_version = 'SD'

          if 'setup_states' in workflow_tuple and 'network_data' in workflow_tuple:
              if 'lora_setup' in workflow_tuple['setup_states'] and workflow_tuple['setup_states']['lora_setup'] == True:
                  loader = networkhandler.getNetworkLoader(workflow_tuple, 'lora', self.LORASCOUNT, True, stack_version)
                  if len(loader) > 0:
                    networkData = networkhandler.LoraHandler(self, loader, model, clip, [], False, lora_keywords_num, use_lora_keyword, lora_keyword_selection, lora_keyword_weight, lora_keyword_placement)
                    model = networkData[0]
                    clip = networkData[1]

              if 'lycoris_setup' in workflow_tuple['setup_states'] and workflow_tuple['setup_states']['lycoris_setup'] == True:
                  loader = networkhandler.getNetworkLoader(workflow_tuple, 'lycoris', self.LYCOSCOUNT, True, stack_version)
                  if len(loader) > 0:
                    networkData = networkhandler.LycorisHandler(self, loader, model, clip, [], False, lycoris_keywords_num, use_lycoris_keyword, lycoris_keyword_selection, lycoris_keyword_weight, lycoris_keyword_placement)
                    model = networkData[0]
                    clip = networkData[1]

              if 'embedding_setup' in workflow_tuple['setup_states'] and workflow_tuple['setup_states']['embedding_setup'] == True:
                  loader = networkhandler.getNetworkLoader(workflow_tuple, 'embedding', self.EMBCOUNT, False, stack_version)
                  if len(loader) > 0:
                    networkData = networkhandler.EmbeddingHandler(self, loader, None, None)
                    if networkData[0][0] is not None:
                        positive_prompt = networkData[0][0] + ',  ' + positive_prompt
                        tokens = clip.tokenize(positive_prompt)
                        clip = clip.encode_from_tokens(tokens, return_pooled=False)

              if 'hypernetwork_setup' in workflow_tuple['setup_states'] and workflow_tuple['setup_states']['hypernetwork_setup'] == True:
                  loader = networkhandler.getNetworkLoader(workflow_tuple, 'hypernetwork', self.HNCOUNT, False, stack_version)
                  if len(loader) > 0:
                      networkData = networkhandler.HypernetworkHandler(self, loader, model, hypernetwork_safe_load)
                      model = networkData[0]

      NETWORK_START = []

      cloned_model = model
      cloned_clip = clip
      list_of_keyword_items = []
      lora_keywords_num_set = lora_keywords_num
      lycoris_keywords_num_set = lycoris_keywords_num
      model_lora_keyword = [None, None]
      model_lyco_keyword = [None, None]
      lora_stack = []
      lycoris_stack = []
      hnet_stack = []

      HypernetworkList = folder_paths.get_filename_list("hypernetworks")
      LoraList = folder_paths.get_filename_list("loras")

      LYCO_DIR = os.path.join(comfy_dir, 'models', 'lycoris')
      folder_paths.add_model_folder_path("lycoris", LYCO_DIR)
      LyCORIS = folder_paths.get_filename_list("lycoris")
      LycorisList = folder_paths.filter_files_extensions(LyCORIS, ['.ckpt', '.safetensors'])

      if process_lora == True:
        NETWORK_START.append('<lora:')

      if process_lycoris == True:
        NETWORK_START.append('<lyco:')

      if process_hypernetwork == True:
        NETWORK_START.append('<hypernet:')

      if len(NETWORK_START) == 0:
          return (model, clip, lora_stack, lycoris_stack, hnet_stack, model_lora_keyword, model_lyco_keyword)
      else:
          NETWORK_END = ['>'] + NETWORK_START
          NETWORK_TUPLE = utility.get_networks_prompt(NETWORK_START, NETWORK_END, positive_prompt)
          if (len(NETWORK_TUPLE) == 0):
              return (model, clip, lora_stack, lycoris_stack, hnet_stack, model_lora_keyword, model_lyco_keyword)
          else:
              for NETWORK_DATA in NETWORK_TUPLE:
                  NetworkName = NETWORK_DATA[0]
                  try:
                    NetworkStrenght = float(NETWORK_DATA[1])
                  except ValueError:
                    NetworkStrenght = 1
                  NetworkType = NETWORK_DATA[2]

                  if (process_lora == True and NetworkType == 'LORA'):
                      lora_name = utility.get_closest_element(NetworkName, LoraList)
                      if lora_name is not None:
                          lora_path = folder_paths.get_full_path("loras", lora_name)
                          lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
                          if (copy_weight_to_clip == True):
                              lora_clip_custom_weight = NetworkStrenght
                          lora_stack.append([lora_name, NetworkStrenght, lora_clip_custom_weight])
                          cloned_model, cloned_clip = comfy.sd.load_lora_for_models(cloned_model, cloned_clip, lora, NetworkStrenght, lora_clip_custom_weight)

                          if use_lora_keyword == True:
                              ModelKvHash = utility.get_model_hash(lora_path)
                              if ModelKvHash is not None:
                                  KEYWORD_PATH = os.path.join(PRIMERE_ROOT, 'front_end', 'keywords', 'lora-keyword.txt')
                                  keywords = utility.get_model_keywords(KEYWORD_PATH, ModelKvHash, lora_name)
                                  if keywords is not None and keywords != "" and isinstance(keywords, str) == True:
                                      if keywords.find('|') > 1:
                                          keyword_list = [word.strip() for word in keywords.split('|')]
                                          keyword_list = list(filter(None, keyword_list))
                                          if (len(keyword_list) > 0):
                                              lora_keywords_num = lora_keywords_num_set
                                              keyword_qty = len(keyword_list)
                                              if (lora_keywords_num > keyword_qty):
                                                  lora_keywords_num = keyword_qty
                                              if lora_keyword_selection == 'Select in order':
                                                  list_of_keyword_items.extend(keyword_list[:lora_keywords_num])
                                              else:
                                                  list_of_keyword_items.extend(random.sample(keyword_list, lora_keywords_num))
                                      else:
                                          list_of_keyword_items.append(keywords)

                      if len(list_of_keyword_items) > 0:
                          if lora_keyword_selection != 'Select in order':
                              random.shuffle(list_of_keyword_items)

                          list_of_keyword_items = list(set(list_of_keyword_items))
                          keywords = ", ".join(list_of_keyword_items)

                          if (lora_keyword_weight != 1):
                              keywords = '(' + keywords + ':' + str(round(lora_keyword_weight, 1)) + ')'

                          model_lora_keyword = [keywords, lora_keyword_placement]

                  if (process_lycoris == True and NetworkType == 'LYCORIS'):
                      lycoris_name = utility.get_closest_element(NetworkName, LycorisList)
                      if lycoris_name is not None:
                          lycoris_path = folder_paths.get_full_path("lycoris", lycoris_name)
                          lycoris = comfy.utils.load_torch_file(lycoris_path, safe_load=True)
                          if (copy_weight_to_clip == True):
                              lycoris_clip_custom_weight = NetworkStrenght
                          lycoris_stack.append([lycoris_name, NetworkStrenght, lycoris_clip_custom_weight])
                          cloned_model, cloned_clip = comfy.sd.load_lora_for_models(cloned_model, cloned_clip, lycoris, NetworkStrenght, lycoris_clip_custom_weight)

                          if use_lycoris_keyword == True:
                              ModelKvHash = utility.get_model_hash(lycoris_path)
                              if ModelKvHash is not None:
                                  KEYWORD_PATH = os.path.join(PRIMERE_ROOT, 'front_end', 'keywords', 'lora-keyword.txt')
                                  keywords = utility.get_model_keywords(KEYWORD_PATH, ModelKvHash, lycoris_name)
                                  if keywords is not None and keywords != "" and isinstance(keywords, str) == True:
                                      if keywords.find('|') > 1:
                                          keyword_list = [word.strip() for word in keywords.split('|')]
                                          keyword_list = list(filter(None, keyword_list))
                                          if (len(keyword_list) > 0):
                                              lycoris_keywords_num = lycoris_keywords_num_set
                                              keyword_qty = len(keyword_list)
                                              if (lycoris_keywords_num > keyword_qty):
                                                  lycoris_keywords_num = keyword_qty
                                              if lycoris_keyword_selection == 'Select in order':
                                                  list_of_keyword_items.extend(keyword_list[:lycoris_keywords_num])
                                              else:
                                                  list_of_keyword_items.extend(random.sample(keyword_list, lycoris_keywords_num))
                                      else:
                                          list_of_keyword_items.append(keywords)

                      if len(list_of_keyword_items) > 0:
                          if lycoris_keyword_selection != 'Select in order':
                              random.shuffle(list_of_keyword_items)

                          list_of_keyword_items = list(set(list_of_keyword_items))
                          keywords = ", ".join(list_of_keyword_items)

                          if (lycoris_keyword_weight != 1):
                              keywords = '(' + keywords + ':' + str(round(lycoris_keyword_weight, 1)) + ')'

                          model_lyco_keyword = [keywords, lycoris_keyword_placement]

                  if (process_hypernetwork == True and NetworkType == 'HYPERNET'):
                      hyper_name = utility.get_closest_element(NetworkName, HypernetworkList)
                      if hyper_name is not None:
                          hypernetwork_path = folder_paths.get_full_path("hypernetworks", hyper_name)
                          model_hypernetwork = cloned_model.clone()
                          try:
                            patch = hypernetwork.load_hypernetwork_patch(hypernetwork_path, NetworkStrenght, hypernetwork_safe_load)
                          except Exception:
                              patch = None
                          if patch is not None:
                              model_hypernetwork.set_model_attn1_patch(patch)
                              model_hypernetwork.set_model_attn2_patch(patch)
                              hnet_stack.append([hyper_name, NetworkStrenght])
                              cloned_model = model_hypernetwork

      return (cloned_model, cloned_clip, lora_stack, lycoris_stack, hnet_stack, model_lora_keyword, model_lyco_keyword)

class PrimereModelKeyword:
    RETURN_TYPES = ("MODEL_KEYWORD",)
    RETURN_NAMES = ("MODEL_KEYWORD",)
    FUNCTION = "load_ckpt_keyword"
    CATEGORY = TREE_DASHBOARD

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": ('CHECKPOINT_NAME', {"forceInput": True, "default": ""}),
                "use_model_keyword": ("BOOLEAN", {"default": False}),
                "model_keyword_placement": (["First", "Last"], {"default": "Last"}),
                # "model_keyword_selection": (["Select in order", "Random select"], {"default": "Select in order"}),
                "model_keywords_num": ("INT", {"default": 1, "min": 1, "max": 50, "step": 1}),
                "model_keyword_weight": ("FLOAT", {"default": 1.0, "min": 0, "max": 10.0, "step": 0.1}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT"
            },
        }

    def load_ckpt_keyword(self, model_name, use_model_keyword, model_keyword_placement, model_keywords_num, model_keyword_weight, prompt, **kwargs):
        model_keyword = [None, None]

        WORKFLOWDATA = kwargs['extra_pnginfo']['workflow']['nodes']
        # selectedKeyword = utility.getDataFromWorkflow(WORKFLOWDATA, 'PrimereModelKeyword', 4)
        selectedKeyword = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereModelKeyword', 'select_keyword', prompt)

        if use_model_keyword == True and selectedKeyword != 'None' and selectedKeyword != None:
            if selectedKeyword != 'Select in order' and selectedKeyword != 'Random select':
                if selectedKeyword.rfind('/') != -1:
                    selectedKeyword = selectedKeyword.rsplit('/', 1)[1].strip()
                if (model_keyword_weight != 1):
                    selectedKeyword = '(' + selectedKeyword + ':' + str(round(model_keyword_weight, 1)) + ')'

                model_keyword = [selectedKeyword, model_keyword_placement]
            else:
                ckpt_path = folder_paths.get_full_path("checkpoints", model_name)
                ModelKvHash = utility.get_model_hash(ckpt_path)
                if ModelKvHash is not None:
                    KEYWORD_PATH = os.path.join(PRIMERE_ROOT, 'front_end', 'keywords', 'model-keyword.txt')
                    keywords = utility.get_model_keywords(KEYWORD_PATH, ModelKvHash, model_name)

                    if keywords is not None and isinstance(keywords, str) == True:
                        if keywords.find('|') > 1:
                            keyword_list = keywords.split("|")
                            if (len(keyword_list) > 0):
                                keyword_qty = len(keyword_list)
                                if (model_keywords_num > keyword_qty):
                                    model_keywords_num = keyword_qty
                                if selectedKeyword == 'Select in order':
                                    list_of_keyword_items = keyword_list[:model_keywords_num]
                                else:
                                    list_of_keyword_items = random.sample(keyword_list, model_keywords_num)

                                clean_keywords = []
                                for keyword_item in list_of_keyword_items:
                                    if keyword_item.rfind('/') != -1:
                                        keyword_item = keyword_item.rsplit('/', 1)[1].strip()
                                    clean_keywords += [keyword_item]

                                keywords = ", ".join(clean_keywords)

                        if (model_keyword_weight != 1):
                            keywords = '(' + keywords + ':' + str(round(model_keyword_weight, 1)) + ')'

                        model_keyword = [keywords, model_keyword_placement]

        return (model_keyword,)

class PrimereUpscaleModel:
    RETURN_TYPES = ("UPSCALE_MODEL", folder_paths.get_filename_list("upscale_models"),)
    RETURN_NAMES = ("UPSCALE_MODEL", 'MODEL_NAME',)
    FUNCTION = "load_upscaler"
    CATEGORY = TREE_DASHBOARD
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (folder_paths.get_filename_list("upscale_models"), ),
            }
        }

    def load_upscaler(self, model_name):
        out = nodes_upscale_model.UpscaleModelLoader.load_model(self, model_name)[0]
        return (out, model_name,)