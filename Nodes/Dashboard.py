import math
from ..components.tree import TREE_DASHBOARD
from ..components.tree import PRIMERE_ROOT
from server import PromptServer
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
from ..components import hypernetwork
from ..components import clipping
from ..components import models as model_loaders
from ..components import nf4_helper
from ..components import sana_utils
import comfy.sd
import comfy.model_detection
import comfy.utils
import comfy_extras.nodes_model_advanced as nodes_model_advanced
import comfy_extras.nodes_upscale_model as nodes_upscale_model
import comfy_extras.nodes_cfg as nodes_cfg
import comfy_extras.nodes_qwen as nodes_qwen
from comfy import model_management
from ..components.gguf import nodes as gguf_nodes
import comfy_extras.nodes_flux as nodes_flux
import comfy_extras.nodes_sd3 as nodes_sd3
from .modules import long_clip
from .modules import networkhandler
from .Networks import PrimereLORA
from .Networks import PrimereEmbedding
from .Networks import PrimereHypernetwork
from .Networks import PrimereLYCORIS
from diffusers import (UNet2DConditionModel, EulerDiscreteScheduler)
from ..components.kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from ..components.kolors.models.tokenization_chatglm import ChatGLMTokenizer
from ..components.kolors.models.modeling_chatglm import ChatGLMModel
import gc
from ComfyUI_ExtraModels.HunYuanDiT.conf import hydit_conf
from ComfyUI_ExtraModels.HunYuanDiT.loader import load_hydit
from ComfyUI_ExtraModels.utils.dtype import string_to_dtype
from ComfyUI_ExtraModels.HunYuanDiT.tenc import load_clip, load_t5
from ComfyUI_ExtraModels.PixArt.conf import pixart_conf
from ComfyUI_ExtraModels.PixArt.loader import load_pixart
from ComfyUI_ExtraModels.Sana.conf import sana_conf, sana_res
from ComfyUI_ExtraModels.Sana.loader import load_sana
from ComfyUI_ExtraModels.VAE.conf import vae_conf
from ComfyUI_ExtraModels.VAE.loader import EXVAE
import numpy as np
import difflib
import datetime
from ..components import llm_enhancer
import pyrallis
from ..components.sana.diffusion.model.builder import build_model
from ..components.sana.pipeline.sana_pipeline import SanaPipeline
from ..components.sana.diffusion.model.dc_ae.efficientvit.ae_model_zoo import create_dc_ae_model_cfg
from ..components.sana.diffusion.model.dc_ae.efficientvit.models.efficientvit.dc_ae import DCAE
from ..components.sana.diffusion.utils.config import SanaConfig
from ..components.sana.diffusion.model.utils import prepare_prompt_ar
from transformers import AutoTokenizer, T5Tokenizer, T5EncoderModel, AutoModelForCausalLM, BitsAndBytesConfig
from ..components.sana.diffusion.data.datasets.utils import ASPECT_RATIO_512_TEST, ASPECT_RATIO_1024_TEST, ASPECT_RATIO_2048_TEST
import node_helpers
from comfy_api.latest import ComfyExtension, io

class PrimereSamplersSteps:
    CATEGORY = TREE_DASHBOARD
    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT", "FLOAT")
    RETURN_NAMES = ("SAMPLER_NAME", "SCHEDULER_NAME", "STEPS", "CFG")
    FUNCTION = "get_sampler_step"

    kolors_schedulers = ["EulerDiscreteScheduler", "EulerAncestralDiscreteScheduler", "DPMSolverMultistepScheduler", "DPMSolverMultistepScheduler_SDE_karras", "UniPCMultistepScheduler", "DEISMultistepScheduler"]
    sana_schedulers = ['flow_dpm-solver']

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler_name": (cls.sana_schedulers + cls.kolors_schedulers + comfy.samplers.KSampler.SCHEDULERS,),
                "steps": ("INT", {"default": 12, "min": 1, "max": 1000, "step": 1}),
                "cfg": ("FLOAT", {"default": 7, "min": 0.1, "max": 100, "step": 0.01}),
            }
        }

    def get_sampler_step(self, sampler_name, scheduler_name, steps=12, cfg=7):
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

    def load_primere_vae(self, vae_name, baked_vae, ):
        if (vae_name == 'Baked VAE'):
            return (baked_vae,)

        if (vae_name == 'External VAE'):
            vae_name = folder_paths.get_filename_list("vae")[0]

        return utility.vae_loader_class.load_vae(vae_name)[0]

class PrimereModelConceptSelector:
    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT", "FLOAT",
                    "OVERRIDE_STEPS", "STRING", "CLIP_SELECTION", "VAE_SELECTION", "VAE_NAME",
                    "FLOAT",
                    "STRING", "INT", "FLOAT",
                    "STRING", "STRING", "STRING", "STRING",
                    "STRING", "INT", "FLOAT",
                    "STRING", "STRING", "STRING", "STRING", "STRING", "STRING", "FLOAT", "STRING", "STRING",
                    "FLUX_HYPER_LORA", "STRING", "INT", "FLOAT", "FLUX_TURBO_LORA", "STRING", "INT", "FLOAT",
                    "FLUX_SRPO_LORA", "FLUX_SRPO_SVDQ_LORA", "STRING", "INT", "FLOAT", "FLUX_NUNCHAKU_LORA", "STRING", "INT", "FLOAT",
                    "STRING", "STRING", "STRING",
                    "STRING", "STRING", "STRING", "STRING", "SD3_HYPER_LORA", "INT", "FLOAT",
                    "STRING",
                    "STRING", "STRING", "STRING", "FLOAT", "STRING", "STRING", "STRING", "FLOAT", "INT", "INT", "FLOAT", "BOOLEAN",
                    "STRING", "STRING", "STRING", "STRING", "STRING",
                    "STRING", "STRING", "STRING", "QWEN_GEN_LIGHTNING_LORA", "FLOAT", "QWEN_GEN_LORA_PRECISION", "INT", "FLOAT",
                    "STRING", "STRING", "STRING", "QWEN_EDIT_LIGHTNING_LORA", "FLOAT", "QWEN_EDIT_LORA_PRECISION", "INT", "FLOAT",
                    "STRING", "STRING",
                    "STRING", "STRING", "STRING",
                    )
    RETURN_NAMES = ("SAMPLER_NAME", "SCHEDULER_NAME", "STEPS", "CFG",
                    "OVERRIDE_STEPS", "MODEL_CONCEPT", "CLIP_SELECTION", "VAE_SELECTION", "VAE_NAME",
                    "STRENGTH_LCM_LORA_MODEL",
                    "LIGHTNING_SELECTOR", "LIGHTNING_MODEL_STEP", "STRENGTH_LIGHTNING_LORA_MODEL",
                    "CASCADE_STAGE_A", "CASCADE_STAGE_B", "CASCADE_STAGE_C", "CASCADE_CLIP",
                    "HYPER-SD_SELECTOR", "HYPER-SD_MODEL_STEP", "STRENGTH_HYPERSD_LORA_MODEL",
                    "FLUX_SELECTOR", "FLUX_DIFFUSION_MODEL", "FLUX_WEIGHT_TYPE", "FLUX_GGUF_MODEL", "FLUX_CLIP_T5XXL", "FLUX_CLIP_L", "FLUX_CLIP_GUIDANCE", "FLUX_VAE", "FLUX_SAMPLER",
                    "USE_FLUX_HYPER_LORA", "FLUX_HYPER_LORA_TYPE", "FLUX_HYPER_LORA_STEP", "FLUX_HYPER_LORA_STRENGTH", "USE_FLUX_TURBO_LORA", "FLUX_TURBO_LORA_TYPE", "FLUX_TURBO_LORA_STEP", "FLUX_TURBO_LORA_STRENGTH",
                    "USE_FLUX_SRPO_LORA", "USE_FLUX_SRPO_SVDQ_LORA", "FLUX_SRPO_LORA_TYPE", "FLUX_SRPO_LORA_RANK", "FLUX_SRPO_LORA_STRENGTH", "USE_FLUX_NUNCHAKU_LORA", "FLUX_NUNCHAKU_LORA_TYPE", "FLUX_NUNCHAKU_LORA_RANK", "FLUX_NUNCHAKU_LORA_STRENGTH",
                    "HUNYUAN_CLIP_T5XXL", "HUNYUAN_CLIP_L", "HUNYUAN_VAE",
                    "SD3_CLIP_G", "SD3_CLIP_L", "SD3_CLIP_T5XXL", "SD3_UNET_VAE", "USE_SD3_HYPER_LORA", "SD3_HYPER_LORA_STEP", "SD3_HYPER_LORA_STRENGTH",
                    "KOLORS_PRECISION",
                    "PIXART_MODEL_TYPE", "PIXART_T5_ENCODER", "PIXART_VAE", "PIXART_DENOISE", "PIXART_REFINER_MODEL", "PIXART_REFINER_SAMPLER", "PIXART_REFINER_SCHEDULER", "PIXART_REFINER_CFG", "PIXART_REFINER_STEPS", "PIXART_REFINER_START", "PIXART_REFINER_DENOISE", "PIXART_REFINER_IGNORE_PROMPT",
                    "SANA_MODEL", "SANA_ENCODER", "SANA_VAE", "SANA_WEIGHT_DTYPE", "SANA_PRECISION",
                    "QWEN_GEN_MODEL", "QWEN_GEN_CLIP", "QWEN_GEN_VAE", "USE_QWEN_GEN_LIGHTNING_LORA", "QWEN_GEN_LIGHTNING_LORA_VERSION", "QWEN_GEN_LIGHTNING_PRECISION", "QWEN_GEN_LIGHTNING_LORA_STEP", "QWEN_GEN_LIGHTNING_LORA_STRENGTH",
                    "QWEN_EDIT_MODEL", "QWEN_EDIT_CLIP", "QWEN_EDIT_VAE", "USE_QWEN_EDIT_LIGHTNING_LORA", "QWEN_EDIT_LIGHTNING_LORA_VERSION", "QWEN_EDIT_LIGHTNING_PRECISION", "QWEN_EDIT_LIGHTNING_LORA_STEP", "QWEN_EDIT_LIGHTNING_LORA_STRENGTH",
                    "AURAFLOW_CLIP", "AURAFLOW_VAE",
                    "ZIMAGE_MODEL", "ZIMAGE_CLIP", "ZIMAGE_VAE",
                    )

    FUNCTION = "select_model_concept"
    CATEGORY = TREE_DASHBOARD

    UNETLIST = folder_paths.get_filename_list("unet")
    DIFFUSIONLIST = folder_paths.get_filename_list("diffusion_models")
    TEXT_ENCODERS = folder_paths.get_filename_list("text_encoders")
    GGUFLIST = folder_paths.get_filename_list("unet_gguf")
    VAELIST = folder_paths.get_filename_list("vae")
    CLIPLIST = folder_paths.get_filename_list("clip")
    CLIPLIST += folder_paths.get_filename_list("clip_gguf")
    MODELLIST = folder_paths.get_filename_list("checkpoints")

    T5_DIR = os.path.join(folder_paths.models_dir, 't5')
    if os.path.isdir(T5_DIR):
        folder_paths.add_model_folder_path("p_t5", T5_DIR)
        T5models = folder_paths.get_filename_list("p_t5")
        T5List = folder_paths.filter_files_extensions(T5models, ['.bin', '.safetensors'])
        CLIPLIST += T5List

    TENC_DIR = os.path.join(folder_paths.models_dir, 'text_encoders')
    LLM_PRIMERE_ROOT = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'LLM')
    TEXT_ENCODERS_PATHS = llm_enhancer.getValidLLMPaths(TENC_DIR)
    TEXT_ENCODERS_PATHS += llm_enhancer.getValidLLMPaths(LLM_PRIMERE_ROOT)

    CONCEPT_LIST = utility.SUPPORTED_MODELS[0:26]

    SAMPLER_INPUTS = {
        'model_version': ("STRING", {"forceInput": True, "default": "SD1"}),
        'model_name': ("CHECKPOINT_NAME", {"forceInput": True})
    }

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
            "override_steps": ("BOOLEAN", {"default": False, "label_off": "Set by sampler settings", "label_on": "Set by model filename"}),

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
            "use_flux_hyper_lora": ("BOOLEAN", {"default": False, "label_on": "Use hyper Lora", "label_off": "Ignore hyper Lora"}),
            "flux_hyper_lora_type": (["FLUX.1-dev", "FLUX.1-dev-fp16"], {"default": "FLUX.1-dev"}),
            "flux_hyper_lora_step": ([8, 16], {"default": 16}),
            "flux_hyper_lora_strength": ("FLOAT", {"default": 0.125, "min": -20.000, "max": 20.000, "step": 0.001}),
            "use_flux_turbo_lora": ("BOOLEAN", {"default": False, "label_on": "Use turbo Lora", "label_off": "Ignore turbo Lora"}),
            "flux_turbo_lora_type": (["TurboAlpha", "TurboRender"], {"default": "TurboAlpha"}),
            "flux_turbo_lora_step": ([4, 6, 8, 10, 12], {"default": 8}),
            "flux_turbo_lora_strength": ("FLOAT", {"default": 1, "min": -20.000, "max": 20.000, "step": 0.001}),
            "use_flux_srpo_lora": ("BOOLEAN", {"default": False, "label_on": "Use SRPO Lora", "label_off": "Ignore SRPO Lora"}),
            "use_flux_srpo_svdq_lora": ("BOOLEAN", {"default": False, "label_on": "Use SRPO-NUNCHAKU Lora", "label_off": "Ignore SRPO-NUNCHAKU Lora"}),
            "flux_srpo_lora_type": (["R&Q", "RockerBOO", "oficial", "adaptive"], {"default": "oficial"}),
            "flux_srpo_lora_rank": ([8, 16, 32, 64, 128, 256], {"default": 8}),
            "flux_srpo_lora_strength": ("FLOAT", {"default": 1, "min": -20.000, "max": 20.000, "step": 0.001}),
            "use_flux_nunchaku_lora": ("BOOLEAN", {"default": False, "label_on": "Use nunchaku Lora", "label_off": "Ignore nunchaku Lora"}),
            "flux_nunchaku_lora_type": (["kontext_deblur", "kontext_face_detailer", "anything_extracted"], {"default": "anything_extracted"}),
            "flux_nunchaku_lora_rank": ([64, 256], {"default": 64}),
            "flux_nunchaku_lora_strength": ("FLOAT", {"default": 1, "min": -20.000, "max": 20.000, "step": 0.001}),

            "zimage_model": (["None"] + DIFFUSIONLIST,),
            "zimage_clip": (["None"] + TEXT_ENCODERS + CLIPLIST,),
            "zimage_vae": (["None"] + VAELIST,),

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

            "kolors_precision": (['fp16', 'quant8', 'quant4'], {"default": "quant8"}),

            "pixart_model_type": (["Auto"] + list(pixart_conf.keys()), {"default": "Auto"}),
            "pixart_T5_encoder": (["None"] + CLIPLIST,),
            "pixart_vae": (["None"] + VAELIST,),
            "pixart_denoise": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
            "pixart_refiner_model": (["None"] + MODELLIST,),
            "pixart_refiner_sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m"}),
            "pixart_refiner_scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
            "pixart_refiner_cfg": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 100, "step": 0.01}),
            "pixart_refiner_steps": ("INT", {"default": 22, "min": 10, "max": 30, "step": 1}),
            "pixart_refiner_start": ("INT", {"default": 12, "min": 1, "max": 1000, "step": 1}),
            "pixart_refiner_denoise": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
            "pixart_refiner_ignore_prompt": ("BOOLEAN", {"default": False, "label_on": "Send prompt to refiner", "label_off": "Ignore prompt"}),

            "sana_model": (["None"] + DIFFUSIONLIST + MODELLIST,),
            "sana_encoder": (["None"] + TEXT_ENCODERS_PATHS,),
            "sana_vae": (["None"] + VAELIST,),
            "sana_weight_dtype": (["Auto", "fp16", "bf16", "fp32"], {"default": "fp16"}),
            "sana_precision": (['fp32', 'fp16', 'quant8', 'quant4'], {"default": "fp16"}),

            "qwen_gen_model": (["None"] + DIFFUSIONLIST,),
            "qwen_gen_clip": (["None"] + CLIPLIST,),
            "qwen_gen_vae": (["None"] + VAELIST,),
            "use_qwen_gen_lightning_lora": ("BOOLEAN", {"default": False, "label_on": "Use lightning Lora", "label_off": "Ignore Lora"}),
            "qwen_gen_lightning_lora_version": ([1.0, 1.1, 2.0], {"default": 2.0}),
            "qwen_gen_lightning_precision": ("BOOLEAN", {"default": True, "label_on": "FP32", "label_off": "BF16"}),
            "qwen_gen_lightning_lora_step": ([4, 8], {"default": 8}),
            "qwen_gen_lightning_lora_strength": ("FLOAT", {"default": 1.00, "min": -20.00, "max": 20.00, "step": 0.01}),

            "qwen_edit_model": (["None"] + DIFFUSIONLIST,),
            "qwen_edit_clip": (["None"] + CLIPLIST,),
            "qwen_edit_vae": (["None"] + VAELIST,),
            "use_qwen_edit_lightning_lora": ("BOOLEAN", {"default": False, "label_on": "Use lightning Lora", "label_off": "Ignore Lora"}),
            "qwen_edit_lightning_lora_version": ([1.0], {"default": 1.0}),
            "qwen_edit_lightning_precision": ("BOOLEAN", {"default": True, "label_on": "FP32", "label_off": "BF16"}),
            "qwen_edit_lightning_lora_step": ([4, 8], {"default": 8}),
            "qwen_edit_lightning_lora_strength": ("FLOAT", {"default": 1.00, "min": -20.00, "max": 20.00, "step": 0.01}),

            "auraflow_clip": (["None"] + CLIPLIST,),
            "auraflow_vae": (["None"] + VAELIST,),
        },
        "hidden": {
            "extra_pnginfo": "EXTRA_PNGINFO",
            "prompt": "PROMPT"
        },
        "optional": SAMPLER_INPUTS
    }

    @classmethod
    def INPUT_TYPES(cls):
        return cls.INPUT_DICT

    def select_model_concept(self, cascade_stage_a, cascade_stage_b, cascade_stage_c, cascade_clip,
                             flux_diffusion, flux_weight_dtype, flux_gguf, flux_clip_t5xxl, flux_clip_l, flux_vae,
                             qwen_gen_model, qwen_gen_clip, qwen_gen_vae,
                             hunyuan_clip_t5xxl, hunyuan_clip_l, hunyuan_vae,
                             sd3_clip_g, sd3_clip_l, sd3_clip_t5xxl, sd3_unet_vae,
                             qwen_edit_model, qwen_edit_clip, qwen_edit_vae,
                             auraflow_clip, auraflow_vae,
                             zimage_model, zimage_clip, zimage_vae,
                             prompt,
                             override_steps=False,
                             use_sd3_hyper_lora=False, sd3_hyper_lora_step=8, sd3_hyper_lora_strength=0.125,
                             use_qwen_gen_lightning_lora=False, qwen_gen_lightning_lora_version=1.1, qwen_gen_lightning_precision=True, qwen_gen_lightning_lora_step=8, qwen_gen_lightning_lora_strength=1.00,
                             use_qwen_edit_lightning_lora=False, qwen_edit_lightning_lora_version=1.1, qwen_edit_lightning_precision=True, qwen_edit_lightning_lora_step=8, qwen_edit_lightning_lora_strength=1.00,
                             kolors_precision='quant8',
                             pixart_model_type="Auto", pixart_T5_encoder='None', pixart_vae='None', pixart_denoise=0.9, pixart_refiner_model='None', pixart_refiner_sampler='dpmpp_2m', pixart_refiner_scheduler='Normal', pixart_refiner_cfg=2.0, pixart_refiner_steps=22, pixart_refiner_start=12, pixart_refiner_denoise=0.9, pixart_refiner_ignore_prompt=False,
                             sana_model="None", sana_encoder="None", sana_vae="None", sana_weight_dtype="Auto", sana_precision="fp16",
                             model_version=None, model_name=None,
                             default_sampler_name='euler', default_scheduler_name='normal', default_cfg_scale=7, default_steps=12,
                             sd_vae="None", sdxl_vae="None",
                             model_concept='Auto',
                             clip_selection=True, vae_selection=True,
                             strength_lcm_lora_model=1,
                             lightning_selector="LORA", lightning_model_step=8, lightning_sampler=False,
                             strength_lightning_lora_model=1,
                             hypersd_selector="LORA", hypersd_model_step=8, hypersd_sampler=False,
                             strength_hypersd_lora_model=1,
                             flux_sampler='ksampler', flux_selector="DIFFUSION", flux_clip_guidance=3.5,
                             use_flux_hyper_lora=False, flux_hyper_lora_type='FLUX.1-dev', flux_hyper_lora_step=16, flux_hyper_lora_strength=0.125,  use_flux_turbo_lora=False, flux_turbo_lora_type="TurboAlpha", flux_turbo_lora_step=8, flux_turbo_lora_strength=1,
                             use_flux_srpo_lora=False, use_flux_srpo_svdq_lora=False, flux_srpo_lora_type='oficial', flux_srpo_lora_rank=8, flux_srpo_lora_strength=1, use_flux_nunchaku_lora=False, flux_nunchaku_lora_type='anything_extracted', flux_nunchaku_lora_rank=64, flux_nunchaku_lora_strength=1,
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

        if model_concept == 'QwenEdit':
            WORKFLOWDATA = kwargs['extra_pnginfo']['workflow']['nodes']
            editmodel_images = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereMultiImage', 'process_list', prompt)
            if editmodel_images != True:
               model_concept = 'QwenGen'

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

        if pixart_model_type == 'Auto' and model_name is not None:
            cutoff_list = list(np.around(np.arange(0.1, 1.01, 0.01).tolist(), 2))[::-1]
            pixartTypes = list(pixart_conf.keys())
            is_found = []
            trycut = 0
            pixart_model_name = Path(model_name).stem
            for trycut in cutoff_list:
                is_found = difflib.get_close_matches(pixart_model_name, pixartTypes, cutoff=trycut)
                if len(is_found) > 0:
                    break

            if trycut <= 0.35:
                pixart_model_type = 'PixArtMS_Sigma_XL_2'
            else:
                pixart_model_type = is_found[0]

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
            use_flux_turbo_lora = None
            flux_turbo_lora_type = None
            flux_turbo_lora_step = None
            flux_turbo_lora_strength = None
            use_flux_srpo_lora = None
            use_flux_srpo_svdq_lora = None
            flux_srpo_lora_type = None
            flux_srpo_lora_rank = None
            flux_srpo_lora_strength = None
            use_flux_nunchaku_lora = None
            flux_nunchaku_lora_type = None
            flux_nunchaku_lora_rank = None
            flux_nunchaku_lora_strength = None

        if model_concept != 'Hunyuan':
            hunyuan_clip_t5xxl = None
            hunyuan_clip_l = None
            hunyuan_vae = None

        if model_concept != 'Z-Image':
            zimage_model = None
            zimage_clip = None
            zimage_vae = None

        if model_concept != 'AuraFlow':
            auraflow_clip = None
            auraflow_vae = None

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

        if model_concept != 'QwenGen':
            qwen_gen_model = None
            qwen_gen_clip = None
            qwen_gen_vae = None
            use_qwen_gen_lightning_lora = None
            qwen_gen_lightning_lora_step = None
            qwen_gen_lightning_lora_strength = None
            qwen_gen_lightning_lora_version = None
            qwen_gen_lightning_precision = None

        if model_concept != 'QwenEdit':
            qwen_edit_model = None
            qwen_edit_clip = None
            qwen_edit_vae = None
            use_qwen_edit_lightning_lora = None
            qwen_edit_lightning_lora_step = None
            qwen_edit_lightning_lora_strength = None
            qwen_edit_lightning_lora_version = None
            qwen_edit_lightning_precision = None

        if model_concept != 'PixartSigma':
            pixart_model_type = None
            pixart_T5_encoder = None
            pixart_vae = None
            pixart_denoise = None
            pixart_refiner_model = None
            pixart_refiner_sampler = None
            pixart_refiner_scheduler = None
            pixart_refiner_cfg = None
            pixart_refiner_steps = None
            pixart_refiner_start = None
            pixart_refiner_denoise = None
            pixart_refiner_ignore_prompt = None

        if model_concept != 'SANA512' and model_concept != 'SANA1024':
            sana_model = None
            sana_encoder = None
            sana_vae = None
            sana_weight_dtype = None
            sana_precision = None

        if model_name is not None and override_steps == True:
            is_steps = re.findall(r"(?i)(\d+)Step", model_name)
            if len(is_steps) > 0:
                steps = int(is_steps[0])
                use_flux_hyper_lora = False
                use_sd3_hyper_lora = False
                use_flux_turbo_lora = False
                use_qwen_gen_lightning_lora = False
                use_qwen_edit_lightning_lora = False
            elif 'lightning' in model_name:
                steps = 5
                use_flux_hyper_lora = False
                use_sd3_hyper_lora = False
                use_flux_turbo_lora = False
                use_qwen_gen_lightning_lora = False
                use_qwen_edit_lightning_lora = False
                cfg_scale = 1.1

        if model_concept == 'Flux' and use_flux_hyper_lora == True:
            steps = flux_hyper_lora_step
        if model_concept == 'Flux' and use_flux_turbo_lora == True:
            steps = flux_turbo_lora_step

        if model_concept == 'QwenGen' and use_qwen_gen_lightning_lora == True:
            steps = qwen_gen_lightning_lora_step + 1
            cfg_scale = 1.1
        if model_concept == 'QwenEdit' and use_qwen_edit_lightning_lora == True:
            steps = qwen_edit_lightning_lora_step + 1
            cfg_scale = 1.1

        if model_concept == 'SD3' and use_sd3_hyper_lora == True:
            fullpathFile = folder_paths.get_full_path('checkpoints', model_name)
            is_link = os.path.islink(str(fullpathFile))
            if is_link == True:
                File_link = Path(str(fullpathFile)).resolve()
                model_ext = os.path.splitext(File_link)[1].lower()
                if model_ext != '.gguf':
                    steps = sd3_hyper_lora_step
            else:
                steps = sd3_hyper_lora_step

        if '_distill' in model_name.lower() and 'Qwen' in model_concept:
            cfg_scale = 1

        return (sampler_name, scheduler_name, steps, round(cfg_scale, 2),
                override_steps, model_concept, clip_selection, vae_selection, vae,
                strength_lcm_lora_model,
                lightning_selector, lightning_model_step, strength_lightning_lora_model,
                cascade_stage_a, cascade_stage_b, cascade_stage_c, cascade_clip,
                hypersd_selector, hypersd_model_step, strength_hypersd_lora_model,
                flux_selector, flux_diffusion, flux_weight_dtype, flux_gguf, flux_clip_t5xxl, flux_clip_l, flux_clip_guidance, flux_vae, flux_sampler,
                use_flux_hyper_lora, flux_hyper_lora_type, flux_hyper_lora_step, flux_hyper_lora_strength, use_flux_turbo_lora, flux_turbo_lora_type, flux_turbo_lora_step, flux_turbo_lora_strength,
                use_flux_srpo_lora, use_flux_srpo_svdq_lora, flux_srpo_lora_type, flux_srpo_lora_rank, flux_srpo_lora_strength, use_flux_nunchaku_lora, flux_nunchaku_lora_type, flux_nunchaku_lora_rank, flux_nunchaku_lora_strength,
                hunyuan_clip_t5xxl, hunyuan_clip_l, hunyuan_vae,
                sd3_clip_g, sd3_clip_l, sd3_clip_t5xxl, sd3_unet_vae, use_sd3_hyper_lora, sd3_hyper_lora_step, sd3_hyper_lora_strength,
                kolors_precision,
                pixart_model_type, pixart_T5_encoder, pixart_vae, pixart_denoise, pixart_refiner_model, pixart_refiner_sampler, pixart_refiner_scheduler, pixart_refiner_cfg, pixart_refiner_steps, pixart_refiner_start, pixart_refiner_denoise, pixart_refiner_ignore_prompt,
                sana_model, sana_encoder, sana_vae, sana_weight_dtype, sana_precision,
                qwen_gen_model, qwen_gen_clip, qwen_gen_vae, use_qwen_gen_lightning_lora, qwen_gen_lightning_lora_version, qwen_gen_lightning_precision, qwen_gen_lightning_lora_step, qwen_gen_lightning_lora_strength,
                qwen_edit_model, qwen_edit_clip, qwen_edit_vae, use_qwen_edit_lightning_lora, qwen_edit_lightning_lora_version, qwen_edit_lightning_precision, qwen_edit_lightning_lora_step, qwen_edit_lightning_lora_strength,
                auraflow_clip, auraflow_vae,
                zimage_model, zimage_clip, zimage_vae
                )

class PrimereAutoSamplerSettings:
    CATEGORY = TREE_DASHBOARD
    RETURN_TYPES = ("TUPLE", comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("DATA", "SAMPLER_NAME", "SCHEDULER_NAME", "STEPS", "CFG", "MODEL_CONCEPT")
    FUNCTION = "get_controlledsampler"
    OUTPUT_NODE = True

    kolors_schedulers = ["EulerDiscreteScheduler", "EulerAncestralDiscreteScheduler", "DPMSolverMultistepScheduler", "DPMSolverMultistepScheduler_SDE_karras", "UniPCMultistepScheduler", "DEISMultistepScheduler"]
    sana_schedulers = ['flow_dpm-solver']

    UNETLIST = PrimereModelConceptSelector.UNETLIST
    DIFFUSIONLIST = PrimereModelConceptSelector.DIFFUSIONLIST
    TEXT_ENCODERS = PrimereModelConceptSelector.TEXT_ENCODERS
    GGUFLIST = PrimereModelConceptSelector.GGUFLIST
    VAELIST = PrimereModelConceptSelector.VAELIST
    CLIPLIST = PrimereModelConceptSelector.CLIPLIST
    MODELLIST = PrimereModelConceptSelector.MODELLIST
    TEXT_ENCODERS_PATHS = PrimereModelConceptSelector.TEXT_ENCODERS_PATHS
    CONCEPT_LIST =  PrimereModelConceptSelector.CONCEPT_LIST

    CUSTOMLORA_DIR = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads')
    folder_paths.add_model_folder_path("customlora", CUSTOMLORA_DIR)
    CustomLoras = folder_paths.get_filename_list("customlora")
    CustomLorasList = folder_paths.filter_files_extensions(CustomLoras, ['.safetensors'])

    LCM_LORAS       = [n for n in CustomLorasList if "lcm" in n.lower()]
    SPEED_LORAS     = [n for n in CustomLorasList if any(s in n.lower() for s in ("lightning", "hyper", "turbo"))]
    SRPO_LORAS      = [n for n in CustomLorasList if "srpo" in n.lower() and "svdq" not in n.lower()]
    SRPO_SVDQ_LORAS = [n for n in CustomLorasList if "srpo" in n.lower() and "svdq" in n.lower()]
    NUNCHAKU_LORAS  = [n for n in CustomLorasList if "nunchaku" in n.lower()]

    REFINER_MODELS  = [n for n in PrimereModelConceptSelector.MODELLIST if "refiner" in os.path.basename(n).lower() or "refiner" in os.path.dirname(n).lower()]

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_concept": ("STRING", {"default": None, "forceInput": True}),
                "model_name": ("CHECKPOINT_NAME", {"default": None, "forceInput": True}),
                "concepts": (["Auto"] + cls.CONCEPT_LIST,),
                "models": (["Auto"] + cls.MODELLIST,),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler_name": (cls.sana_schedulers + cls.kolors_schedulers + comfy.samplers.KSampler.SCHEDULERS,),
                "steps": ("INT", {"default": 12, "min": 1, "max": 1000, "step": 1}),
                "override_steps": ("BOOLEAN", {"default": False, "label_off": "Set by sampler settings", "label_on": "Set by model filename"}),
                "cfg": ("FLOAT", {"default": 7, "min": 0.1, "max": 100, "step": 0.01}),
                "rescale_cfg": ("FLOAT", {"default": 1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "last_layer": ("INT", {"default": 0, "min": -24, "max": 0, "step": 1}),
                "sigma_max": ("FLOAT", {"default": 120, "min": 1, "max": 200, "step": 0.001}),
                "sigma_min": ("FLOAT", {"default": 1, "min": 0.001, "max": 100, "step": 0.001}),
                "vae": (cls.VAELIST,),
                "vae_selection": ("BOOLEAN", {"default": True, "label_on": "Use baked if exist", "label_off": "Always use custom"}),
                "clip_selection": ("BOOLEAN", {"default": True, "label_on": "Use baked if exist", "label_off": "Always use custom"}),
                "encoder_1": (list(dict.fromkeys(["None"] + cls.TEXT_ENCODERS + cls.CLIPLIST + cls.UNETLIST + cls.TEXT_ENCODERS_PATHS)),),
                "encoder_2": (list(dict.fromkeys(["None"] + cls.TEXT_ENCODERS + cls.CLIPLIST + cls.UNETLIST + cls.TEXT_ENCODERS_PATHS)),),
                "encoder_3": (list(dict.fromkeys(["None"] + cls.TEXT_ENCODERS + cls.CLIPLIST + cls.UNETLIST + cls.TEXT_ENCODERS_PATHS)),),
                "sampler": (["custom_advanced", "ksampler"], {"default": "ksampler"}),
                "guidance": ('FLOAT', {"default": 3.5, "min": 0.0, "max": 100.0, "step": 0.1}),
                "weight_dtype": (["None"] + ["Auto", "default", "fp16", "bf16", "fp32", "fp8_e4m3fn", "fp8_e5m2"], {"default": "default"}),
                "precision": (["None"] + ['fp32', 'fp16', 'quant8', 'quant4'], {"default": "fp16"}),
                "lcm_lora": ("BOOLEAN", {"default": False, "label_on": "LCM lora ON", "label_off": "LCM lora OFF"}),
                # "lcm_lora_name": (cls.LCM_LORAS,),
                "lcm_lora_strength": ("FLOAT", {"default": 1.000, "min": -20.000, "max": 20.000, "step": 0.001}),
                "speed_lora": ("BOOLEAN", {"default": False, "label_on": "Speed lora ON", "label_off": "Speed lora OFF"}),
                "speed_lora_name": (cls.SPEED_LORAS,),
                # "speed_lora_version": ([1.0, 1.1, 2.0], {"default": 2.0}),
                # "speed_lora_precision": ("BOOLEAN", {"default": True, "label_on": "FP32", "label_off": "BF16"}),
                # "speed_lora_step": ([4, 6, 8, 10, 12, 16], {"default": 8}),
                "speed_lora_strength": ("FLOAT", {"default": 1.00, "min": -20.00, "max": 20.00, "step": 0.01}),
                "speed_lora_cfg": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 100, "step": 0.01}),
                "speed_lora_steps_offset": ("INT", {"default": 0, "min": -5, "max": 5, "step": 1}),
                "srpo_lora": ("BOOLEAN", {"default": False, "label_on": "Use SRPO Lora", "label_off": "Ignore SRPO Lora"}),
                "srpo_lora_name": (cls.SRPO_LORAS,),
                # "srpo_lora_type": (["R&Q", "RockerBOO", "oficial", "adaptive"], {"default": "oficial"}),
                # "srpo_lora_rank": ([8, 16, 32, 64, 128, 256], {"default": 8}),
                "srpo_lora_strength": ("FLOAT", {"default": 1, "min": -20.000, "max": 20.000, "step": 0.001}),
                "srpo_svdq_lora": ("BOOLEAN", {"default": False, "label_on": "Use SRPO SVDQ Lora", "label_off": "Ignore SRPO SVDQ Lora"}),
                "srpo_svdq_lora_name": (cls.SRPO_SVDQ_LORAS,),
                "nunchaku_lora": ("BOOLEAN", {"default": False, "label_on": "Use nunchaku Lora", "label_off": "Ignore nunchaku Lora"}),
                "nunchaku_lora_name": (cls.NUNCHAKU_LORAS,),
                # "nunchaku_lora_type": (["kontext_deblur", "kontext_face_detailer", "anything_extracted"], {"default": "anything_extracted"}),
                # "nunchaku_lora_rank": ([64, 256], {"default": 64}),
                "nunchaku_lora_strength": ("FLOAT", {"default": 1, "min": -20.000, "max": 20.000, "step": 0.001}),
                "refiner": ("BOOLEAN", {"default": False, "label_on": "Refiner ON", "label_off": "Refiner OFF"}),
                "refiner_model": (cls.REFINER_MODELS,),
                "refiner_sampler": (comfy.samplers.KSampler.SAMPLERS, {"default": "dpmpp_2m"}),
                "refiner_scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"default": "normal"}),
                "refiner_cfg": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 100, "step": 0.01}),
                "refiner_steps": ("INT", {"default": 22, "min": 10, "max": 30, "step": 1}),
                "refiner_start": ("INT", {"default": 12, "min": 1, "max": 1000, "step": 1}),
                "refiner_denoise": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "refiner_sampling_denoise": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "refiner_ignore_prompt": ("BOOLEAN", {"default": True, "label_on": "Ignore prompt", "label_off": "Send prompt to refiner"}),
            }
        }

    def get_controlledsampler(self, **kwargs):
        model_concept = kwargs.pop('model_concept', 'SD1')
        model_name = kwargs.pop('model_name', None)
        concepts = kwargs.pop('concepts', 'Auto')
        models = kwargs.pop('models', 'Auto')
        sampler_name = kwargs.pop('sampler_name', comfy.samplers.KSampler.SAMPLERS[0])
        scheduler_name = kwargs.pop('scheduler_name', comfy.samplers.KSampler.SCHEDULERS[0])
        steps = kwargs.pop('steps', 12)
        cfg = kwargs.pop('cfg', 7.0)
        active_concept = model_concept if concepts == "Auto" else concepts
        if concepts == "Auto" and models == "Auto":
            raw_model = model_name
            model_key = os.path.splitext(os.path.basename(raw_model))[0] if raw_model else None
            json_path = os.path.join(PRIMERE_ROOT, 'front_end', 'model_concept.json')
            concept_data = utility.json2tuple(json_path)
            if model_key and concept_data and model_key in concept_data:
                lookup_key = model_key
            else:
                lookup_key = active_concept
                model_key = None
            active_display = model_key if model_key else active_concept
            if not concept_data or lookup_key not in concept_data:
                PromptServer.instance.send_sync("primere.concept_setting", {"status": "missing", "concept": active_concept})
            else:
                saved = concept_data[lookup_key]
                sampler_name = saved.get('sampler_name', sampler_name)
                scheduler_name = saved.get('scheduler_name', scheduler_name)
                steps = saved.get('steps', steps)
                cfg = saved.get('cfg', cfg)
                for k, v in saved.items():
                    if k in kwargs:
                        kwargs[k] = v
        else:
            active_display = active_concept

        if kwargs.get('override_steps') == True:
            found = re.findall(r"(?i)(\d+)step", model_name.lower())
            if found:
                steps = int(found[0])

        if kwargs.get('speed_lora') == True:
            speed_lora_name_val = kwargs.get('speed_lora_name', '')
            if speed_lora_name_val:
                found = re.findall(r"(?i)(\d+)step", speed_lora_name_val.lower())
                if found:
                    steps = int(found[0])
                    offset = int(kwargs.get('speed_lora_steps_offset', 0))
                    if offset:
                        steps = max(1, steps + offset)
            speed_lora_cfg_val = kwargs.get('speed_lora_cfg')
            if speed_lora_cfg_val is not None:
                cfg = float(speed_lora_cfg_val)
        suppressed = [k + "_" for k, v in kwargs.items() if v == "None" or v is False]
        kwargs = {k: v for k, v in kwargs.items() if v != "None" and not any(k.startswith(p) for p in suppressed)}
        kwargs['encoders'] = [kwargs[k] for k in ('encoder_1', 'encoder_2', 'encoder_3') if kwargs.get(k) not in (None, 'None')]
        kwargs['model_concept'] = active_concept
        kwargs['sampler_name'] = sampler_name
        kwargs['scheduler_name'] = scheduler_name
        kwargs['steps'] = steps
        kwargs['cfg'] = round(cfg, 2)
        return {"ui": {"active_concept": [active_display]}, "result": (kwargs, sampler_name, scheduler_name, steps, round(cfg, 2), active_concept,)}

class PrimereConceptDataTuple:
    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT", "FLOAT", "TUPLE",)
    RETURN_NAMES = ("SAMPLER_NAME", "SCHEDULER_NAME", "STEPS", "CFG", "DATA",)
    FUNCTION = "load_concept_collector"
    CATEGORY = TREE_DASHBOARD

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "data": ("TUPLE", {"forceInput": True}),
            },
        }

    def load_concept_collector(self, data):
        sampler_name = data.get("sampler_name", comfy.samplers.KSampler.SAMPLERS[0])
        scheduler_name = data.get("scheduler_name", comfy.samplers.KSampler.SCHEDULERS[0])
        steps = data.get("steps", 20)
        cfg = data.get("cfg", 7.0)
        return (sampler_name, scheduler_name, steps, cfg, data,)

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
                # "model_concept": ("STRING", {"forceInput": True}),
                "concept_data": ("TUPLE", {"default": None, "forceInput": True}),
                "loaded_model": ('MODEL', {"forceInput": True, "default": None}),
                "loaded_clip": ('CLIP', {"forceInput": True, "default": None}),
                "loaded_vae": ('VAE', {"forceInput": True, "default": None}),
            },
        }

    def load_primere_ckpt(self, ckpt_name, use_yaml, concept_data=None, loaded_model=None, loaded_clip=None, loaded_vae=None):
        model_concept = concept_data['model_concept']

        try:
            comfy.model_management.soft_empty_cache()
            comfy.model_management.cleanup_models(True)
            comfy.model_management.unload_all_models()
        except Exception:
            print('No need to clear cache...')

        modelname_only = Path(ckpt_name).stem
        MODEL_VERSION_ORIGINAL = utility.get_value_from_cache('model_version', modelname_only)
        if MODEL_VERSION_ORIGINAL is None:
            MODEL_VERSION_ORIGINAL = utility.getModelType(ckpt_name, 'checkpoints')
            utility.add_value_to_cache('model_version', ckpt_name, MODEL_VERSION_ORIGINAL)

        path = Path(ckpt_name)
        ModelName = path.stem
        ModelConfigPath = path.parent.joinpath(ModelName + '.yaml')
        ModelConfigFullPath = Path(folder_paths.models_dir).joinpath('checkpoints').joinpath(ModelConfigPath)

        LOADED_CHECKPOINT = []
        if loaded_model is not None and loaded_clip is not None and loaded_vae is not None:
            LOADED_CHECKPOINT.insert(0, loaded_model)
            LOADED_CHECKPOINT.insert(1, loaded_clip)
            LOADED_CHECKPOINT.insert(2, loaded_vae)
            OUTPUT_MODEL = LOADED_CHECKPOINT[0]

        match model_concept:
            case 'SD1' | 'SD2' | 'SDXL' | 'Illustrious' | 'Turbo' | 'Pony':
                OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE = model_loaders.load_sd_model(self, ckpt_name, use_yaml, ModelConfigFullPath, concept_data)
            case 'SD3':
                OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE = model_loaders.load_sd3_model(self, ckpt_name, concept_data)
            case 'StableCascade':
                OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE = model_loaders.load_stable_cascade_model(self, ckpt_name, concept_data)
            case 'Z-Image':
                OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE = model_loaders.load_zimage_model(self, ckpt_name, concept_data)
            case 'Flux':
                OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE = model_loaders.load_flux_model(self, ckpt_name, concept_data)
            case 'LCM':
                OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE = model_loaders.load_lcm_model(self, ckpt_name, concept_data)
            case 'Hyper' | 'Lightning':
                OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE = model_loaders.load_lightning_hyper_model(self, ckpt_name, concept_data)
            case 'Playground':
                OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE = model_loaders.load_playground_model(self, ckpt_name, use_yaml, ModelConfigFullPath, concept_data)
            case 'PixartSigma':
                OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE = model_loaders.load_pixart_model(self, ckpt_name, concept_data)
            case 'AuraFlow':
                OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE = model_loaders.load_auraflow_model(self, ckpt_name, concept_data)
            case 'SANA1024' | 'SANA512':
                OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE = model_loaders.load_sana_model(self, ckpt_name, concept_data)
            case 'KwaiKolors':
                OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE = model_loaders.load_kolors_model(self, ckpt_name, concept_data)
            case 'Hunyuan':
                OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE = model_loaders.load_hunyuan_model(self, ckpt_name, concept_data)
            case 'QwenGen' | 'QwenEdit':
                OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE = model_loaders.load_qwen_model(self, ckpt_name, concept_data)
            case 'Chroma':
                OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE = model_loaders.load_chroma_model(self, ckpt_name, concept_data)

        return (OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE, MODEL_VERSION_ORIGINAL)

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
        return {
            "required": {
                "select": ("INT", {"default": 1, "min": 1, "max": 20, "step": 1}),
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
                "seed": ("INT", {"default": -1, "min": -1, "max": utility.MAX_SEED}),
            }
        }

    def seed(self, seed=-1):
        return (seed,)

class PrimereFastSeed:
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("SEED",)
    FUNCTION = "primere_fastseed"
    OUTPUT_NODE = True
    CATEGORY = TREE_DASHBOARD

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "seed_setup": (['Random', 'Increase', 'Decrease', 'UseLast', 'UseCustom'], {"default": 'Random'}),
                "custom_seed": ("INT", {"default": 42, "min": 0, "max": utility.MAX_SEED, "step": 1}),
            }
        }

    @classmethod
    def IS_CHANGED(self, **kwargs):
        if kwargs['seed_setup'] == 'UseCustom' and seed != kwargs['custom_seed']:
            return float("NaN")

        if kwargs['seed_setup'] != 'UseLast' and kwargs['seed_setup'] != 'UseCustom':
            return float("NaN")

    def primere_fastseed(self, seed_setup, custom_seed):
        global seed

        random.seed(datetime.datetime.now().timestamp())
        match seed_setup:
            case "Random":
                seed = random.randint(1000, utility.MAX_SEED)
            case "Increase":
                if 'seed' in globals():
                    seed = seed + 1
                else:
                    seed = random.randint(1000, utility.MAX_SEED)
            case "Decrease":
                if 'seed' in globals():
                    seed = seed - 1
                else:
                    seed = random.randint(1000, utility.MAX_SEED)
            case "UseLast":
                if 'seed' in globals():
                    seed = seed
                else:
                    seed = random.randint(1000, utility.MAX_SEED)
            case "UseCustom":
                seed = custom_seed

        return {"ui": {"text": [f'Last seed: [{seed}]']}, "result": (seed,)}

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
            "noise_seed": ("INT", {"default": 0, "min": -1, "max": utility.MAX_SEED, "forceInput": True}),
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

    def primere_latent_noise(self, width, height, rand_noise_type, noise_type, rand_alpha_exponent, alpha_exponent, alpha_exp_rand_min, alpha_exp_rand_max, rand_modulator, modulator, modulator_rand_min, modulator_rand_max, noise_seed, rand_device, device, optional_vae=None, workflow_tuple=None, expand_random_limits=False):
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

        power_law = PowerLawNoise(device=device)

        if rand_alpha_exponent == True:
            alpha_exponent = round(random.uniform(alpha_exp_rand_min, alpha_exp_rand_max), 3)

        if rand_modulator == True:
            modulator = round(random.uniform(modulator_rand_min, modulator_rand_max), 2)

        tensors = power_law(1, width, height, scale=1, alpha=alpha_exponent, modulator=modulator, noise_type=noise_type, seed=noise_seed)
        alpha_channel = torch.ones((1, height, width, 1), dtype=tensors.dtype, device="cpu")
        tensors = torch.cat((tensors, alpha_channel), dim=3)

        if optional_vae is None:
            latents = tensors.permute(0, 3, 1, 2)
            latents = F.interpolate(latents, size=((height // 8), (width // 8)), mode='nearest-exact')
            return {'samples': latents}, tensors, workflow_tuple

        encoder = nodes.VAEEncode()
        latents = []
        for tensor in tensors:
            tensor = tensor.unsqueeze(0)
            try:
                latents.append(encoder.encode(optional_vae, tensor)[0]['samples'])
            except Exception:
                latents = tensors.permute(0, 3, 1, 2)
                latents = F.interpolate(latents, size=((height // 8), (width // 8)), mode='nearest-exact')
                return {'samples': latents}, tensors, workflow_tuple
        latents = torch.cat(latents)

        if workflow_tuple is not None:
            workflow_tuple['latent_data'] = {}
            workflow_tuple['latent_data']['noise_type'] = noise_type
            workflow_tuple['latent_data']['alpha_exponent'] = alpha_exponent
            workflow_tuple['latent_data']['modulator'] = modulator
            workflow_tuple['latent_data']['device'] = device

        return {'samples': latents}, tensors, workflow_tuple

class PrimereCLIP:
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING", "STRING", "STRING", "STRING", "TUPLE")
    RETURN_NAMES = ("COND+", "COND-", "PROMPT+", "PROMPT-", "T5XXL_PROMPT", "PROMPT L+", "PROMPT L-", "WORKFLOW_TUPLE")
    FUNCTION = "clip_encode"
    CATEGORY = TREE_DASHBOARD

    @staticmethod
    def get_default_neg(toml_path: str):
        with open(toml_path, "rb") as f:
            style_def_neg = tomli.load(f)
        return style_def_neg
    @classmethod
    def INPUT_TYPES(cls):
        DEF_TOML_DIR = os.path.join(PRIMERE_ROOT, 'Toml')
        cls.default_neg = cls.get_default_neg(os.path.join(DEF_TOML_DIR, "default_neg.toml"))
        cls.default_pos = cls.get_default_neg(os.path.join(DEF_TOML_DIR, "default_pos.toml"))
        CLIPLIST = folder_paths.get_filename_list("clip")
        CLIPLIST += folder_paths.get_filename_list("clip_gguf")
        T5_DIR = os.path.join(folder_paths.models_dir, 't5')
        if os.path.isdir(T5_DIR):
            folder_paths.add_model_folder_path("p_t5", T5_DIR)
            T5models = folder_paths.get_filename_list("p_t5")
            T5List = folder_paths.filter_files_extensions(T5models, ['.bin', '.safetensors'])
            CLIPLIST += T5List
        cls.CLIPLIST = CLIPLIST

        return {
            "required": {
                "clip": ("CLIP", {"forceInput": True}),
                "concept_data": ("TUPLE", {"default": None, "forceInput": True}),
                "positive_prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": ("STRING", {"forceInput": True}),
                "negative_strength": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.01}),
                "use_int_style": ("BOOLEAN", {"default": False}),
                "int_style_pos": (['None'] + sorted(list(cls.default_pos.keys())),),
                "int_style_pos_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "int_style_neg": (['None'] + sorted(list(cls.default_neg.keys())),),
                "int_style_neg_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "adv_encode": ("BOOLEAN", {"default": False}),
                "token_normalization": (["none", "mean", "length", "length+mean"], {"default": "mean"}),
                "weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"], {"default": "comfy++"}),
            },
            "optional": {
                "enhanced_prompt": ("STRING", {"forceInput": True}),
                "enhanced_prompt_usage": (['None', 'Add', 'Replace', 'T5-XXL'], {"default": "T5-XXL"}),
                "enhanced_prompt_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "edit_image_list": ("IMAGE", {"forceInput": True}, {"default": None}),
                "edit_vae": ("VAE", {"forceInput": True}, {"default": None}),
                "model_keywords": ("MODEL_KEYWORD", {"forceInput": True}),
                "lora_keywords": ("MODEL_KEYWORD", {"forceInput": True}),
                "lycoris_keywords": ("MODEL_KEYWORD", {"forceInput": True}),
                "embedding_pos": ("EMBEDDING", {"forceInput": True}),
                "embedding_neg": ("EMBEDDING", {"forceInput": True}),

                "opt_pos_prompt": ("STRING", {"forceInput": True}),
                "opt_pos_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "opt_neg_prompt": ("STRING", {"forceInput": True}),
                "opt_neg_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),

                "style_handling": ("BOOLEAN", {"default": False, "label_on": "Style-prompt separation", "label_off": "Style-prompt merge"}),
                "style_position": ("BOOLEAN", {"default": False, "label_on": "Style to front of prompt", "label_off": "Style to end of prompt"}),
                "style_swap": ("BOOLEAN", {"default": False, "label_on": "Style to default clip - prompt to T5/L", "label_off": "Style to T5/L - prompt to default clip"}),

                "style_pos_prompt": ("STRING", {"forceInput": True}),
                "style_pos_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "style_neg_prompt": ("STRING", {"forceInput": True}),
                "style_neg_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),

                "positive_l": ("STRING", {"forceInput": True}),
                "negative_l": ("STRING", {"forceInput": True}),
                "l_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION, "forceInput": True}),
                "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION, "forceInput": True}),
                "workflow_tuple": ("TUPLE", {"default": None}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT"
            }
        }

    def clip_encode(self, clip, concept_data, negative_strength, int_style_pos_strength, int_style_neg_strength, opt_pos_strength, opt_neg_strength, style_pos_strength, style_neg_strength, style_handling, style_swap, enhanced_prompt_strength, int_style_pos, int_style_neg, adv_encode, token_normalization, weight_interpretation, l_strength, extra_pnginfo, prompt, copy_prompt_to_l=True, width=1024, height=1024, positive_prompt="", negative_prompt="", enhanced_prompt="", enhanced_prompt_usage="T5-XXL", clip_model='Default', longclip_model='Default', model_keywords=None, lora_keywords=None, lycoris_keywords=None, embedding_pos=None, embedding_neg=None, opt_pos_prompt="", opt_neg_prompt="", style_position=False, style_neg_prompt="", style_pos_prompt="", positive_l="", negative_l="", use_int_style=False, edit_image_list=None, edit_vae=None, workflow_tuple=None):
        model_concept = concept_data.get('model_concept', 'SD1')

        advanced_default = ['StableCascade', 'Chroma', 'KwaiKolors', 'Flux', "Z-Image", 'Pony', 'SD1', 'SD2', 'SD3', 'Lightning', 'Hunyuan', 'QwenGen', 'QwenEdit', 'AuraFlow']
        if model_concept in advanced_default:
            adv_encode = False

        positive_text, negative_text, t5xxl_prompt, positive_l, negative_l = clipping.build_prompt_context(
            model_concept, positive_prompt, negative_prompt,
            enhanced_prompt, enhanced_prompt_usage, enhanced_prompt_strength,
            style_pos_prompt, style_neg_prompt,
            style_handling, style_swap, style_position,
            style_pos_strength, style_neg_strength,
            opt_pos_prompt, opt_neg_prompt, opt_pos_strength, opt_neg_strength,
            negative_strength,
            int_style_pos, int_style_neg, int_style_pos_strength, int_style_neg_strength,
            use_int_style, self.default_pos, self.default_neg,
            l_strength, positive_l, negative_l,
            model_keywords, lora_keywords, lycoris_keywords,
            embedding_pos, embedding_neg,
        )

        match model_concept:
            case 'SD3':
                return clipping.encode_sd3(clip, positive_text, negative_text, t5xxl_prompt, workflow_tuple)
            case 'StableCascade':
                return clipping.encode_stable_cascade(clip, positive_text, negative_text, workflow_tuple)
            case 'Flux':
                return clipping.encode_flux(clip, positive_text, negative_text, t5xxl_prompt, workflow_tuple)
            case 'PixartSigma':
                return clipping.encode_pixart_sigma(clip, positive_text, negative_text, workflow_tuple)
            case 'SANA1024' | 'SANA512':
                return clipping.encode_sana(clip, positive_text, negative_text, t5xxl_prompt, workflow_tuple)
            case 'KwaiKolors':
                return clipping.encode_kolors(clip, positive_text, negative_text, t5xxl_prompt, workflow_tuple)
            case 'Hunyuan':
                return clipping.encode_hunyuan(self, clip, positive_text, negative_text, t5xxl_prompt, workflow_tuple)
            case 'QwenEdit':
                return clipping.encode_qwen_edit(self, clip, positive_text, negative_text, t5xxl_prompt, edit_vae, edit_image_list, workflow_tuple)
            # case 'Chroma':
            #    return clipping.encode_chroma(clip, positive_text, negative_text, workflow_tuple)
            case _:
                clip = clipping.apply_clip_overrides(self, clip, workflow_tuple)
                return clipping.encode_standard(clip, positive_text, negative_text, t5xxl_prompt, adv_encode, token_normalization, weight_interpretation, positive_l, negative_l, width, height, workflow_tuple, advanced_encode)

class PrimereResolution:
    RETURN_TYPES = ("INT", "INT", "INT", "STRING")
    RETURN_NAMES = ("WIDTH", "HEIGHT", "SQUARE_SHAPE", "ASPECT_RATIO")
    FUNCTION = "calculate_imagesize"
    CATEGORY = TREE_DASHBOARD

    @staticmethod
    def get_ratios(toml_path: str):
        with open(toml_path, "rb") as f:
            image_ratios = tomli.load(f)
        return image_ratios

    @classmethod
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
                "manual_res": (utility.VALID_SHAPES, {"default": utility.VALID_SHAPES[2]}),
                "rnd_orientation": ("BOOLEAN", {"default": False}),
                "orientation": (["Horizontal", "Vertical"], {"default": "Horizontal"}),
                "round_to_standard": ("BOOLEAN", {"default": False}),

                "calculate_by_custom": ("BOOLEAN", {"default": False}),
                "custom_side_a": ("FLOAT", {"default": 1.6, "min": 1.0, "max": 100.0, "step": 0.05}),
                "custom_side_b": ("FLOAT", {"default": 2.8, "min": 1.0, "max": 100.0, "step": 0.05}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": -1, "max": utility.MAX_SEED, "forceInput": True}),
                "model_version": ("STRING", {"default": 'SDXL', "forceInput": True}),
                "model_concept": ("STRING", {"default": "Auto", "forceInput": True}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT"
            }
        }

    def calculate_imagesize(self, ratio: str, resolution: bool, rnd_orientation: bool, orientation: str, round_to_standard: bool, calculate_by_custom: bool, custom_side_a: float, custom_side_b: float, seed: int = 0, model_version: str = "SDXL", model_concept: str = 'Auto', **kwargs):
        square_shape = kwargs['manual_res']

        if seed < 1:
            seed = random.randint(0, 9)

        if rnd_orientation == True:
            if (seed % 2) == 0:
                orientation = "Horizontal"
            else:
                orientation = "Vertical"

        if resolution == True:
            if model_version == model_concept and model_concept == 'Hyper':
                WORKFLOWDATA = kwargs['extra_pnginfo']['workflow']['nodes']
                OriginalBaseModel = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereVisualCKPT', 'base_model', kwargs['prompt'])
                fullpathFile = folder_paths.get_full_path('checkpoints', OriginalBaseModel)
                is_link = os.path.islink(str(fullpathFile))
                if is_link == False:
                    model_version = utility.getModelType(OriginalBaseModel, 'checkpoints')
            square_shape = utility.getResolutionByType(model_version)
        else:
            input_string = model_version.lower() + '_res'
            if input_string in kwargs:
                square_shape = int(kwargs[input_string])

        if square_shape is None:
            square_shape = kwargs['manual_res']

        standard = 'STANDARD'
        if model_version == 'StableCascade':
            standard = 'CASCADE'
        if model_version == 'PixartSigma':
            round_to_standard = True

        dimensions = utility.get_dimensions_by_shape(self, ratio, square_shape, orientation, round_to_standard, calculate_by_custom, custom_side_a, custom_side_b, standard)
        dimension_x = dimensions[0]
        dimension_y = dimensions[1]

        return dimension_x, dimension_y, square_shape, f"{dimensions[2]}:{dimensions[3]}"

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

    def multiply_imagesize_mpx(self, width: int, height: int, use_multiplier: bool, upscale_to_mpx: int, triggered_prescale=False, image=None, area_trigger_mpx=0.60, area_target_mpx=1.05, upscale_model='None', upscale_method='bicubic'):
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
                        loaded_upscale_model = nodes_upscale_model.UpscaleModelLoader.execute(upscale_model)[0]
                        prescaledImage = nodes_upscale_model.ImageUpscaleWithModel.execute(loaded_upscale_model, image)[0]

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

        try:
            comfy.model_management.soft_empty_cache()
            comfy.model_management.cleanup_models(True)
        except Exception:
            print('No need to clear cache...')

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
                loaded_upscale_model = nodes_upscale_model.UpscaleModelLoader.execute(upscale_model)[0]

                reference_image_model = nodes_upscale_model.ImageUpscaleWithModel.execute(loaded_upscale_model, reference_image)[0]
                reference_image = nodes.ImageScale.upscale(self, reference_image_model, upscale_method, ref_width_resized, ref_height_resized, "disabled")[0]

                if keep_slave_ratio == True:
                    slave_image_model = nodes_upscale_model.ImageUpscaleWithModel.execute(loaded_upscale_model, slave_image)[0]
                    slave_image = nodes.ImageScaleBy.upscale(self, slave_image_model, upscale_method, slave_squareDiff)[0]

                slave_image = nodes.ImageScale.upscale(self, slave_image, upscale_method, slave_width_resized, slave_height_resized, "disabled")[0]

        else:
            ref_width_resized = ref_width
            ref_height_resized = ref_height
            slave_width_resized = slave_width
            slave_height_resized = slave_height

        return (ref_width, ref_height, slave_width, slave_height, reference_image, ref_width_resized, ref_height_resized, slave_image, slave_width_resized, slave_height_resized)

class PrimereClearNetworkTagsPrompt:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("PROMPT+", "PROMPT-")
    FUNCTION = "clean_network_tags_prompt"
    CATEGORY = TREE_DASHBOARD

    @classmethod
    def INPUT_TYPES(cls):
        CONCEPT_LIST = utility.SUPPORTED_MODELS[0:26]
        CONCEPT_INPUTS = {}
        for concept in CONCEPT_LIST:
            CONCEPT_INPUTS["remove_from_" + concept.lower()] = ("BOOLEAN", {"default": True, "label_on": "REMOVE " + concept.upper(), "label_off": "KEEP " + concept.upper()})

        return {
            "required": {
                "model_version": ("STRING", {"default": 'SD1', "forceInput": True}),
                "positive_prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": ("STRING", {"forceInput": True}),
                "remove_embedding": ("BOOLEAN", {"default": False}),
                "remove_lora": ("BOOLEAN", {"default": False}),
                "remove_lycoris": ("BOOLEAN", {"default": False}),
                "remove_hypernetwork": ("BOOLEAN", {"default": False}),
                "auto_remover": ("BOOLEAN", {"default": True, "label_on": "Auto clean", "label_off": "Manual clean"}),
                **CONCEPT_INPUTS
            }
        }

    def clean_network_tags_prompt(self, positive_prompt, negative_prompt, remove_embedding, remove_lora, remove_lycoris, remove_hypernetwork, auto_remover = True, model_version='SD1', **kwargs):
        clean_state = True

        if auto_remover == True:
            NETWORK_START_GETVER = []
            NETWORK_START_GETVER.append('<lora:')
            NETWORK_START_GETVER.append('<lyco:')
            NETWORK_START_GETVER.append('embedding:')
            NETWORK_END = ['\n', '>', ' ', ',', '}', ')', '|'] + NETWORK_START_GETVER

            NETWORK_TUPLE_POS = utility.get_networks_prompt(NETWORK_START_GETVER, NETWORK_END, positive_prompt)
            NETWORK_TUPLE_NEG = utility.get_networks_prompt(NETWORK_START_GETVER, NETWORK_END, negative_prompt)

            if (len(NETWORK_TUPLE_POS) > 0 or len(NETWORK_TUPLE_NEG) > 0):
                LoraList = folder_paths.get_filename_list("loras")
                EmbeddingList = folder_paths.get_filename_list("embeddings")
                LYCO_DIR = os.path.join(folder_paths.models_dir, 'lycoris')
                folder_paths.add_model_folder_path("lycoris", LYCO_DIR)
                LyCORIS = folder_paths.get_filename_list("lycoris")
                LycorisList = folder_paths.filter_files_extensions(LyCORIS, ['.ckpt', '.safetensors'])

                if (len(NETWORK_TUPLE_POS) > 0):
                    for NETWORK_DATA in NETWORK_TUPLE_POS:
                        NetworkName = NETWORK_DATA[0]
                        NetworkType = NETWORK_DATA[2]
                        if NetworkType == 'LORA' and remove_lora == True:
                            lora_name = utility.get_closest_element(NetworkName, LoraList)
                            modelname_only = Path(lora_name).stem
                            network_model_version = utility.get_value_from_cache('lora_version', modelname_only)
                            if model_version != network_model_version:
                                positive_prompt = utility.clear_prompt(NETWORK_START_GETVER, NETWORK_END, positive_prompt, modelname_only)

                        if NetworkType == 'LYCORIS' and remove_lycoris == True:
                            lycoris_name = utility.get_closest_element(NetworkName, LycorisList)
                            modelname_only = Path(lycoris_name).stem
                            network_model_version = utility.get_value_from_cache('lycoris_version', modelname_only)
                            if model_version != network_model_version:
                                positive_prompt = utility.clear_prompt(NETWORK_START_GETVER, NETWORK_END, positive_prompt, modelname_only)

                        if NetworkType == 'EMBEDDING' and remove_embedding == True:
                            embed_name = utility.get_closest_element(NetworkName, EmbeddingList)
                            modelname_only = Path(embed_name).stem
                            network_model_version = utility.get_value_from_cache('embedding_version', modelname_only)
                            if model_version != network_model_version:
                                positive_prompt = utility.clear_prompt(NETWORK_START_GETVER, NETWORK_END, positive_prompt, modelname_only)

                if (len(NETWORK_TUPLE_NEG) > 0):
                    for NETWORK_DATA in NETWORK_TUPLE_NEG:
                        NetworkName = NETWORK_DATA[0]
                        NetworkType = NETWORK_DATA[2]
                        if NetworkType == 'LORA' and remove_lora == True:
                            lora_name = utility.get_closest_element(NetworkName, LoraList)
                            modelname_only = Path(lora_name).stem
                            network_model_version = utility.get_value_from_cache('lora_version', modelname_only)
                            if model_version != network_model_version:
                                negative_prompt = utility.clear_prompt(NETWORK_START_GETVER, NETWORK_END, negative_prompt, modelname_only)

                        if NetworkType == 'LYCORIS' and remove_lycoris == True:
                            lycoris_name = utility.get_closest_element(NetworkName, LycorisList)
                            modelname_only = Path(lycoris_name).stem
                            network_model_version = utility.get_value_from_cache('lycoris_version', modelname_only)
                            if model_version != network_model_version:
                                negative_prompt = utility.clear_prompt(NETWORK_START_GETVER, NETWORK_END, negative_prompt, modelname_only)

                        if NetworkType == 'EMBEDDING' and remove_embedding == True:
                            embed_name = utility.get_closest_element(NetworkName, EmbeddingList)
                            modelname_only = Path(embed_name).stem
                            network_model_version = utility.get_value_from_cache('embedding_version', modelname_only)
                            if model_version != network_model_version:
                                negative_prompt = utility.clear_prompt(NETWORK_START_GETVER, NETWORK_END, negative_prompt, modelname_only)

            return (positive_prompt, negative_prompt,)

        else:
            input_data = kwargs
            SUPPORTED_CONCEPTS = utility.SUPPORTED_MODELS
            SUPPORTED_CONCEPTS_UC = [x.upper() for x in SUPPORTED_CONCEPTS]
            concept_processor = []
            for inputKey, inputValue in input_data.items():
                if inputKey.startswith("remove_from_") == True:
                    conceptSignUC = inputKey[len("remove_from_"):].upper()
                    conceptIndex = SUPPORTED_CONCEPTS_UC.index(conceptSignUC)
                    CONCEPT_SIGN = SUPPORTED_CONCEPTS[conceptIndex]
                    concept_processor.append(inputValue)
                    if inputValue == False and model_version == CONCEPT_SIGN:
                        clean_state = False

        if clean_state == False:
            return (positive_prompt, negative_prompt,)

        NETWORK_START = []
        if remove_embedding == True:
            NETWORK_START.append('embedding:')

        if remove_lora == True:
            NETWORK_START.append('<lora:')

        if remove_lycoris == True:
            NETWORK_START.append('<lyco:')

        if remove_hypernetwork == True:
            NETWORK_START.append('<hypernet:')

        '''if remove_a1111_embedding == True:
            positive_prompt = positive_prompt.replace('embedding:', '')
            negative_prompt = negative_prompt.replace('embedding:', '')
            EMBEDDINGS = folder_paths.get_filename_list("embeddings")
            for embeddings_path in EMBEDDINGS:
                path = Path(embeddings_path)
                embedding_name = path.stem
                positive_prompt = re.sub("(\(" + embedding_name + ":\d+\.\d+\))|(\(" + embedding_name + ":\d+\))|(" + embedding_name + ":\d+\.\d+)|(" + embedding_name + ":\d+)|(" + embedding_name + ":)|(\(" + embedding_name + "\))|(" + embedding_name + ")", "", positive_prompt)
                negative_prompt = re.sub("(\(" + embedding_name + ":\d+\.\d+\))|(\(" + embedding_name + ":\d+\))|(" + embedding_name + ":\d+\.\d+)|(" + embedding_name + ":\d+)|(" + embedding_name + ":)|(\(" + embedding_name + "\))|(" + embedding_name + ")", "", negative_prompt)
                positive_prompt = re.sub(r'(, )\1+', r', ', positive_prompt).strip(', ').replace(' ,', ',')
                negative_prompt = re.sub(r'(, )\1+', r', ', negative_prompt).strip(', ').replace(' ,', ',')'''

        if len(NETWORK_START) > 0:
            NETWORK_END = ['\n', '>', ' ', ',', '}', ')', '|'] + NETWORK_START
            positive_prompt = utility.clear_prompt(NETWORK_START, NETWORK_END, positive_prompt)
            negative_prompt = utility.clear_prompt(NETWORK_START, NETWORK_END, negative_prompt)

        return (positive_prompt, negative_prompt,)

class PrimereDiTPurifyPrompt:
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("PROMPT+", "PROMPT-")
    FUNCTION = "dit_purify_prompt"
    CATEGORY = TREE_DASHBOARD

    @classmethod
    def INPUT_TYPES(cls):
        CONCEPT_LIST = utility.SUPPORTED_MODELS[0:26]
        CONCEPT_INPUTS = {}
        for concept in CONCEPT_LIST:
            CONCEPT_INPUTS["purify_" + concept.lower()] = ("BOOLEAN", {"default": True, "label_on": "PURIFY " + concept.upper(), "label_off": "KEEP " + concept.upper()})

        return {
            "required": {
                "model_version": ("STRING", {"default": 'SD1', "forceInput": True}),
                "positive_prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": ("STRING", {"forceInput": True}),
                "max_length": ("INT", {"default": 0, "min": 0, "max": 100000, "step": 10}),
                **CONCEPT_INPUTS
            }
        }

    def dit_purify_prompt(self, positive_prompt, negative_prompt, model_version = "SD1", max_length = 0, **kwargs):
        purify_state = True
        input_data = kwargs
        SUPPORTED_CONCEPTS = utility.SUPPORTED_MODELS
        SUPPORTED_CONCEPTS_UC = [x.upper() for x in SUPPORTED_CONCEPTS]
        concept_processor = []
        for inputKey, inputValue in input_data.items():
            if inputKey.startswith("purify_") == True:
                conceptSignUC = inputKey[len("purify_"):].upper()
                conceptIndex = SUPPORTED_CONCEPTS_UC.index(conceptSignUC)
                CONCEPT_SIGN = SUPPORTED_CONCEPTS[conceptIndex]
                concept_processor.append(inputValue)
                if inputValue == False and model_version == CONCEPT_SIGN:
                    purify_state = False

        if purify_state == False:
            return (positive_prompt, negative_prompt,)

        positive_prompt = utility.DiT_cleaner(positive_prompt, max_length)
        negative_prompt = utility.DiT_cleaner(negative_prompt, max_length)

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

    def load_networks(self, model, clip, positive_prompt, process_lora, process_lycoris, process_hypernetwork, copy_weight_to_clip, lora_clip_custom_weight, lycoris_clip_custom_weight, use_lora_keyword, use_lycoris_keyword, lora_keyword_placement, lycoris_keyword_placement, lora_keyword_selection, lycoris_keyword_selection, lora_keywords_num, lycoris_keywords_num, lora_keyword_weight, lycoris_keyword_weight, hypernetwork_safe_load=True, workflow_tuple=None):
        if workflow_tuple is not None and len(workflow_tuple) > 0 and 'setup_states' in workflow_tuple and 'exif_status' in workflow_tuple and workflow_tuple['exif_status'] == 'SUCCEED':
            concept = 'Auto'
            stack_version = workflow_tuple['model_version']
            if 'model_concept' in workflow_tuple:
                concept = workflow_tuple['model_concept']
            if 'model_version' in workflow_tuple:
                if concept == 'Auto' and workflow_tuple['model_version'] == 'SDXL':
                    stack_version = 'SDXL'

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

        LYCO_DIR = os.path.join(folder_paths.models_dir, 'lycoris')
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
                            try:
                                cloned_model, cloned_clip = comfy.sd.load_lora_for_models(cloned_model, cloned_clip, lora, NetworkStrenght, lora_clip_custom_weight)
                            except Exception:
                                print('Lora load error...')
                                use_lora_keyword = False

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
                            try:
                                cloned_model, cloned_clip = comfy.sd.load_lora_for_models(cloned_model, cloned_clip, lycoris, NetworkStrenght, lycoris_clip_custom_weight)
                            except Exception:
                                print('Lycoris load error...')
                                use_lycoris_keyword = False

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
                "use_model_keyword": ("BOOLEAN", {"default": False}),
                "model_keyword_placement": (["First", "Last"], {"default": "Last"}),
                # "model_keyword_selection": (["Select in order", "Random select"], {"default": "Select in order"}),
                "model_keywords_num": ("INT", {"default": 1, "min": 1, "max": 50, "step": 1}),
                "model_keyword_weight": ("FLOAT", {"default": 1.0, "min": 0, "max": 10.0, "step": 0.1}),
                "select_keyword": (utility.KEYWORD_SELECTOR_VALUES, {"default": "Select in order"}),
            },
            "optional": {
                "model_name": ('CHECKPOINT_NAME', {"forceInput": True, "default": None})
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
                is_link = os.path.islink(str(ckpt_path))
                ModelKvHash = None
                if is_link == False:
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
                "model_name": (folder_paths.get_filename_list("upscale_models"),),
            }
        }

    def load_upscaler(self, model_name):
        out = nodes_upscale_model.UpscaleModelLoader.execute(model_name)[0]
        return (out, model_name,)