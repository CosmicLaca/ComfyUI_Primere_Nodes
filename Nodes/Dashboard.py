import math
from ..components.tree import TREE_DASHBOARD
from ..components.tree import PRIMERE_ROOT
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

class PrimereConceptDataTuple:
    RETURN_TYPES = ("TUPLE",)
    RETURN_NAMES = ("CONCEPT_DATA",)
    FUNCTION = "load_concept_collector"
    CATEGORY = TREE_DASHBOARD

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"forceInput": True}),
                "scheduler_name": (comfy.samplers.KSampler.SCHEDULERS, {"forceInput": True}),
                "steps": ("INT", {"forceInput": True}),
                "cfg": ("FLOAT", {"forceInput": True}),

                "override_steps": ("OVERRIDE_STEPS", {"default": False, "forceInput": True}),
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
                "use_flux_turbo_lora": ("FLUX_TURBO_LORA", {"forceInput": True}),
                "flux_turbo_lora_type": ("STRING", {"forceInput": True}),
                "flux_turbo_lora_step": ("INT", {"forceInput": True}),
                "flux_turbo_lora_strength": ("FLOAT", {"default": 0.125, "forceInput": True}),
                "use_flux_srpo_lora": ("FLUX_SRPO_LORA", {"forceInput": True}),
                "use_flux_srpo_svdq_lora": ("FLUX_SRPO_SVDQ_LORA", {"forceInput": True}),
                "flux_srpo_lora_type": ("STRING", {"default": "oficial", "forceInput": True}),
                "flux_srpo_lora_rank": ("INT", {"default": 8, "forceInput": True}),
                "flux_srpo_lora_strength": ("FLOAT", {"default": 1.000, "forceInput": True}),
                "use_flux_nunchaku_lora": ("FLUX_NUNCHAKU_LORA", {"forceInput": True}),
                "flux_nunchaku_lora_type": ("STRING", {"default": "anything_extracted", "forceInput": True}),
                "flux_nunchaku_lora_rank": ("INT", {"default": 64, "forceInput": True}),
                "flux_nunchaku_lora_strength": ("FLOAT", {"default": 1.000, "forceInput": True}),

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

                "pixart_model_type": ("STRING", {"forceInput": True}),
                "pixart_T5_encoder": ("STRING", {"forceInput": True}),
                "pixart_vae": ("STRING", {"forceInput": True}),
                "pixart_denoise": ("FLOAT", {"forceInput": True}),
                "pixart_refiner_model": ("STRING", {"forceInput": True}),
                "pixart_refiner_sampler": ("STRING", {"forceInput": True}),
                "pixart_refiner_scheduler": ("STRING", {"forceInput": True}),
                "pixart_refiner_cfg": ("FLOAT", {"forceInput": True}),
                "pixart_refiner_steps": ("INT", {"forceInput": True}),
                "pixart_refiner_start": ("INT", {"forceInput": True}),
                "pixart_refiner_denoise": ("FLOAT", {"forceInput": True}),
                "pixart_refiner_ignore_prompt": ("BOOLEAN", {"forceInput": True}),

                "sana_model": ("STRING", {"forceInput": True}),
                "sana_encoder": ("STRING", {"forceInput": True}),
                "sana_vae": ("STRING", {"forceInput": True}),
                "sana_weight_dtype": ("STRING", {"forceInput": True}),
                "sana_precision": ("STRING", {"forceInput": True}),

                "qwen_gen_model": ("STRING", {"forceInput": True}),
                "qwen_gen_clip": ("STRING", {"forceInput": True}),
                "qwen_gen_vae":("STRING", {"forceInput": True}),
                "use_qwen_gen_lightning_lora": ("QWEN_GEN_LIGHTNING_LORA", {"forceInput": True}),
                "qwen_gen_lightning_lora_version": ("FLOAT", {"forceInput": True}),
                "qwen_gen_lightning_precision": ("QWEN_GEN_LORA_PRECISION", {"forceInput": True}),
                "qwen_gen_lightning_lora_step": ("INT", {"default": 8, "forceInput": True}),
                "qwen_gen_lightning_lora_strength": ("FLOAT", {"default": 1.00, "forceInput": True}),

                "qwen_edit_model": ("STRING", {"forceInput": True}),
                "qwen_edit_clip": ("STRING", {"forceInput": True}),
                "qwen_edit_vae": ("STRING", {"forceInput": True}),
                "use_qwen_edit_lightning_lora": ("QWEN_EDIT_LIGHTNING_LORA", {"forceInput": True}),
                "qwen_edit_lightning_lora_version": ("FLOAT", {"forceInput": True}),
                "qwen_edit_lightning_precision": ("QWEN_EDIT_LORA_PRECISION", {"forceInput": True}),
                "qwen_edit_lightning_lora_step": ("INT", {"default": 8, "forceInput": True}),
                "qwen_edit_lightning_lora_strength": ("FLOAT", {"default": 1.00, "forceInput": True}),

                "auraflow_clip": ("STRING", {"forceInput": True}),
                "auraflow_vae": ("STRING", {"forceInput": True}),

                "zimage_model": ("STRING", {"forceInput": True}),
                "zimage_clip": ("STRING", {"forceInput": True}),
                "zimage_vae": ("STRING", {"forceInput": True})
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
                          model_concept=None, concept_data=None,
                          clip_selection=True, vae_selection=True, vae_name="Baked",
                          strength_lcm_lora_model=1,
                          lightning_selector='LORA', lightning_model_step=8,
                          strength_lightning_lora_model=1,
                          hypersd_selector='LORA', hypersd_model_step=8,
                          strength_hypersd_lora_model=1,
                          cascade_stage_a=None, cascade_stage_b=None, cascade_stage_c=None, cascade_clip=None,
                          loaded_model=None, loaded_clip=None, loaded_vae=None,
                          flux_selector='DIFFUSION', flux_diffusion=None, flux_weight_dtype=None, flux_gguf=None, flux_clip_t5xxl=None, flux_clip_l=None, flux_clip_guidance=None, flux_vae=None,
                          use_flux_hyper_lora=False, flux_hyper_lora_type='FLUX.1-dev-fp16', flux_hyper_lora_step=8, flux_hyper_lora_strength=0.125, use_flux_turbo_lora=False, flux_turbo_lora_type="TurboAlpha", flux_turbo_lora_step=8, flux_turbo_lora_strength=1,
                          use_flux_srpo_lora=False, use_flux_srpo_svdq_lora=False, flux_srpo_lora_type='oficial', flux_srpo_lora_rank=8, flux_srpo_lora_strength=1, use_flux_nunchaku_lora=False, flux_nunchaku_lora_type='anything_extracted', flux_nunchaku_lora_rank=64, flux_nunchaku_lora_strength=1,
                          hunyuan_clip_t5xxl=None, hunyuan_clip_l=None, hunyuan_vae=None,
                          sd3_clip_g=None, sd3_clip_l=None, sd3_clip_t5xxl=None, sd3_unet_vae=None, use_sd3_hyper_lora=False, sd3_hyper_lora_step=8, sd3_hyper_lora_strength=0.125,
                          qwen_gen_model=None, qwen_gen_clip=None, qwen_gen_vae=None, use_qwen_gen_lightning_lora=False, qwen_gen_lightning_lora_version=1.1, qwen_gen_lightning_precision=True, qwen_gen_lightning_lora_step=8, qwen_gen_lightning_lora_strength=1.00,
                          qwen_edit_model=None, qwen_edit_clip=None, qwen_edit_vae=None, use_qwen_edit_lightning_lora=False, qwen_edit_lightning_lora_version=1.1, qwen_edit_lightning_precision=True, qwen_edit_lightning_lora_step=8, qwen_edit_lightning_lora_strength=1.00,
                          kolors_precision='quant8',
                          pixart_model_type="PixArtMS_Sigma_XL_2", pixart_T5_encoder='None', pixart_vae='None', pixart_denoise=0.9, pixart_refiner_model='None', pixart_refiner_sampler='dpmpp_2m', pixart_refiner_scheduler='Normal', pixart_refiner_cfg=2.0, pixart_refiner_steps=22, pixart_refiner_start=12, pixart_refiner_denoise=0.9, pixart_refiner_ignore_prompt=False,
                          sana_model="None", sana_encoder="None", sana_vae="None", sana_weight_dtype="Auto", sana_precision="fp16",
                          auraflow_vae=None, auraflow_clip=None,
                          zimage_model=None, zimage_clip=None, zimage_vae=None
                          ):

        playground_sigma_max = 120
        playground_sigma_min = 0.002

        try:
            comfy.model_management.soft_empty_cache()
            comfy.model_management.cleanup_models(True)
            comfy.model_management.unload_all_models()
        except Exception:
            print('No need to clear cache...')

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
            if 'use_flux_turbo_lora' in concept_data:
                use_flux_turbo_lora = concept_data['use_flux_turbo_lora']
            if 'flux_turbo_lora_type' in concept_data:
                flux_turbo_lora_type = concept_data['flux_turbo_lora_type']
            if 'flux_turbo_lora_step' in concept_data:
                flux_turbo_lora_step = concept_data['flux_turbo_lora_step']
            if 'flux_turbo_lora_strength' in concept_data:
                flux_turbo_lora_strength = concept_data['flux_turbo_lora_strength']
            if 'use_flux_srpo_lora' in concept_data:
                use_flux_srpo_lora = concept_data['use_flux_srpo_lora']
            if 'use_flux_srpo_svdq_lora' in concept_data:
                use_flux_srpo_svdq_lora = concept_data['use_flux_srpo_svdq_lora']
            if 'flux_srpo_lora_type' in concept_data:
                flux_srpo_lora_type = concept_data['flux_srpo_lora_type']
            if 'flux_srpo_lora_rank' in concept_data:
                flux_srpo_lora_rank = concept_data['flux_srpo_lora_rank']
            if 'flux_srpo_lora_strength' in concept_data:
                flux_srpo_lora_strength = concept_data['flux_srpo_lora_strength']
            if 'use_flux_nunchaku_lora' in concept_data:
                use_flux_nunchaku_lora = concept_data['use_flux_nunchaku_lora']
            if 'flux_nunchaku_lora_type' in concept_data:
                flux_nunchaku_lora_type = concept_data['flux_nunchaku_lora_type']
            if 'flux_nunchaku_lora_rank' in concept_data:
                flux_nunchaku_lora_rank = concept_data['flux_nunchaku_lora_rank']
            if 'flux_nunchaku_lora_strength' in concept_data:
                flux_nunchaku_lora_strength = concept_data['flux_nunchaku_lora_strength']

            if 'hunyuan_clip_t5xxl' in concept_data:
                hunyuan_clip_t5xxl = concept_data['hunyuan_clip_t5xxl']
            if 'hunyuan_clip_l' in concept_data:
                hunyuan_clip_l = concept_data['hunyuan_clip_l']
            if 'hunyuan_vae' in concept_data:
                hunyuan_vae = concept_data['hunyuan_vae']

            if 'zimage_model' in concept_data:
                zimage_model = concept_data['zimage_model']
            if 'zimage_clip' in concept_data:
                zimage_clip = concept_data['zimage_clip']
            if 'zimage_vae' in concept_data:
                zimage_vae = concept_data['zimage_vae']

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

            if 'qwen_gen_model' in concept_data:
                qwen_gen_model = concept_data['qwen_gen_model']
            if 'qwen_gen_clip' in concept_data:
                qwen_gen_clip = concept_data['qwen_gen_clip']
            if 'qwen_gen_vae' in concept_data:
                qwen_gen_vae = concept_data['qwen_gen_vae']
            if 'use_qwen_gen_lightning_lora' in concept_data:
                use_qwen_gen_lightning_lora = concept_data['use_qwen_gen_lightning_lora']
            if 'qwen_gen_lightning_lora_version' in concept_data:
                qwen_gen_lightning_lora_version = concept_data['qwen_gen_lightning_lora_version']
            if 'qwen_gen_lightning_precision' in concept_data:
                qwen_gen_lightning_precision = concept_data['qwen_gen_lightning_precision']
            if 'qwen_gen_lightning_lora_step' in concept_data:
                qwen_gen_lightning_lora_step = concept_data['qwen_gen_lightning_lora_step']
            if 'qwen_gen_lightning_lora_strength' in concept_data:
                qwen_gen_lightning_lora_strength = concept_data['qwen_gen_lightning_lora_strength']

            if 'qwen_edit_model' in concept_data:
                qwen_edit_model = concept_data['qwen_edit_model']
            if 'qwen_edit_clip' in concept_data:
                qwen_edit_clip = concept_data['qwen_edit_clip']
            if 'qwen_edit_vae' in concept_data:
                qwen_edit_vae = concept_data['qwen_edit_vae']
            if 'use_qwen_edit_lightning_lora' in concept_data:
                use_qwen_edit_lightning_lora = concept_data['use_qwen_edit_lightning_lora']
            if 'qwen_edit_lightning_lora_version' in concept_data:
                qwen_edit_lightning_lora_version = concept_data['qwen_edit_lightning_lora_version']
            if 'qwen_edit_lightning_precision' in concept_data:
                qwen_edit_lightning_precision = concept_data['qwen_edit_lightning_precision']
            if 'qwen_edit_lightning_lora_step' in concept_data:
                qwen_edit_lightning_lora_step = concept_data['qwen_edit_lightning_lora_step']
            if 'qwen_edit_lightning_lora_strength' in concept_data:
                qwen_edit_lightning_lora_strength = concept_data['qwen_edit_lightning_lora_strength']

            if 'kolors_precision' in concept_data:
                kolors_precision = concept_data['kolors_precision']

            if 'pixart_model_type' in concept_data:
                pixart_model_type = concept_data['pixart_model_type']
            if 'pixart_T5_encoder' in concept_data:
                pixart_T5_encoder = concept_data['pixart_T5_encoder']
            if 'pixart_vae' in concept_data:
                pixart_vae = concept_data['pixart_vae']
            if 'pixart_denoise' in concept_data:
                pixart_denoise = concept_data['pixart_denoise']
            if 'pixart_refiner_model' in concept_data:
                pixart_refiner_model = concept_data['pixart_refiner_model']
            if 'pixart_refiner_sampler' in concept_data:
                pixart_refiner_sampler = concept_data['pixart_refiner_sampler']
            if 'pixart_refiner_scheduler' in concept_data:
                pixart_refiner_scheduler = concept_data['pixart_refiner_scheduler']
            if 'pixart_refiner_cfg' in concept_data:
                pixart_refiner_cfg = concept_data['pixart_refiner_cfg']
            if 'pixart_refiner_steps' in concept_data:
                pixart_refiner_steps = concept_data['pixart_refiner_steps']
            if 'pixart_refiner_start' in concept_data:
                pixart_refiner_start = concept_data['pixart_refiner_start']
            if 'pixart_refiner_denoise' in concept_data:
                pixart_refiner_denoise = concept_data['pixart_refiner_denoise']
            if 'pixart_refiner_ignore_prompt' in concept_data:
                pixart_refiner_ignore_prompt = concept_data['pixart_refiner_ignore_prompt']

            # if 'sana_model' in concept_data:
            #    sana_model = concept_data['sana_model']
            if 'sana_encoder' in concept_data:
                sana_encoder = concept_data['sana_encoder']
            if 'sana_vae' in concept_data:
                sana_vae = concept_data['sana_vae']
            if 'sana_weight_dtype' in concept_data:
                sana_weight_dtype = concept_data['sana_weight_dtype']
            if 'sana_precision' in concept_data:
                sana_precision = concept_data['sana_precision']

            if 'auraflow_clip' in concept_data:
                auraflow_clip = concept_data['auraflow_clip']
            if 'auraflow_vae' in concept_data:
                auraflow_vae = concept_data['auraflow_vae']

        modelname_only = Path(ckpt_name).stem
        MODEL_VERSION_ORIGINAL = utility.get_value_from_cache('model_version', modelname_only)
        if MODEL_VERSION_ORIGINAL is None:
            MODEL_VERSION_ORIGINAL = utility.getModelType(ckpt_name, 'checkpoints')
            utility.add_value_to_cache('model_version', ckpt_name, MODEL_VERSION_ORIGINAL)

        if model_concept == "LCM" or (model_concept == 'Lightning' and lightning_selector == 'LORA') or (model_concept == 'Hyper' and hypersd_selector == 'LORA'):
            MODEL_VERSION = MODEL_VERSION_ORIGINAL
        else:
            if model_concept is not None:
                MODEL_VERSION = model_concept
            else:
                MODEL_VERSION = MODEL_VERSION_ORIGINAL

        def lcm(self, model, zsnr=False):
            m = model.clone()
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

        sd3_gguf = False
        match model_concept:
            case 'AuraFlow':
                fullpathFile = folder_paths.get_full_path('checkpoints', ckpt_name)
                is_link = os.path.islink(str(fullpathFile))
                if is_link == True:
                    File_link = Path(str(fullpathFile)).resolve()
                    linkName_U = str(folder_paths.folder_names_and_paths["diffusion_models"][0][0])
                    linkName_D = str(folder_paths.folder_names_and_paths["diffusion_models"][0][1])
                    linkedFileName = str(File_link).replace(linkName_U + '\\', '').replace(linkName_D + '\\', '')
                    model_ext = os.path.splitext(linkedFileName)[1].lower()
                    if str(Path(linkName_U).stem) in linkedFileName:
                        linkedFileName = linkedFileName.split(Path(linkName_U).stem + '\\', 1)[1]
                    if str(Path(linkName_D).stem) in linkedFileName:
                        linkedFileName = linkedFileName.split(Path(linkName_D).stem + '\\', 1)[1]
                    if model_ext == '.gguf':
                        OUTPUT_MODEL = gguf_nodes.UnetLoaderGGUF.load_unet(self, linkedFileName)[0]
                    else:
                        OUTPUT_MODEL = nodes.UNETLoader.load_unet(self, linkedFileName, 'default')[0]

                    OUTPUT_VAE = utility.vae_loader_class.load_vae(auraflow_vae)[0]
                    OUTPUT_CLIP = nodes.CLIPLoader.load_clip(self, auraflow_clip, 'stable_diffusion')[0]
                    return (OUTPUT_MODEL,) + (OUTPUT_CLIP,) + (OUTPUT_VAE,) + (MODEL_VERSION,)
                else:
                    OUTPUT_MODEL = nodes.CheckpointLoaderSimple.load_checkpoint(self, ckpt_name)[0]

            case 'SANA1024' | 'SANA512':
                precision = sana_precision
                encoder_path = sana_encoder
                device = model_management.get_torch_device()
                if MODEL_VERSION == MODEL_VERSION_ORIGINAL:
                    fullpathFile = folder_paths.get_full_path('checkpoints', ckpt_name)
                    is_link = os.path.islink(str(fullpathFile))
                    if is_link == True:
                        fullpathFile = Path(str(fullpathFile)).resolve()

                    dtype = utility.get_dtype_by_name(sana_weight_dtype)
                    text_encoder_dir = os.path.join(folder_paths.models_dir, 'text_encoders', encoder_path)
                    if os.path.exists(text_encoder_dir) == False:
                        LLM_PRIMERE_ROOT = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'LLM')
                        text_encoder_dir = os.path.join(LLM_PRIMERE_ROOT, encoder_path)

                    vae_path = folder_paths.get_full_path("vae", sana_vae)
                    text_encoder_dtype = model_management.text_encoder_dtype(device)

                    if concept_data['scheduler_name'] == 'flow_dpm-solver':
                        cfg = create_dc_ae_model_cfg('dc-ae-f32c32-sana-1.0')
                        vae = DCAE(cfg)
                        state_dict = comfy.utils.load_torch_file(vae_path, safe_load=True)
                        vae.load_state_dict(state_dict, strict=False)
                        vae_dtype = model_management.vae_dtype(device, [torch.float16, torch.bfloat16, torch.float32])
                        vae.to(vae_dtype).eval()

                        if "T5" in encoder_path:
                            tokenizer = T5Tokenizer.from_pretrained(str(text_encoder_dir))
                            llm_model = None
                            text_encoder = T5EncoderModel.from_pretrained(str(text_encoder_dir), torch_dtype=text_encoder_dtype)
                        else:
                            tokenizer = AutoTokenizer.from_pretrained(text_encoder_dir)
                            quantization_config = BitsAndBytesConfig(load_in_8bit=True) if precision == '8-bit' else BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=text_encoder_dtype) if precision == '4-bit' else None
                            llm_model = AutoModelForCausalLM.from_pretrained(text_encoder_dir, quantization_config=quantization_config, torch_dtype=text_encoder_dtype) if '-4bit' not in encoder_path else AutoModelForCausalLM.from_pretrained(text_encoder_dir, torch_dtype=text_encoder_dtype)
                            tokenizer.padding_side = "right"
                            text_encoder = llm_model.get_decoder()

                        text_encoder.to(device)
                        state_dict = comfy.utils.load_torch_file(str(fullpathFile), safe_load=True)
                        is_1600M = state_dict['final_layer.scale_shift_table'].shape[1] == 2240  # 1.6b: 2240 0.6b: 1152
                        if '512px' in ckpt_name:
                            config_path = os.path.join(PRIMERE_ROOT, 'components', 'sana', 'configs', 'sana_config', '512ms', 'Sana_1600M_img512.yaml') if is_1600M else os.path.join(PRIMERE_ROOT, 'components', 'sana', 'configs', 'sana_config', '512ms', 'Sana_600M_img512.yaml')
                        else:
                            config_path = os.path.join(PRIMERE_ROOT, 'components', 'sana', 'configs', 'sana_config', '1024ms', 'Sana_1600M_img1024_AdamW.yaml') if is_1600M else os.path.join(PRIMERE_ROOT, 'components', 'sana', 'configs', 'sana_config', '1024ms', 'Sana_600M_img1024.yaml')
                        config = pyrallis.load(SanaConfig, open(config_path))

                        pred_sigma = getattr(config.scheduler, "pred_sigma", True)
                        learn_sigma = getattr(config.scheduler, "learn_sigma", True) and pred_sigma
                        image_size = config.model.image_size
                        latent_size = image_size // config.vae.vae_downsample_rate
                        model_kwargs = {
                            "input_size": latent_size,
                            "pe_interpolation": config.model.pe_interpolation,
                            "config": config,
                            "model_max_length": config.text_encoder.model_max_length,
                            "qk_norm": config.model.qk_norm,
                            "micro_condition": config.model.micro_condition,
                            "caption_channels": text_encoder.config.hidden_size,  # Gemma2: 2304
                            "y_norm": config.text_encoder.y_norm,
                            "attn_type": config.model.attn_type,
                            "ffn_type": config.model.ffn_type,
                            "mlp_ratio": config.model.mlp_ratio,
                            "mlp_acts": list(config.model.mlp_acts),
                            "in_channels": config.vae.vae_latent_dim,
                            "y_norm_scale_factor": config.text_encoder.y_norm_scale_factor,
                            "use_pe": config.model.use_pe,
                            "pred_sigma": pred_sigma,
                            "learn_sigma": learn_sigma,
                            "use_fp32_attention": config.model.get("fp32_attention", False) and config.model.mixed_precision != "bf16",
                        }
                        unet = build_model(config.model.model, **model_kwargs)
                        unet.to(dtype)
                        state_dict = state_dict.get("state_dict", state_dict)
                        if "pos_embed" in state_dict:
                            del state_dict["pos_embed"]
                        missing, unexpected = unet.load_state_dict(state_dict, strict=False)
                        del state_dict
                        unet.eval().to(dtype)
                        pipe = SanaPipeline(config, vae, dtype, unet)

                        SANA_MODEL = {
                            'pipe': pipe,
                            'unet': unet,
                            'text_encoder_model': llm_model,
                            'tokenizer': tokenizer,
                            'text_encoder': text_encoder,
                            'vae': vae,
                            'device': device
                        }

                        SANA_VAE = sana_utils.first_stage_model(vae)
                        SANA_CLIP = sana_utils.cond_stage_model(tokenizer, text_encoder)
                    else:
                        model = list(sana_conf.keys())
                        model_conf = sana_conf[model[1]]
                        SANA_MODEL = load_sana(model_path = str(fullpathFile), model_conf = model_conf)

                        vae_config = vae_conf['dcae-f32c32-sana-1.0']
                        SANA_VAE = EXVAE(vae_path, vae_config, string_to_dtype(sana_weight_dtype.upper(), "vae"))

                        tokenizer = AutoTokenizer.from_pretrained(text_encoder_dir)
                        text_encoder_model = AutoModelForCausalLM.from_pretrained(text_encoder_dir, torch_dtype = text_encoder_dtype)
                        tokenizer.padding_side = "right"
                        text_encoder = text_encoder_model.get_decoder()
                        if device != "cpu":
                            text_encoder = text_encoder.to(device)

                        SANA_CLIP = {
                            "tokenizer": tokenizer,
                            "text_encoder": text_encoder,
                            "text_encoder_model": text_encoder_model,
                        }

                    return (SANA_MODEL,) + (SANA_CLIP,) + (SANA_VAE,) + (MODEL_VERSION,)

            case 'PixartSigma':
                ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
                model_conf = pixart_conf[pixart_model_type]
                PIXART_CHECKPOINT = load_pixart(model_path=ckpt_path, model_conf=model_conf, )
                PIXART_CLIP = nodes.CLIPLoader.load_clip(self, pixart_T5_encoder, 'sd3')[0]

                PIXART_REFINER_CHECKPOINT = {}
                if pixart_refiner_model != 'None':
                    PIXART_REFINER_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, pixart_refiner_model)
                    PIXART_VAE = PIXART_REFINER_CHECKPOINT[2]
                else:
                    PIXART_REFINER_CHECKPOINT[0] = None
                    PIXART_REFINER_CHECKPOINT[1] = None
                    PIXART_VAE = utility.vae_loader_class.load_vae(pixart_vae)[0]
                return ({'main': PIXART_CHECKPOINT, 'refiner': PIXART_REFINER_CHECKPOINT[0]},) + ({'main': PIXART_CLIP, 'refiner': PIXART_REFINER_CHECKPOINT[1]},) + (PIXART_VAE,) + (MODEL_VERSION,)

            case 'Hunyuan':
                HUNYUAN_VAE = utility.vae_loader_class.load_vae(hunyuan_vae)[0]
                T5 = None
                try:
                    LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, ckpt_name)
                    HUNYUAN_MODEL = LOADED_CHECKPOINT[0]
                    CLIP = LOADED_CHECKPOINT[1]
                except Exception:
                    model = 'G/2-1.2'
                    ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
                    model_conf = hydit_conf[model]
                    HUNYUAN_MODEL = load_hydit(model_path=ckpt_path, model_conf=model_conf)

                    dtype = string_to_dtype('FP32', "text_encoder")
                    device = 'GPU'
                    CLIP = load_clip(
                        model_path=folder_paths.get_full_path("clip", hunyuan_clip_l),
                        device=device,
                        dtype=dtype
                    )

                    T5_DIR = os.path.join(folder_paths.models_dir, 't5')
                    if os.path.isdir(T5_DIR):
                        folder_paths.add_model_folder_path("p_t5", T5_DIR)
                        T5FileFullPath = folder_paths.get_full_path("p_t5", hunyuan_clip_t5xxl)
                        if T5FileFullPath is None:
                            T5FileFullPath = folder_paths.get_full_path("clip", hunyuan_clip_t5xxl)

                        T5 = load_t5(
                            model_path=T5FileFullPath,
                            device=device,
                            dtype=dtype
                        )

                HUNYUAN_CLIP = {'clip': CLIP, 't5': T5}
                return (HUNYUAN_MODEL,) + (HUNYUAN_CLIP,) + (HUNYUAN_VAE,) + (MODEL_VERSION,)

            case 'KwaiKolors':
                precision = kolors_precision
                model_name = Path(ckpt_name).stem

                if MODEL_VERSION == MODEL_VERSION_ORIGINAL:
                    fullpathFile = folder_paths.get_full_path('checkpoints', ckpt_name)
                    is_link = os.path.islink(str(fullpathFile))
                    if is_link == True:
                        LinkPath = Path(str(fullpathFile)).resolve()
                        model_name = Path(LinkPath.parent.parent).stem

                dtype = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}['fp16']
                pbar = comfy.utils.ProgressBar(4)
                model_path = os.path.join(folder_paths.models_dir, "diffusers", model_name)
                pbar.update(1)
                scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder='scheduler')
                unet = UNet2DConditionModel.from_pretrained(model_path, subfolder='unet', variant="fp16", revision=None, low_cpu_mem_usage=True).to(dtype).eval()
                pipeline = StableDiffusionXLPipeline(unet=unet, scheduler=scheduler)
                KOLORS_MODEL = {'pipeline': pipeline, 'dtype': dtype}

                pbar = comfy.utils.ProgressBar(2)
                text_encoder_path = os.path.join(model_path, "text_encoder")
                pbar.update(1)
                text_encoder = ChatGLMModel.from_pretrained(text_encoder_path, torch_dtype=torch.float16)
                if precision == 'quant8':
                    try:
                        text_encoder.quantize(8)
                    except Exception:
                        print('Quantitization 8 faliled...')
                elif precision == 'quant4':
                    try:
                        text_encoder.quantize(4)
                    except Exception:
                        print('Quantitization 4 faliled...')
                tokenizer = ChatGLMTokenizer.from_pretrained(text_encoder_path)
                pbar.update(1)
                CHATGLM3_MODEL = {'text_encoder': text_encoder, 'tokenizer': tokenizer}

                if vae_name != "Baked":
                    print('1')
                    OUTPUT_VAE = utility.vae_loader_class.load_vae(vae_name)[0]
                else:
                    vae_list = folder_paths.get_filename_list("vae")
                    allLSDXLvae = list(filter(lambda a: 'sdxl_'.casefold() in a.casefold(), vae_list))
                    OUTPUT_VAE = utility.vae_loader_class.load_vae(allLSDXLvae[0])[0]

                return (KOLORS_MODEL,) + (CHATGLM3_MODEL,) + (OUTPUT_VAE,) + (MODEL_VERSION,)

            case 'StableCascade':
                if cascade_stage_a is not None and cascade_stage_b is not None and cascade_stage_c is not None and cascade_clip is not None:
                    OUTPUT_CLIP_CAS = nodes.CLIPLoader.load_clip(self, cascade_clip, 'stable_cascade')[0]
                    OUTPUT_VAE_CAS = utility.vae_loader_class.load_vae(cascade_stage_a)[0]
                    if MODEL_VERSION == MODEL_VERSION_ORIGINAL:
                        fullpathFile = folder_paths.get_full_path('checkpoints', ckpt_name)
                        is_link = os.path.islink(str(fullpathFile))
                        if is_link == False:
                            MODEL_C_CAS = nodes.UNETLoader.load_unet(self, ckpt_name, 'default')[0]
                        else:
                            File_link = Path(str(fullpathFile)).resolve()
                            linkName_U = str(folder_paths.folder_names_and_paths["diffusion_models"][0][0])
                            linkName_D = str(folder_paths.folder_names_and_paths["diffusion_models"][0][1])
                            linkedFileName = str(File_link).replace(linkName_U + '\\', '').replace(linkName_D + '\\', '')

                            if str(Path(linkName_U).stem) in linkedFileName:
                                linkedFileName = linkedFileName.split(Path(linkName_U).stem + '\\', 1)[1]
                            if str(Path(linkName_D).stem) in linkedFileName:
                                linkedFileName = linkedFileName.split(Path(linkName_D).stem + '\\', 1)[1]

                            MODEL_C_CAS = nodes.UNETLoader.load_unet(self, linkedFileName, 'default')[0]
                    else:
                        MODEL_C_CAS = nodes.UNETLoader.load_unet(self, cascade_stage_c, 'default')[0]
                    MODEL_B_CAS = nodes.UNETLoader.load_unet(self, cascade_stage_b, 'default')[0]

                    OUTPUT_MODEL_CAS = [MODEL_B_CAS, MODEL_C_CAS]
                    return (OUTPUT_MODEL_CAS,) + (OUTPUT_CLIP_CAS,) + (OUTPUT_VAE_CAS,) + (MODEL_VERSION,)

            case 'SD3':
                fullpathFile = folder_paths.get_full_path('checkpoints', ckpt_name)
                is_link = os.path.islink(str(fullpathFile))
                if is_link == True:
                    File_link = Path(str(fullpathFile)).resolve()
                    model_ext = os.path.splitext(File_link)[1].lower()
                    if model_ext == '.gguf':
                        linkName_U = str(folder_paths.folder_names_and_paths["diffusion_models"][0][0])
                        linkName_D = str(folder_paths.folder_names_and_paths["diffusion_models"][0][1])
                        sd3_gguf = str(File_link).replace(linkName_U + '\\', '').replace(linkName_D + '\\', '')
                        if str(Path(linkName_U).stem) in sd3_gguf:
                            sd3_gguf = sd3_gguf.split(Path(linkName_U).stem + '\\', 1)[1]
                        if str(Path(linkName_D).stem) in sd3_gguf:
                            sd3_gguf = sd3_gguf.split(Path(linkName_D).stem + '\\', 1)[1]

            case "Z-Image":
                zimage_weight_dtype = 'default'
                if 'e4m3fn' in ckpt_name:
                    zimage_weight_dtype = 'fp8_e4m3fn'
                if 'e5m2' in ckpt_name:
                    zimage_weight_dtype = 'fp8_e5m2'

                fullpathFile = folder_paths.get_full_path('checkpoints', ckpt_name)
                is_link = os.path.islink(str(fullpathFile))
                if is_link == True:
                    File_link = Path(str(fullpathFile)).resolve()
                    linkName_U = str(folder_paths.folder_names_and_paths["diffusion_models"][0][0])
                    linkName_D = str(folder_paths.folder_names_and_paths["diffusion_models"][0][1])
                    linkedFileName = str(File_link).replace(linkName_U + '\\', '').replace(linkName_D + '\\', '')
                    model_ext = os.path.splitext(linkedFileName)[1].lower()
                    if str(Path(linkName_U).stem) in linkedFileName:
                        linkedFileName = linkedFileName.split(Path(linkName_U).stem + '\\', 1)[1]
                    if str(Path(linkName_D).stem) in linkedFileName:
                        linkedFileName = linkedFileName.split(Path(linkName_D).stem + '\\', 1)[1]

                    if 'diffusion_models' in str(File_link):
                        if model_ext == '.gguf':
                            MODEL_DIFFUSION = gguf_nodes.UnetLoaderGGUF.load_unet(self, linkedFileName)[0]
                        else:
                            MODEL_DIFFUSION = nodes.UNETLoader.load_unet(self, linkedFileName, zimage_weight_dtype)[0]
                    elif 'unet' in str(File_link):
                        if model_ext == '.gguf':
                            MODEL_DIFFUSION = gguf_nodes.UnetLoaderGGUF.load_unet(self, linkedFileName)[0]
                        else:
                            try:
                                MODEL_DIFFUSION = nodes.UNETLoader.load_unet(self, linkedFileName, zimage_weight_dtype)[0]
                            except Exception:
                                MODEL_DIFFUSION = nf4_helper.UNETLoaderNF4.load_nf4unet(linkedFileName)[0]
                else:
                    MODEL_DIFFUSION = nodes.CheckpointLoaderSimple.load_checkpoint(self, ckpt_name)[0]

                clip_ext = os.path.splitext(zimage_clip)[1].lower()
                if clip_ext == '.gguf':
                    ZIMAGE_CLIP = gguf_nodes.CLIPLoaderGGUF.load_clip(self, zimage_clip, 'qwen_image')[0]
                else:
                    ZIMAGE_CLIP = nodes.CLIPLoader.load_clip(self, zimage_clip, 'flux2')[0]

                ZIMAGE_VAE = utility.vae_loader_class.load_vae(zimage_vae)[0]
                return (MODEL_DIFFUSION,) + (ZIMAGE_CLIP,) + (ZIMAGE_VAE,) + (MODEL_VERSION,)

            case 'QwenGen' | 'QwenEdit':
                FULL_LORA_PATH = None
                concept_type = 'qwen_image'
                qwen_weight_dtype = 'default'
                if 'e4m3fn' in ckpt_name:
                    qwen_weight_dtype = 'e4m3fn'

                fullpathFile = folder_paths.get_full_path('checkpoints', ckpt_name)
                qwen_ver_list = re.findall(r'(?<!\d)\d{4}(?!\d)', ckpt_name)
                if len(qwen_ver_list) < 1:
                    qwen_ver = '2509'
                else:
                    qwen_ver = str(qwen_ver_list[0])

                is_link = os.path.islink(str(fullpathFile))
                if is_link == True:
                    File_link = Path(str(fullpathFile)).resolve()
                    linkName_U = str(folder_paths.folder_names_and_paths["diffusion_models"][0][0])
                    linkName_D = str(folder_paths.folder_names_and_paths["diffusion_models"][0][1])
                    linkedFileName = str(File_link).replace(linkName_U + '\\', '').replace(linkName_D + '\\', '')
                    model_ext = os.path.splitext(linkedFileName)[1].lower()
                    if str(Path(linkName_U).stem) in linkedFileName:
                        linkedFileName = linkedFileName.split(Path(linkName_U).stem + '\\', 1)[1]
                    if str(Path(linkName_D).stem) in linkedFileName:
                        linkedFileName = linkedFileName.split(Path(linkName_D).stem + '\\', 1)[1]

                    if 'diffusion_models' in str(File_link):
                        if model_ext == '.gguf':
                            MODEL_DIFFUSION = gguf_nodes.UnetLoaderGGUF.load_unet(self, linkedFileName)[0]
                        else:
                            MODEL_DIFFUSION = nodes.UNETLoader.load_unet(self, linkedFileName, qwen_weight_dtype)[0]
                    elif 'unet' in str(File_link):
                        if model_ext == '.gguf':
                            MODEL_DIFFUSION = gguf_nodes.UnetLoaderGGUF.load_unet(self, linkedFileName)[0]
                        else:
                            try:
                                MODEL_DIFFUSION = nodes.UNETLoader.load_unet(self, linkedFileName, qwen_weight_dtype)[0]
                            except Exception:
                                MODEL_DIFFUSION = nf4_helper.UNETLoaderNF4.load_nf4unet(linkedFileName)[0]
                else:
                    MODEL_DIFFUSION = nodes.CheckpointLoaderSimple.load_checkpoint(self, ckpt_name)[0]

                if model_concept == 'QwenGen':
                    QWEN_CLIP = nodes.CLIPLoader.load_clip(self, qwen_gen_clip, concept_type)[0]
                    QWEN_VAE = utility.vae_loader_class.load_vae(qwen_gen_vae)[0]
                if model_concept == 'QwenEdit':
                    QWEN_CLIP = nodes.CLIPLoader.load_clip(self, qwen_edit_clip, concept_type)[0]
                    QWEN_VAE = utility.vae_loader_class.load_vae(qwen_edit_vae)[0]

                if use_qwen_gen_lightning_lora == True and model_concept == 'QwenGen':
                    QWENGEN_4_V1 = 'https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V1.0.safetensors?download=true'
                    QWENGEN_4_V1_BF16 = 'https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors?download=true'
                    QWENGEN_4_V2 = 'https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V2.0.safetensors?download=true'
                    QWENGEN_4_V2_BF16 = 'https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors?download=true'
                    QWENGEN_8_V1 = 'https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V1.0.safetensors?download=true'
                    QWENGEN_8_V1_1 = 'https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V1.1.safetensors?download=true'
                    QWENGEN_8_V1_1_BF16 = 'https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors?download=true'
                    QWENGEN_8_V2 = 'https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V2.0.safetensors?download=true'
                    QWENGEN_8_V2_BF16 = 'https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors?download=true'

                    DOWNLOADED_QWENGEN_4_V1 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Qwen-Image-Lightning-4steps-V1.0.safetensors')
                    DOWNLOADED_QWENGEN_4_V1_BF16 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Qwen-Image-Lightning-4steps-V1.0-bf16.safetensors')
                    DOWNLOADED_QWENGEN_4_V2 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Qwen-Image-Lightning-4steps-V2.0.safetensors')
                    DOWNLOADED_QWENGEN_4_V2_BF16 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Qwen-Image-Lightning-4steps-V2.0-bf16.safetensors')
                    DOWNLOADED_QWENGEN_8_V1 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Qwen-Image-Lightning-8steps-V1.0.safetensors')
                    DOWNLOADED_QWENGEN_8_V1_1 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Qwen-Image-Lightning-8steps-V1.1.safetensors')
                    DOWNLOADED_QWENGEN_8_V1_1_BF16 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Qwen-Image-Lightning-8steps-V1.1-bf16.safetensors')
                    DOWNLOADED_QWENGEN_8_V2 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Qwen-Image-Lightning-8steps-V2.0.safetensors')
                    DOWNLOADED_QWENGEN_8_V2_BF16 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Qwen-Image-Lightning-8steps-V2.0-bf16.safetensors')

                    utility.fileDownloader(DOWNLOADED_QWENGEN_4_V1, QWENGEN_4_V1)
                    utility.fileDownloader(DOWNLOADED_QWENGEN_4_V1_BF16, QWENGEN_4_V1_BF16)
                    utility.fileDownloader(DOWNLOADED_QWENGEN_4_V2, QWENGEN_4_V2)
                    utility.fileDownloader(DOWNLOADED_QWENGEN_4_V2_BF16, QWENGEN_4_V2_BF16)
                    utility.fileDownloader(DOWNLOADED_QWENGEN_8_V1, QWENGEN_8_V1)
                    utility.fileDownloader(DOWNLOADED_QWENGEN_8_V1_1, QWENGEN_8_V1_1)
                    utility.fileDownloader(DOWNLOADED_QWENGEN_8_V1_1_BF16, QWENGEN_8_V1_1_BF16)
                    utility.fileDownloader(DOWNLOADED_QWENGEN_8_V2, QWENGEN_8_V2)
                    utility.fileDownloader(DOWNLOADED_QWENGEN_8_V2_BF16, QWENGEN_8_V2_BF16)

                    downloaded_filelist_filtered = utility.getDownloadedFiles()
                    if qwen_ver != '2509':
                        allQwenLoras = list(filter(lambda a: 'qwen-image'.casefold() in a.casefold() and 'edit'.casefold() not in a.casefold() and qwen_ver.casefold() in a.casefold(), downloaded_filelist_filtered))
                    else:
                        allQwenLoras = list(filter(lambda a: 'qwen-image'.casefold() in a.casefold() and 'edit'.casefold() not in a.casefold() and (qwen_ver.casefold() in a.casefold() or '25'.casefold() not in a.casefold()), downloaded_filelist_filtered))

                    if qwen_weight_dtype != 'default':
                        allQwenLoras_bytype = list(filter(lambda a: qwen_weight_dtype.casefold() in a.casefold(), allQwenLoras))
                    else:
                        allQwenLoras_bytype = list(filter(lambda a: 'e4m3fn'.casefold() not in a.casefold(), allQwenLoras))

                    if len(allQwenLoras_bytype) > 0:
                        allQwenLoras = allQwenLoras_bytype

                    qwen_gen_ver = f"v{qwen_gen_lightning_lora_version:.1f}"

                    if qwen_gen_lightning_precision == False:
                        finalLoras = list(filter(lambda a: str(qwen_gen_lightning_lora_step) + 'step'.casefold() in a.casefold() and '-bf16'.casefold() in a.casefold() and qwen_gen_ver.casefold() in a.casefold(), allQwenLoras))
                    else:
                        finalLoras = list(filter(lambda a: str(qwen_gen_lightning_lora_step) + 'step'.casefold() in a.casefold() and '-bf16'.casefold() not in a.casefold() and qwen_gen_ver.casefold() in a.casefold(), allQwenLoras))

                    if len(finalLoras) == 0 and len(allQwenLoras) > 0:
                        if qwen_gen_lightning_precision == False:
                            finalLoras_bfcheck = list(filter(lambda a: '-bf16'.casefold() in a.casefold(), allQwenLoras))
                        else:
                            finalLoras_bfcheck = list(filter(lambda a: '-bf16'.casefold() not in a.casefold(), allQwenLoras))
                        if len(finalLoras_bfcheck) > 0:
                            finalLoras = finalLoras_bfcheck[0]
                        else:
                            finalLoras = allQwenLoras[0]

                    if len(finalLoras) > 0:
                        LORA_FILE = finalLoras[0]
                        FULL_LORA_PATH = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', LORA_FILE)

                    print(f'=================== lora check for QWEN-GEN for {ckpt_name} and {qwen_weight_dtype} and V:{qwen_ver} step: {qwen_gen_lightning_lora_step} / {qwen_gen_lightning_precision} ============================')
                    print(finalLoras)

                    if FULL_LORA_PATH is not None and os.path.exists(FULL_LORA_PATH) == True:
                        print(FULL_LORA_PATH)
                        if qwen_gen_lightning_lora_strength != 0:
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
                                MODEL_DIFFUSION = comfy.sd.load_lora_for_models(MODEL_DIFFUSION, None, lora, qwen_gen_lightning_lora_strength, 0)[0]

                    print('===================lora check end============================')

                if use_qwen_edit_lightning_lora == True and model_concept == 'QwenEdit':
                    QWENEDIT_4_V1 = 'https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-Lightning-4steps-V1.0.safetensors?download=true'
                    QWENEDIT_4_V1_BF16 = 'https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors?download=true'
                    QWENEDIT_8_V1 = 'https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors?download=true'
                    QWENEDIT_8_V1_BF16 = 'https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors?download=true'

                    QWENEDIT_4_2509_V1 = 'https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-fp32.safetensors?download=true'
                    QWENEDIT_4_2509_V1_BF16 = 'https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors?download=true'
                    QWENEDIT_8_2509_V1 = 'https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-fp32.safetensors?download=true'
                    QWENEDIT_8_2509_V1_BF16 = 'https://huggingface.co/lightx2v/Qwen-Image-Lightning/resolve/main/Qwen-Image-Edit-2509/Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors?download=true'

                    DOWNLOADED_QWENEDIT_4_V1 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Qwen-Image-Edit-Lightning-4steps-V1.0.safetensors')
                    DOWNLOADED_QWENEDIT_4_V1_BF16 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Qwen-Image-Edit-Lightning-4steps-V1.0-bf16.safetensors')
                    DOWNLOADED_QWENEDIT_8_V1 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Qwen-Image-Edit-Lightning-8steps-V1.0.safetensors')
                    DOWNLOADED_QWENEDIT_8_V1_BF16 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Qwen-Image-Edit-Lightning-8steps-V1.0-bf16.safetensors')

                    DOWNLOADED_QWENEDIT_4_2509_V1 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Qwen-Image-Edit-2509-Lightning-4steps-V1.0.safetensors')
                    DOWNLOADED_QWENEDIT_4_2509_V1_BF16 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Qwen-Image-Edit-2509-Lightning-4steps-V1.0-bf16.safetensors')
                    DOWNLOADED_QWENEDIT_8_2509_V1 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Qwen-Image-Edit-2509-Lightning-8steps-V1.0.safetensors')
                    DOWNLOADED_QWENEDIT_8_2509_V1_BF16 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Qwen-Image-Edit-2509-Lightning-8steps-V1.0-bf16.safetensors')

                    utility.fileDownloader(DOWNLOADED_QWENEDIT_4_V1, QWENEDIT_4_V1)
                    utility.fileDownloader(DOWNLOADED_QWENEDIT_4_V1_BF16, QWENEDIT_4_V1_BF16)
                    utility.fileDownloader(DOWNLOADED_QWENEDIT_8_V1, QWENEDIT_8_V1)
                    utility.fileDownloader(DOWNLOADED_QWENEDIT_8_V1_BF16, QWENEDIT_8_V1_BF16)

                    utility.fileDownloader(DOWNLOADED_QWENEDIT_4_2509_V1, QWENEDIT_4_2509_V1)
                    utility.fileDownloader(DOWNLOADED_QWENEDIT_4_2509_V1_BF16, QWENEDIT_4_2509_V1_BF16)
                    utility.fileDownloader(DOWNLOADED_QWENEDIT_8_2509_V1, QWENEDIT_8_2509_V1)
                    utility.fileDownloader(DOWNLOADED_QWENEDIT_8_2509_V1_BF16, QWENEDIT_8_2509_V1_BF16)

                    downloaded_filelist_filtered = utility.getDownloadedFiles()
                    allQwenLoras = list(filter(lambda a: 'qwen-image-edit'.casefold() in a.casefold() and qwen_ver.casefold() in a.casefold(), downloaded_filelist_filtered))
                    if len(allQwenLoras) < 1:
                        allQwenLoras = list(filter(lambda a: 'qwen-image-edit'.casefold() in a.casefold() and (qwen_ver.casefold() in a.casefold() or '25'.casefold() not in a.casefold()), downloaded_filelist_filtered))

                    if qwen_weight_dtype != 'default':
                        allQwenLoras_bytype = list(filter(lambda a: qwen_weight_dtype.casefold() in a.casefold(), allQwenLoras))
                    else:
                        allQwenLoras_bytype = list(filter(lambda a: 'e4m3fn'.casefold() not in a.casefold(), allQwenLoras))

                    if len(allQwenLoras_bytype) > 0:
                        allQwenLoras = allQwenLoras_bytype

                    qwen_edit_ver = f"v{qwen_edit_lightning_lora_version:.1f}"

                    if qwen_edit_lightning_precision == False:
                        finalLoras = list(filter(lambda a: str(qwen_edit_lightning_lora_step) + 'step'.casefold() in a.casefold() and '-bf16'.casefold() in a.casefold() and qwen_edit_ver.casefold() in a.casefold(), allQwenLoras))
                    else:
                        finalLoras = list(filter(lambda a: str(qwen_edit_lightning_lora_step) + 'step'.casefold() in a.casefold() and '-bf16'.casefold() not in a.casefold() and qwen_edit_ver.casefold() in a.casefold(), allQwenLoras))

                    if len(finalLoras) == 0 and len(allQwenLoras) > 0:
                        if qwen_edit_lightning_precision == False:
                            finalLoras_bfcheck = list(filter(lambda a: '-bf16'.casefold() in a.casefold(), allQwenLoras))
                        else:
                            finalLoras_bfcheck = list(filter(lambda a: '-bf16'.casefold() not in a.casefold(), allQwenLoras))
                        if len(finalLoras_bfcheck) > 0:
                            finalLoras = finalLoras_bfcheck[0]
                        else:
                            finalLoras = allQwenLoras[0]

                    if len(finalLoras) > 0:
                        LORA_FILE = finalLoras[0]
                        FULL_LORA_PATH = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', LORA_FILE)

                    print(f'=================== lora check for QWEN-EDIT for {ckpt_name} and {qwen_weight_dtype} and V:{qwen_ver} step: {qwen_edit_lightning_lora_step} / {qwen_edit_lightning_precision} ============================')
                    print(finalLoras)

                    if FULL_LORA_PATH is not None and os.path.exists(FULL_LORA_PATH) == True:
                        print(FULL_LORA_PATH)
                        if qwen_edit_lightning_lora_strength != 0:
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
                                MODEL_DIFFUSION = comfy.sd.load_lora_for_models(MODEL_DIFFUSION, None, lora, qwen_edit_lightning_lora_strength, 0)[0]

                    print('===================lora check end============================')

                if model_concept == 'QwenEdit':
                    MODEL_DIFFUSION = nodes_model_advanced.ModelSamplingSD3.patch(self, MODEL_DIFFUSION, 3, 1.0)[0]
                    MODEL_DIFFUSION = nodes_cfg.CFGNorm.execute(MODEL_DIFFUSION, 1)[0]

                return (MODEL_DIFFUSION,) + (QWEN_CLIP,) + (QWEN_VAE,) + (MODEL_VERSION,)

            case 'Flux':
                downloaded_filelist_filtered = utility.getDownloadedFiles()
                if flux_selector is not None and flux_diffusion is not None and flux_weight_dtype is not None and flux_gguf is not None and flux_clip_t5xxl is not None and flux_clip_l is not None and flux_clip_guidance is not None and flux_vae is not None:
                    if MODEL_VERSION == MODEL_VERSION_ORIGINAL:
                        flux_selector = 'SAFETENSOR'
                        fullpathFile = folder_paths.get_full_path('checkpoints', ckpt_name)
                        is_link = os.path.islink(str(fullpathFile))
                        if is_link == True:
                            File_link = Path(str(fullpathFile)).resolve()
                            model_ext = os.path.splitext(File_link)[1].lower()
                            match model_ext:
                                case '.gguf':
                                    flux_selector = 'GGUF'
                                    linkName_U = str(folder_paths.folder_names_and_paths["diffusion_models"][0][0])
                                    linkName_D = str(folder_paths.folder_names_and_paths["diffusion_models"][0][1])
                                    flux_gguf = str(File_link).replace(linkName_U + '\\', '').replace(linkName_D + '\\', '')

                                    if str(Path(linkName_U).stem) in flux_gguf:
                                        flux_gguf = flux_gguf.split(Path(linkName_U).stem + '\\', 1)[1]
                                    if str(Path(linkName_D).stem) in flux_gguf:
                                        flux_gguf = flux_gguf.split(Path(linkName_D).stem + '\\', 1)[1]

                    match flux_selector:
                        case 'DIFFUSION':
                            try:
                                MODEL_DIFFUSION = nodes.UNETLoader.load_unet(self, flux_diffusion, flux_weight_dtype)[0]
                            except Exception:
                                MODEL_DIFFUSION = nf4_helper.UNETLoaderNF4.load_nf4unet(flux_diffusion)[0]
                            DUAL_CLIP = nodes.DualCLIPLoader.load_clip(self, flux_clip_t5xxl, flux_clip_l, 'flux')[0]
                            FLUX_VAE = utility.vae_loader_class.load_vae(flux_vae)[0]

                        case 'GGUF':
                            MODEL_DIFFUSION = gguf_nodes.UnetLoaderGGUF.load_unet(self, flux_gguf)[0]
                            DUAL_CLIP = gguf_nodes.DualCLIPLoaderGGUF.load_clip(self, flux_clip_t5xxl, flux_clip_l, 'flux')[0]
                            FLUX_VAE = utility.vae_loader_class.load_vae(flux_vae)[0]

                        case 'SAFETENSOR':
                            if is_link == False:
                                MODEL_DIFFUSION = nodes.CheckpointLoaderSimple.load_checkpoint(self, ckpt_name)[0]
                                DUAL_CLIP = nodes.DualCLIPLoader.load_clip(self, flux_clip_t5xxl, flux_clip_l, 'flux')[0]
                                FLUX_VAE = utility.vae_loader_class.load_vae(flux_vae)[0]
                            else:
                                linkName_U = str(folder_paths.folder_names_and_paths["diffusion_models"][0][0])
                                linkName_D = str(folder_paths.folder_names_and_paths["diffusion_models"][0][1])
                                linkedFileName = str(File_link).replace(linkName_U + '\\', '').replace(linkName_D + '\\', '')

                                if str(Path(linkName_U).stem) in linkedFileName:
                                    linkedFileName = linkedFileName.split(Path(linkName_U).stem + '\\', 1)[1]
                                if str(Path(linkName_D).stem) in linkedFileName:
                                    linkedFileName = linkedFileName.split(Path(linkName_D).stem + '\\', 1)[1]

                                if 'diffusion_models' in str(File_link):
                                    model_ext = os.path.splitext(linkedFileName)[1].lower()
                                    if model_ext == '.gguf':
                                        MODEL_DIFFUSION = gguf_nodes.UnetLoaderGGUF.load_unet(self, linkedFileName)[0]
                                        DUAL_CLIP = gguf_nodes.DualCLIPLoaderGGUF.load_clip(self, flux_clip_t5xxl, flux_clip_l, 'flux')[0]
                                    else:
                                        MODEL_DIFFUSION = nodes.UNETLoader.load_unet(self, linkedFileName, flux_weight_dtype)[0]
                                        DUAL_CLIP = nodes.DualCLIPLoader.load_clip(self, flux_clip_t5xxl, flux_clip_l, 'flux')[0]
                                    FLUX_VAE = utility.vae_loader_class.load_vae(flux_vae)[0]
                                elif 'unet' in str(File_link):
                                    try:
                                        MODEL_DIFFUSION = nodes.UNETLoader.load_unet(self, linkedFileName, flux_weight_dtype)[0]
                                    except Exception:
                                        MODEL_DIFFUSION = nf4_helper.UNETLoaderNF4.load_nf4unet(linkedFileName)[0]
                                    DUAL_CLIP = nodes.DualCLIPLoader.load_clip(self, flux_clip_t5xxl, flux_clip_l, 'flux')[0]
                                    FLUX_VAE = utility.vae_loader_class.load_vae(flux_vae)[0]
                                else:
                                    MODEL_DIFFUSION = nodes.CheckpointLoaderSimple.load_checkpoint(self, linkedFileName)[0]
                                    DUAL_CLIP = nodes.DualCLIPLoader.load_clip(self, flux_clip_t5xxl, flux_clip_l, 'flux')[0]
                                    FLUX_VAE = utility.vae_loader_class.load_vae(flux_vae)[0]

                    finalLoras = None
                    extra_lora_strength = 0
                    if use_flux_srpo_lora == True and use_flux_srpo_svdq_lora == False and use_flux_nunchaku_lora == False:
                        allSRPOFluxLoras = list(filter(lambda a: 'srpo_'.casefold() in a.casefold(), downloaded_filelist_filtered))
                        finalLoras = list(filter(lambda a: 'svdq-'.casefold() not in a.casefold() and f"{flux_srpo_lora_type}".casefold() in a.casefold() and f"_{flux_srpo_lora_rank}_".casefold() in a.casefold(), allSRPOFluxLoras))
                        extra_lora_strength = flux_srpo_lora_strength

                    if use_flux_srpo_svdq_lora == True and use_flux_nunchaku_lora == False:
                        allSRPO_SVDQFluxLoras = list(filter(lambda a: '-srpo_'.casefold() in a.casefold(), downloaded_filelist_filtered))
                        finalLoras = list(filter(lambda a: 'svdq-'.casefold() in a.casefold() and f"{flux_srpo_lora_type}".casefold() in a.casefold() and f"_{flux_srpo_lora_rank}_".casefold() in a.casefold(), allSRPO_SVDQFluxLoras))
                        extra_lora_strength = flux_srpo_lora_strength

                    if use_flux_nunchaku_lora == True:
                        allNunchakuFluxLoras = list(filter(lambda a: '_nunchaku'.casefold() in a.casefold() or 'insert-anything_extracted'.casefold() in a.casefold(), downloaded_filelist_filtered))
                        print(allNunchakuFluxLoras)
                        finalLoras = list(filter(lambda a: f"{flux_nunchaku_lora_type}".casefold() in a.casefold() and (f"_{flux_nunchaku_lora_rank}".casefold() in a.casefold() or (f"_{flux_nunchaku_lora_rank}".casefold() not in a.casefold()) and '_nunchaku'.casefold() in a.casefold()), allNunchakuFluxLoras))
                        extra_lora_strength = flux_nunchaku_lora_strength

                    if finalLoras is not None and type(finalLoras).__name__ == "list" and len(finalLoras) > 0:
                        LORA_FILE = finalLoras[0]
                        print(LORA_FILE)
                        FULL_LORA_PATH = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', LORA_FILE)
                        print(FULL_LORA_PATH)
                        if FULL_LORA_PATH is not None and os.path.exists(FULL_LORA_PATH) == True:
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
                                    lora = comfy.utils.load_torch_file(FULL_LORA_PATH, safe_load=True)
                                    self.loaded_lora = (FULL_LORA_PATH, lora)

                                MODEL_DIFFUSION = comfy.sd.load_lora_for_models(MODEL_DIFFUSION, None, lora, extra_lora_strength, 0)[0]

                    if use_flux_hyper_lora == True:
                        FLUX_DEV_LORA8 = 'https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-FLUX.1-dev-8steps-lora.safetensors?download=true'
                        FLUX_DEV_FP16_LORA8 = 'https://huggingface.co/nakodanei/Hyper-FLUX.1-dev-8steps-lora-fp16/resolve/main/Hyper-FLUX.1-dev-8steps-lora-fp16.safetensors?download=true'
                        FLUX_DEV_LORA16 = 'https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-FLUX.1-dev-16steps-lora.safetensors?download=true'

                        DOWNLOADED_FLUX_DEV_LORA8 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Hyper-FLUX.1-dev-8steps-lora-fp16.safetensors')
                        DOWNLOADED_FLUX_DEV_FP16_LORA8 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Hyper-FLUX.1-dev-8steps-lora.safetensors')
                        DOWNLOADED_FLUX_DEV_LORA16 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Hyper-FLUX.1-dev-16steps-lora.safetensors')

                        utility.fileDownloader(DOWNLOADED_FLUX_DEV_LORA8, FLUX_DEV_LORA8)
                        utility.fileDownloader(DOWNLOADED_FLUX_DEV_FP16_LORA8, FLUX_DEV_FP16_LORA8)
                        utility.fileDownloader(DOWNLOADED_FLUX_DEV_LORA16, FLUX_DEV_LORA16)

                        allHyperFluxLoras = list(filter(lambda a: 'hyper-flux'.casefold() in a.casefold(), downloaded_filelist_filtered))
                        finalLoras = list(filter(lambda a: str(flux_hyper_lora_step) + 'step'.casefold() in a.casefold() and '-fp16'.casefold() not in a.casefold(), allHyperFluxLoras))
                        if flux_hyper_lora_type == 'FLUX.1-dev-fp16':
                            finalLoras_pre = list(filter(lambda a: str(flux_hyper_lora_step) + 'step'.casefold() in a.casefold() and '-fp16'.casefold() in a.casefold(), allHyperFluxLoras))
                            if len(finalLoras_pre) > 0:
                                finalLoras = finalLoras_pre

                        LORA_FILE = finalLoras[0]
                        FULL_LORA_PATH = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', LORA_FILE)

                        if FULL_LORA_PATH is not None and os.path.exists(FULL_LORA_PATH) == True:
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
                                    lora = comfy.utils.load_torch_file(FULL_LORA_PATH, safe_load=True)
                                    self.loaded_lora = (FULL_LORA_PATH, lora)

                                MODEL_DIFFUSION = comfy.sd.load_lora_for_models(MODEL_DIFFUSION, None, lora, flux_hyper_lora_strength, 0)[0]

                    if use_flux_turbo_lora == True:
                        FLUX_TURBO_LORA8 = 'https://huggingface.co/alimama-creative/FLUX.1-Turbo-Alpha/resolve/main/diffusion_pytorch_model.safetensors?download=true'
                        FLUX_TURBORENDER_LORA = 'https://huggingface.co/DarkMoonDragon/TurboRender-flux-dev/resolve/main/pytorch_lora_weights.safetensors?download=true'

                        DOWNLOADED_FLUX_TURBO_LORA8 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Turbo-FLUX.1-dev-8steps-lora.safetensors')
                        DOWNLOADED_FLUX_TURBORENDER_LORA = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Turbo-FLUX.1-dev-turborender-lora.safetensors')

                        utility.fileDownloader(DOWNLOADED_FLUX_TURBO_LORA8, FLUX_TURBO_LORA8)
                        utility.fileDownloader(DOWNLOADED_FLUX_TURBORENDER_LORA, FLUX_TURBORENDER_LORA)

                        allTurboTFluxLoras = list(filter(lambda a: 'turbo-flux'.casefold() in a.casefold(), downloaded_filelist_filtered))
                        LORA_FILE = None
                        if flux_turbo_lora_type == 'TurboAlpha' and 'Turbo-FLUX.1-dev-8steps-lora.safetensors' in allTurboTFluxLoras:
                            LORA_FILE = 'Turbo-FLUX.1-dev-8steps-lora.safetensors'
                        elif flux_turbo_lora_type == 'TurboRender' and 'Turbo-FLUX.1-dev-turborender-lora.safetensors' in allTurboTFluxLoras:
                            LORA_FILE = 'Turbo-FLUX.1-dev-turborender-lora.safetensors'
                        else:
                            finalTLoras_pre = list(filter(lambda a: str(flux_turbo_lora_step) + 'step'.casefold() in a.casefold(), allTurboTFluxLoras))
                            if len(finalTLoras_pre) > 0:
                                finalTLoras = finalTLoras_pre
                                LORA_FILE = finalTLoras[0]

                        if LORA_FILE is not None:
                            FULL_LORA_PATH = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', LORA_FILE)
                            if FULL_LORA_PATH is not None and os.path.exists(FULL_LORA_PATH) == True:
                                if flux_turbo_lora_strength != 0:
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

                                    MODEL_DIFFUSION = comfy.sd.load_lora_for_models(MODEL_DIFFUSION, None, lora, flux_turbo_lora_strength, 0)[0]

                    return (MODEL_DIFFUSION,) + (DUAL_CLIP,) + (FLUX_VAE,) + (MODEL_VERSION,)

            case 'Hyper':
                fullpathFile = folder_paths.get_full_path('checkpoints', ckpt_name)
                is_link = os.path.islink(str(fullpathFile))
                if is_link == True:
                    hypersd_selector = 'UNET'

                if hypersd_selector == 'UNET':
                    ModelConceptChanges = utility.ModelConceptNames(ckpt_name, model_concept, lightning_selector, lightning_model_step, hypersd_selector, hypersd_model_step, 'SDXL')
                    lora_name = ModelConceptChanges['lora_name']
                    unet_name = ModelConceptChanges['unet_name']
                    hyperModeValid = ModelConceptChanges['hyperModeValid']
                    OUTPUT_MODEL = utility.BDanceConceptHelper(self, model_concept, hyperModeValid, hypersd_selector, hypersd_model_step, None, lora_name, unet_name, None)
                    return (OUTPUT_MODEL[0],) + (OUTPUT_MODEL[1],) + (OUTPUT_MODEL[2],) + (MODEL_VERSION,)

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
        else:
            if sd3_gguf == False:
                if os.path.isfile(ModelConfigFullPath) and use_yaml == True:
                    ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
                    try:
                        LOADED_CHECKPOINT = comfy.sd.load_checkpoint(ModelConfigFullPath, ckpt_path, True, True, None, None, None)
                    except Exception:
                        LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, ckpt_name)
                else:
                    fullpathFile = folder_paths.get_full_path('checkpoints', ckpt_name)
                    is_link = os.path.islink(str(fullpathFile))
                    if is_link == False:
                        try:
                            LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, ckpt_name)
                        except Exception:
                            LOADED_CHECKPOINT = nodes.UNETLoader.load_unet(self, ckpt_name, 'default')
                        OUTPUT_MODEL = LOADED_CHECKPOINT[0]
                    else:
                        File_link = Path(str(fullpathFile)).resolve()
                        linkName_U = str(folder_paths.folder_names_and_paths["diffusion_models"][0][0])
                        linkName_D = str(folder_paths.folder_names_and_paths["diffusion_models"][0][1])
                        linkedFileName = str(File_link).replace(linkName_U + '\\', '').replace(linkName_D + '\\', '')
                        model_ext = os.path.splitext(linkedFileName)[1].lower()
                        if str(Path(linkName_U).stem) in linkedFileName:
                            linkedFileName = linkedFileName.split(Path(linkName_U).stem + '\\', 1)[1]
                        if str(Path(linkName_D).stem) in linkedFileName:
                            linkedFileName = linkedFileName.split(Path(linkName_D).stem + '\\', 1)[1]
                        if model_ext == '.gguf':
                            LOADED_CHECKPOINT = gguf_nodes.UnetLoaderGGUF.load_unet(self, linkedFileName)
                        else:
                            LOADED_CHECKPOINT = nodes.UNETLoader.load_unet(self, linkedFileName, 'default')
                        clip_list = folder_paths.get_filename_list("clip")
                        allLSDXLclip = list(filter(lambda a: 'clip_l'.casefold() in a.casefold(), clip_list))
                        OUTPUT_CLIP = nodes.CLIPLoader.load_clip(self, allLSDXLclip[0], 'stable_diffusion')[0]
                        OUTPUT_MODEL = LOADED_CHECKPOINT[0]
                        LOADED_CHECKPOINT.insert(1, OUTPUT_CLIP)

        if model_concept == 'SD3':
            if sd3_gguf != False:
                use_sd3_hyper_lora = False
                OUTPUT_MODEL = gguf_nodes.UnetLoaderGGUF.load_unet(self, sd3_gguf)[0]

            if len(LOADED_CHECKPOINT) < 2 or (len(LOADED_CHECKPOINT) >= 2 and type(LOADED_CHECKPOINT[1]).__name__ != 'CLIP') or clip_selection == False:
                OUTPUT_CLIP = nodes_sd3.TripleCLIPLoader.execute(sd3_clip_g, sd3_clip_l, sd3_clip_t5xxl)[0]
            else:
                OUTPUT_CLIP = LOADED_CHECKPOINT[1]

            if len(LOADED_CHECKPOINT) == 3 and type(LOADED_CHECKPOINT[2]).__name__ == 'VAE':
                OUTPUT_VAE = LOADED_CHECKPOINT[2]
            else:
                OUTPUT_VAE = utility.vae_loader_class.load_vae(sd3_unet_vae)[0]

            if use_sd3_hyper_lora == True:
                SD3_LORA4 = 'https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SD3-4steps-CFG-lora.safetensors?download=true'
                SD3_LORA8 = 'https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SD3-8steps-CFG-lora.safetensors?download=true'
                SD3_LORA16 = 'https://huggingface.co/ByteDance/Hyper-SD/resolve/main/Hyper-SD3-16steps-CFG-lora.safetensors?download=true'

                DOWNLOADED_SD3_LORA4 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Hyper-SD3-4steps-CFG-lora.safetensors')
                DOWNLOADED_SD3_LORA8 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Hyper-SD3-8steps-CFG-lora.safetensors')
                DOWNLOADED_SD3_LORA16 = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'Hyper-SD3-16steps-CFG-lora.safetensors')

                utility.fileDownloader(DOWNLOADED_SD3_LORA4, SD3_LORA4)
                utility.fileDownloader(DOWNLOADED_SD3_LORA8, SD3_LORA8)
                utility.fileDownloader(DOWNLOADED_SD3_LORA16, SD3_LORA16)
                downloaded_filelist_filtered = utility.getDownloadedFiles()
                allHyperSD3Loras = list(filter(lambda a: 'hyper-sd3'.casefold() in a.casefold(), downloaded_filelist_filtered))
                finalLoras = list(filter(lambda a: str(sd3_hyper_lora_step) + 'step'.casefold() in a.casefold(), allHyperSD3Loras))
                LORA_FILE = finalLoras[0]
                FULL_LORA_PATH = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', LORA_FILE)

                if FULL_LORA_PATH is not None and os.path.exists(FULL_LORA_PATH) == True:
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

            return (OUTPUT_MODEL,) + (OUTPUT_CLIP,) + (OUTPUT_VAE,) + (MODEL_VERSION,)
        else:
            OUTPUT_CLIP = LOADED_CHECKPOINT[1]

        if model_concept == 'Lightning' and lightning_selector == 'LORA' and MODEL_VERSION != 'SDXL':
            model_concept = MODEL_VERSION

        match model_concept:
            case 'Hyper' | 'Lightning':
                if ckpt_name is not None and model_concept == 'Lightning' and lightning_selector == 'CUSTOM':
                    is_force_lora = re.findall(r"(FoLo)", ckpt_name)
                    if len(is_force_lora) > 0:
                        lightning_selector = 'LORA'

                HYPER_LIGHTNING_ORIGINAL_VERSION = utility.getModelType(ckpt_name, 'checkpoints')
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

                ModelConceptChanges = utility.ModelConceptNames(ckpt_name, model_concept, lightning_selector, lightning_model_step, hypersd_selector, hypersd_model_step, HYPER_LIGHTNING_ORIGINAL_VERSION)
                ckpt_name = ModelConceptChanges['ckpt_name']
                if ModelConceptChanges['lora_name'] is not None:
                    lora_name = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', ModelConceptChanges['lora_name'])
                else:
                    lora_name = None
                unet_name = ModelConceptChanges['unet_name']
                lightningModeValid = ModelConceptChanges['lightningModeValid']
                hyperModeValid = ModelConceptChanges['hyperModeValid']

                if model_concept == MODEL_VERSION and model_concept == 'Lightning':
                    if lora_name is None and unet_name is None and lightningModeValid == False and ckpt_name is not None and loaded_model is None:
                        OUTPUT_MODEL = utility.BDanceConceptHelper(self, model_concept, True, 'SAFETENSOR', lightning_model_step, OUTPUT_MODEL, lora_name, unet_name, ckpt_name, strength_lightning_lora_model)

                force_lora_weighth = re.findall(r"(?i)(\d+)LoWe", ckpt_name)
                if len(force_lora_weighth) > 0:
                    if len(str(force_lora_weighth[0])) > 1:
                        FirstWCH = str(force_lora_weighth[0])[:1]
                        OtherWCH = str(force_lora_weighth[0])[1:]
                        strength_lora_float = float(FirstWCH + '.' + OtherWCH)
                        strength_lightning_lora_model = strength_lora_float
                        strength_hypersd_lora_model = strength_lora_float
                    else:
                        strength_lightning_lora_model = int(force_lora_weighth[0])
                        strength_hypersd_lora_model = int(force_lora_weighth[0])

                if lightningModeValid == True and loaded_model is None:
                    OUTPUT_MODEL = utility.BDanceConceptHelper(self, model_concept, lightningModeValid, lightning_selector, lightning_model_step, OUTPUT_MODEL, lora_name, unet_name, ckpt_name, strength_lightning_lora_model)

                if hyperModeValid == True and loaded_model is None:
                    OUTPUT_MODEL = utility.BDanceConceptHelper(self, model_concept, hyperModeValid, hypersd_selector, hypersd_model_step, OUTPUT_MODEL, lora_name, unet_name, ckpt_name, strength_hypersd_lora_model)
                    vae_selection = True

            case 'LCM':
                vae_selection = True
                if MODEL_VERSION == 'SD1' or MODEL_VERSION == 'SDXL':
                    SDXL_LORA = 'https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors?download=true'
                    SD_LORA = 'https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors?download=true'

                    DOWNLOADED_SD_LORA = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'lcm_lora_sd.safetensors')
                    DOWNLOADED_SDXL_LORA = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'lcm_lora_sdxl.safetensors')

                    utility.fileDownloader(DOWNLOADED_SD_LORA, SD_LORA)
                    utility.fileDownloader(DOWNLOADED_SDXL_LORA, SDXL_LORA)

                    LORA_PATH = None
                    if MODEL_VERSION == 'SDXL':
                        LORA_PATH = DOWNLOADED_SDXL_LORA
                    elif MODEL_VERSION == 'SD1':
                        LORA_PATH = DOWNLOADED_SD_LORA

                    if LORA_PATH is not None and os.path.exists(LORA_PATH) == True:
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
                                lora = comfy.utils.load_torch_file(LORA_PATH, safe_load=True)
                                self.loaded_lora = (LORA_PATH, lora)

                            MODEL_LORA = comfy.sd.load_lora_for_models(OUTPUT_MODEL, None, lora, strength_lcm_lora_model, 0)[0]
                            OUTPUT_MODEL = lcm(self, MODEL_LORA, False)

            case 'Playground':
                OUTPUT_MODEL = nodes_model_advanced.ModelSamplingContinuousEDM.patch(self, OUTPUT_MODEL, 'edm_playground_v2.5', playground_sigma_max, playground_sigma_min)[0]

        if model_concept == 'SD2':
            vae_selection = True

        if len(LOADED_CHECKPOINT) < 3 or (len(LOADED_CHECKPOINT) == 3 and type(LOADED_CHECKPOINT[2]).__name__ != 'VAE') or vae_selection == False:
            if vae_name != "Baked":
                OUTPUT_VAE = utility.vae_loader_class.load_vae(vae_name)[0]
            else:
                vae_list = folder_paths.get_filename_list("vae")
                if MODEL_VERSION == 'SD1':
                    allLSD1vae = list(filter(lambda a: 'sdxl'.casefold() not in a.casefold() and 'flux'.casefold() not in a.casefold() and 'hunyuan'.casefold() not in a.casefold() and 'stage'.casefold() not in a.casefold(), vae_list))
                    OUTPUT_VAE = utility.vae_loader_class.load_vae(allLSD1vae[0])[0]
                else:
                    allLSDXLvae = list(filter(lambda a: 'sdxl_'.casefold() in a.casefold(), vae_list))
                    OUTPUT_VAE = utility.vae_loader_class.load_vae(allLSDXLvae[0])[0]
        else:
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
                "token_normalization": (["none", "mean", "length", "length+mean"], {"default": "mean"}),
                "weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"], {"default": "comfy++"}),
            },
            "optional": {
                # "clip_raw": ("CLIP", {"forceInput": True}),
                "enhanced_prompt": ("STRING", {"forceInput": True}),
                "enhanced_prompt_usage": (['None', 'Add', 'Replace', 'T5-XXL'], {"default": "T5-XXL"}),
                "enhanced_prompt_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "model_concept": ("STRING", {"default": "SD1", "forceInput": True}),
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
                # "copy_prompt_to_l": ("BOOLEAN", {"default": True}),
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

    def clip_encode(self, clip, clip_mode, last_layer, negative_strength, int_style_pos_strength, int_style_neg_strength, opt_pos_strength, opt_neg_strength, style_pos_strength, style_neg_strength, style_handling, style_swap, enhanced_prompt_strength, int_style_pos, int_style_neg, adv_encode, token_normalization, weight_interpretation, l_strength, extra_pnginfo, prompt, copy_prompt_to_l=True, width=1024, height=1024, positive_prompt="", negative_prompt="", enhanced_prompt="", enhanced_prompt_usage="T5-XXL", clip_model='Default', longclip_model='Default', model_keywords=None, lora_keywords=None, lycoris_keywords=None, embedding_pos=None, embedding_neg=None, opt_pos_prompt="", opt_neg_prompt="", style_position=False, style_neg_prompt="", style_pos_prompt="", positive_l="", negative_l="", use_int_style=False, model_version="SDXL", model_concept="Normal", edit_image_list=None, edit_vae=None, workflow_tuple=None):
        copy_prompt_to_l = True

        clip_mode_default = ['PixartSigma', 'StableCascade', 'Hunyuan', 'SD3', 'Hyper', 'Pony', 'AuraFlow']
        if model_concept in clip_mode_default:
            clip_mode = True
            clip_model = 'Default'
            longclip_model = 'Default'

        model_version_default = ['Hunyuan', 'KwaiKolors', 'SD3', 'Playground', 'StableCascade', 'Turbo', 'Flux', "Z-Image", 'Lightning', 'Illustrious', 'QwenGen', 'QwenEdit']
        if model_concept in model_version_default:
            model_version = 'SDXL'

        advanced_default = ['StableCascade', 'KwaiKolors', 'Flux', "Z-Image", 'Pony', 'SD1', 'SD2', 'SD3', 'Lightning', 'Hunyuan', 'QwenGen', 'QwenEdit', 'AuraFlow']
        if model_concept in advanced_default:
            adv_encode = False

        if model_concept == 'Flux':
            last_layer = 0
            clip_model = 'Default'
            # longclip_model = 'Default'

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
                    l_strength = workflow_tuple['prompt_encoder']['l_strength']
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
                    if 'enhanced_prompt_usage' in workflow_tuple['prompt_encoder']:
                        enhanced_prompt_usage = workflow_tuple['prompt_encoder']['enhanced_prompt_usage']
                    if 'enhanced_prompt_strength' in workflow_tuple['prompt_encoder']:
                        enhanced_prompt_strength = workflow_tuple['prompt_encoder']['enhanced_prompt_strength']
                    if 'style_handling' in workflow_tuple['prompt_encoder']:
                        style_handling = workflow_tuple['prompt_encoder']['style_handling']
                    if 'style_swap' in workflow_tuple['prompt_encoder']:
                        style_swap = workflow_tuple['prompt_encoder']['style_swap']
                if workflow_tuple['setup_states']['clip_optional_prompts'] == True:
                    opt_pos_prompt = workflow_tuple['prompt_encoder']['opt_pos_prompt']
                    opt_pos_strength = workflow_tuple['prompt_encoder']['opt_pos_strength']
                    opt_neg_prompt = workflow_tuple['prompt_encoder']['opt_neg_prompt']
                    opt_neg_strength = workflow_tuple['prompt_encoder']['opt_neg_strength']
                if workflow_tuple['setup_states']['enhanced_prompt_usage'] != 'None':
                    enhanced_prompt = workflow_tuple['prompt_encoder']['enhanced_prompt']
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

        is_sdxl = 0
        match model_version:
            case 'SDXL' | 'AuraFlow' | 'Z-Image':
                is_sdxl = 1

        t5xxl_prompt = ""
        if len(enhanced_prompt) > 5:
            match enhanced_prompt_usage:  # 'None', 'Add', 'Replace', 'T5-XXL'
                case 'Add':
                    if enhanced_prompt_strength != 1:
                        enhanced_prompt = f'({enhanced_prompt}:{enhanced_prompt_strength:.2f})'
                    if enhanced_prompt_strength != 0:
                        positive_prompt = positive_prompt + ', ' + enhanced_prompt
                case 'Replace':
                    positive_prompt = enhanced_prompt
                case 'T5-XXL':
                    t5xxl_prompt = enhanced_prompt
        else:
            if len(style_pos_prompt) > 5 and style_handling == True:
                if style_swap == True:
                    positive_prompt_tmp = positive_prompt
                    positive_prompt = style_pos_prompt
                    style_pos_prompt = positive_prompt_tmp

                enhanced_prompt = style_pos_prompt
                t5xxl_prompt = enhanced_prompt
                style_pos_prompt = None
                positive_l = style_pos_prompt
                copy_prompt_to_l = False

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
            positive_l = positive_prompt
            negative_l = negative_prompt

        if l_strength != 1:
            positive_l = f'({positive_l}:{l_strength:.2f})'.replace(":1.00", "") if positive_l is not None and positive_l.strip(' ,;') != '' else ''
            negative_l = f'({negative_l}:{l_strength:.2f})'.replace(":1.00", "") if negative_l is not None and negative_l.strip(' ,;') != '' else ''
        else:
            positive_l = f'{positive_l}'.replace(":1.00", "") if positive_l is not None and positive_l.strip(' ,;') != '' else ''
            negative_l = f'{negative_l}'.replace(":1.00", "") if negative_l is not None and negative_l.strip(' ,;') != '' else ''

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

        WORKFLOWDATA = extra_pnginfo['workflow']['nodes']
        CONCEPT_SELECTOR = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereModelConceptSelector', 'model_concept', prompt)
        if CONCEPT_SELECTOR == 'Flux' and (model_concept == 'Flux' and model_concept is not None):
            adv_encode = False
            clip_model = 'Default'
            # clip_mode = True
            last_layer = 0

        if 'SANA' in model_concept:
            device = model_management.get_torch_device()

            if 'scheduler_name' in workflow_tuple:
                sana_scheduler_name = workflow_tuple['scheduler_name']
            else:
                sana_scheduler_name = 'flow_dpm-solver'

            preset_te_prompt = ['Create one detailed perfect prompt from given User Prompt for stable diffusion text-to-image text2image modern DiT models.', 'Generate only the one enhanced description for the prompt below, avoid including any additional questions comments or evaluations.', 'User Prompt: ']
            chi_prompt = "\n".join(preset_te_prompt)

            if sana_scheduler_name == 'flow_dpm-solver' and hasattr(clip, 'text_encoder'):
                base_ratios = eval(f"ASPECT_RATIO_{1024}_TEST")
                clip.text_encoder.to(device)

                null_caption_token = clip.tokenizer(
                    negative_text,
                    max_length=300,
                    padding="max_length",
                    truncation=True,
                    return_tensors="pt",
                ).to(device)
                null_caption_embs = clip.text_encoder(null_caption_token.input_ids, null_caption_token.attention_mask)[0]
                prompts = []
                with torch.no_grad():
                    prompts.append(prepare_prompt_ar(positive_text, base_ratios, device=device, show=False)[0].strip())
                    prompts_all = [chi_prompt + positive_text]
                    num_chi_prompt_tokens = len(clip.tokenizer.encode(chi_prompt))
                    max_length_all = (num_chi_prompt_tokens + 300 - 2)  # magic number 2: [bos], [_]
                    caption_token = clip.tokenizer(
                        prompts_all,
                        max_length=max_length_all,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt",
                    ).to(device)
                    select_index = [0] + list(range(-300 + 1, 0))
                    caption_embs = clip.text_encoder(caption_token.input_ids, caption_token.attention_mask)[0][:, None][:, :, select_index]
                    emb_masks = caption_token.attention_mask[:, select_index]
                    null_y = null_caption_embs.repeat(len(prompts), 1, 1)[:, None]

                clip.text_encoder.to(model_management.text_encoder_offload_device())
                comfy.model_management.soft_empty_cache(True)

                return ([[caption_embs, {"emb_masks": emb_masks}]], [[null_y, {}]], positive_text, negative_text, t5xxl_prompt, "", "", workflow_tuple)
            else:
                tokenizer = clip["tokenizer"]
                text_encoder = clip["text_encoder"]
                with torch.no_grad():
                    full_prompt = chi_prompt + positive_text
                    num_chi_tokens = len(tokenizer.encode(chi_prompt))
                    max_length = num_chi_tokens + 300 - 2

                    tokens = tokenizer(
                        [full_prompt],
                        max_length=max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    ).to(text_encoder.device)
                    select_idx = [0] + list(range(-300 + 1, 0))
                    embs_plus = text_encoder(tokens.input_ids, tokens.attention_mask)[0][:, None][:, :, select_idx]
                    emb_plus_masks = tokens.attention_mask[:, select_idx]
                sana_embs_pos = embs_plus * emb_plus_masks.unsqueeze(-1)

                with torch.no_grad():
                    full_prompt = chi_prompt + negative_text
                    num_chi_tokens = len(tokenizer.encode(chi_prompt))
                    max_length = num_chi_tokens + 300 - 2

                    tokens = tokenizer(
                        [full_prompt],
                        max_length=max_length,
                        padding="max_length",
                        truncation=True,
                        return_tensors="pt"
                    ).to(text_encoder.device)
                    select_idx = [0] + list(range(-300 + 1, 0))
                    embs_minus = text_encoder(tokens.input_ids, tokens.attention_mask)[0][:, None][:, :, select_idx]
                    emb_minus_masks = tokens.attention_mask[:, select_idx]
                sana_embs_neg = embs_minus * emb_minus_masks.unsqueeze(-1)

                return ([[sana_embs_pos, {}]], [[sana_embs_neg, {}]], positive_text, negative_text, t5xxl_prompt, "", "", workflow_tuple)

        if model_concept == 'QwenEdit' and (type(edit_image_list).__name__ == "list" or type(edit_image_list).__name__ == "Tensor" and edit_image_list is not None):
            if type(edit_image_list).__name__ == "Tensor":
                edit_image_list = [edit_image_list]

            positive_text = utility.DiT_cleaner(positive_text)
            negative_text = utility.DiT_cleaner(negative_text)

            conditioning = utility.edit_encoder(clip, positive_text, edit_vae, edit_image_list)

            tokens_neg = clip.tokenize(negative_text, images=[])
            conditioning_neg = clip.encode_from_tokens_scheduled(tokens_neg)

            return (conditioning, conditioning_neg, positive_text, negative_text, t5xxl_prompt, "", "", workflow_tuple)

        if model_concept == 'PixartSigma':
            cond_pos_ref = None
            out_pos_ref = None
            cond_neg_ref = None
            out_neg_ref = None

            positive_text = utility.DiT_cleaner(positive_text)
            negative_text = utility.DiT_cleaner(negative_text)

            if clip['refiner'] is not None:
                clipRef = clip['refiner']
                tokens_pos_ref = clipRef.tokenize(positive_text)
                tokens_neg_ref = clipRef.tokenize(negative_text)
                out_pos_ref = clipRef.encode_from_tokens(tokens_pos_ref, return_pooled=True, return_dict=True)
                out_neg_ref = clipRef.encode_from_tokens(tokens_neg_ref, return_pooled=True, return_dict=True)
                cond_pos_ref = out_pos_ref.pop("cond")
                cond_neg_ref = out_neg_ref.pop("cond")

            clipMain = clip['main']
            tokens_pos_main = clipMain.tokenize(positive_text)
            tokens_neg_main = clipMain.tokenize(negative_text)
            out_pos_main = clipMain.encode_from_tokens(tokens_pos_main, return_pooled=True, return_dict=True)
            out_neg_main = clipMain.encode_from_tokens(tokens_neg_main, return_pooled=True, return_dict=True)
            cond_pos_main = out_pos_main.pop("cond")
            cond_neg_main = out_neg_main.pop("cond")

            return ({'refiner': [[cond_pos_ref, out_pos_ref]], 'main': [[cond_pos_main, out_pos_main]]}, {'refiner': [[cond_neg_ref, out_neg_ref]], 'main': [[cond_neg_main, out_neg_main]]}, positive_text, negative_text, t5xxl_prompt, "", "", workflow_tuple)

        if model_concept == 'Hunyuan':
            last_layer = 0
            if clip['t5'] is not None:
                CLIPDIT = clip['clip']
                CLIPT5 = clip['t5']

                positive_text = utility.DiT_cleaner(positive_text, 0)
                negative_text = utility.DiT_cleaner(negative_text, 0)
                t5xxl_prompt = utility.DiT_cleaner(t5xxl_prompt, 0)

                pos_out = clipping.HunyuanClipping(self, positive_text, t5xxl_prompt, CLIPDIT, CLIPT5)
                neg_out = clipping.HunyuanClipping(self, negative_text, "", CLIPDIT, CLIPT5)
                return (pos_out[0], neg_out[0], positive_text, negative_text, t5xxl_prompt, "", "", workflow_tuple)
            else:
                clip = clip['clip']
                positive_text = utility.DiT_cleaner(positive_text, 512)
                negative_text = utility.DiT_cleaner(negative_text, 512)

        if model_concept == 'KwaiKolors':
            positive_text = utility.DiT_cleaner(positive_text)
            negative_text = utility.DiT_cleaner(negative_text)

            # device = model_management.get_torch_device()
            # device = "cuda" if torch.cuda.is_available() else "cpu"
            device = model_management.text_encoder_device()
            # offload_device = model_management.unet_offload_device()
            # offload_device = model_management.text_encoder_offload_device()
            try:
                model_management.unload_all_models()
                model_management.soft_empty_cache()
            except Exception:
                print('Cannot clear cache...')
            tokenizer = clip['tokenizer']
            text_encoder = clip['text_encoder']
            model_management.soft_empty_cache()

            prompt_embeds_dtype = torch.float16
            if text_encoder is not None:
                prompt_embeds_dtype = text_encoder.dtype
            try:
                text_encoder.to(dtype = prompt_embeds_dtype, device=device) # todo: python error!
            except Exception:
                print('Device init error...')
            text_inputs = tokenizer(positive_text, padding="max_length", max_length=256, truncation=True, return_tensors="pt", ).to(device)
            output = text_encoder(input_ids=text_inputs['input_ids'], attention_mask=text_inputs['attention_mask'], position_ids=text_inputs['position_ids'], output_hidden_states=True)
            prompt_embeds = output.hidden_states[-2].permute(1, 0, 2).clone()
            text_proj = output.hidden_states[-1][-1, :, :].clone()
            bs_embed, seq_len, _ = prompt_embeds.shape

            num_images_per_prompt = 1
            batch_size = 1
            prompt_embeds = prompt_embeds.repeat(1, num_images_per_prompt, 1)
            prompt_embeds = prompt_embeds.view(bs_embed * num_images_per_prompt, seq_len, -1)

            uncond_tokens = [negative_text]
            max_length = prompt_embeds.shape[1]
            uncond_input = tokenizer(uncond_tokens, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt", ).to(device)
            output = text_encoder(input_ids=uncond_input['input_ids'], attention_mask=uncond_input['attention_mask'], position_ids=uncond_input['position_ids'], output_hidden_states=True)
            negative_prompt_embeds = output.hidden_states[-2].permute(1, 0, 2).clone()
            negative_text_proj = output.hidden_states[-1][-1, :, :].clone()
            seq_len = negative_prompt_embeds.shape[1]
            negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoder.dtype, device=device)
            negative_prompt_embeds = negative_prompt_embeds.repeat(1, num_images_per_prompt, 1)
            negative_prompt_embeds = negative_prompt_embeds.view(batch_size * num_images_per_prompt, seq_len, -1)

            bs_embed = text_proj.shape[0]
            text_proj = text_proj.repeat(1, num_images_per_prompt).view(bs_embed * num_images_per_prompt, -1)
            negative_text_proj = negative_text_proj.repeat(1, num_images_per_prompt).view(bs_embed * num_images_per_prompt, -1)
            # text_encoder.to(offload_device)
            text_encoder.to(device)
            try:
                model_management.soft_empty_cache()
            except Exception:
                print('Cannot clear cache...')
            gc.collect()
            kolors_embeds = {
                'prompt_embeds': prompt_embeds.half(),
                'negative_prompt_embeds': negative_prompt_embeds.half(),
                'pooled_prompt_embeds': text_proj.half(),
                'negative_pooled_prompt_embeds': negative_text_proj.half()
            }

            # kolors_embeds = nodes_kwai.KolorsTextEncode.encode(self, clip, positive_text, negative_text, 1)[0]
            return (kolors_embeds, None, positive_text, negative_text, t5xxl_prompt, "", "", workflow_tuple)

        if model_concept == 'SD3':
            if len(enhanced_prompt) > 1 and enhanced_prompt_usage == 'T5-XXL' and len(t5xxl_prompt) > 5:
                pos_out = nodes_sd3.CLIPTextEncodeSD3.execute(clip, positive_text, positive_text, t5xxl_prompt, 'none')

                tokens_neg = clip.tokenize(negative_text)
                out_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True, return_dict=True)
                cond_neg = out_neg.pop("cond")

                return (pos_out[0], [[cond_neg, out_neg]], positive_text, negative_text, t5xxl_prompt, "", "", workflow_tuple)

        if clip_mode == False:
            if longclip_model == 'Default':
                longclip_model = 'longclip-L.pt'

            LONGCLIPL_PATH = os.path.join(folder_paths.models_dir, 'clip')
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
            workflow_tuple['prompt_encoder']['enhanced_prompt'] = enhanced_prompt
            workflow_tuple['prompt_encoder']['enhanced_prompt_usage'] = enhanced_prompt_usage
            workflow_tuple['prompt_encoder']['enhanced_prompt_strength'] = enhanced_prompt_strength
            workflow_tuple['prompt_encoder']['style_pos_strength'] = style_pos_strength
            workflow_tuple['prompt_encoder']['style_neg_prompt'] = style_neg_prompt
            workflow_tuple['prompt_encoder']['style_neg_strength'] = style_neg_strength

            if copy_prompt_to_l == False:
                workflow_tuple['prompt_encoder']['positive_l'] = positive_l
                workflow_tuple['prompt_encoder']['negative_l'] = negative_l
            workflow_tuple['prompt_encoder']['copy_prompt_to_l'] = copy_prompt_to_l
            workflow_tuple['prompt_encoder']['l_strength'] = l_strength

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

        if adv_encode == True:
            tokens_p = clip.tokenize(positive_text)
            tokens_n = clip.tokenize(negative_text)
            if is_sdxl == 0 or 'l' not in tokens_p or 'g' not in tokens_p or 'l' not in tokens_n or 'g' not in tokens_n:
                embeddings_final_pos, pooled_pos = advanced_encode(clip, positive_text, token_normalization, weight_interpretation, w_max=1.0, apply_to_pooled=True)
                embeddings_final_neg, pooled_neg = advanced_encode(clip, negative_text, token_normalization, weight_interpretation, w_max=1.0, apply_to_pooled=True)
                return ([[embeddings_final_pos, {"pooled_output": pooled_pos}]], [[embeddings_final_neg, {"pooled_output": pooled_neg}]], positive_text, negative_text, t5xxl_prompt, "", "", workflow_tuple)
            else:
                # tokens_p = clip.tokenize(positive_text)
                if 'l' in clip.tokenize(positive_l):
                    tokens_p["l"] = clip.tokenize(positive_l)["l"]
                    if len(tokens_p["l"]) != len(tokens_p["g"]):
                        empty = clip.tokenize("")
                        while len(tokens_p["l"]) < len(tokens_p["g"]):
                            tokens_p["l"] += empty["l"]
                        while len(tokens_p["l"]) > len(tokens_p["g"]):
                            tokens_p["g"] += empty["g"]

                # tokens_n = clip.tokenize(negative_text)
                if 'l' in clip.tokenize(negative_l):
                    tokens_n["l"] = clip.tokenize(negative_l)["l"]

                    if len(tokens_n["l"]) != len(tokens_n["g"]):
                        empty = clip.tokenize("")
                        while len(tokens_n["l"]) < len(tokens_n["g"]):
                            tokens_n["l"] += empty["l"]
                        while len(tokens_n["l"]) > len(tokens_n["g"]):
                            tokens_n["g"] += empty["g"]

                cond_p, pooled_p = clip.encode_from_tokens(tokens_p, return_pooled=True)
                cond_n, pooled_n = clip.encode_from_tokens(tokens_n, return_pooled=True)

                return ([[cond_p, {"pooled_output": pooled_p, "width": width, "height": height, "crop_w": 0, "crop_h": 0, "target_width": width, "target_height": height}]], [[cond_n, {"pooled_output": pooled_n, "width": width, "height": height, "crop_w": 0, "crop_h": 0, "target_width": width, "target_height": height}]], positive_text, negative_text, positive_l, negative_l, workflow_tuple)

        else:
            if clip_mode == True:
                if model_concept == 'Flux':
                    WORKFLOWDATA = extra_pnginfo['workflow']['nodes']
                    FLUX_SAMPLER = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereModelConceptSelector', 'flux_sampler', prompt)
                    FLUX_GUIDANCE = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereModelConceptSelector', 'flux_clip_guidance', prompt)
                    if FLUX_GUIDANCE is None:
                        FLUX_GUIDANCE = 2
                    if FLUX_SAMPLER == 'ksampler':
                        CONDITIONING_POS = nodes_flux.CLIPTextEncodeFlux.execute(clip, positive_text, t5xxl_prompt, FLUX_GUIDANCE)[0]
                        if workflow_tuple is not None and 'cfg' in workflow_tuple and int(workflow_tuple['cfg']) < 1.2:
                            CONDITIONING_NEG = CONDITIONING_POS
                        else:
                            CONDITIONING_NEG = nodes_flux.CLIPTextEncodeFlux.execute(clip, negative_text, negative_text, FLUX_GUIDANCE)[0]
                        return (CONDITIONING_POS, CONDITIONING_NEG, positive_text, negative_text, t5xxl_prompt, "", "", workflow_tuple)
                    else:
                        if (enhanced_prompt_usage == 'T5-XXL' or style_handling == True) and len(t5xxl_prompt) > 5:
                            CONDITIONING_POS = nodes_flux.CLIPTextEncodeFlux.execute(clip, positive_text, t5xxl_prompt, FLUX_GUIDANCE)[0]
                            return (CONDITIONING_POS, CONDITIONING_POS, positive_text, negative_text, t5xxl_prompt, "", "", workflow_tuple)

            tokens_pos = clip.tokenize(positive_text)
            tokens_neg = clip.tokenize(negative_text)

            if model_concept == 'StableCascade':
                positive_text = utility.DiT_cleaner(positive_text)
                negative_text = utility.DiT_cleaner(negative_text)

                cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
                cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
                return ([[cond_pos, {"pooled_output": pooled_pos}]], [[cond_neg, {"pooled_output": pooled_neg}]], positive_text, negative_text, t5xxl_prompt, "", "", workflow_tuple)

            try:
                comfy.model_management.soft_empty_cache()
            except Exception:
                print('No need to clear cache...')

            out_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True, return_dict=True)
            out_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True, return_dict=True)

            cond_pos = out_pos.pop("cond")
            cond_neg = out_neg.pop("cond")

            return ([[cond_pos, out_pos]], [[cond_neg, out_neg]], positive_text, negative_text, t5xxl_prompt, "", "", workflow_tuple)

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