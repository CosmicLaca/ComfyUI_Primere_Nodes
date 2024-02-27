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
from .modules.adv_encode import advanced_encode, advanced_encode_XL
from nodes import MAX_RESOLUTION
from ..components import utility
from pathlib import Path
import re
import requests
from ..components import hypernetwork
import comfy.sd
import comfy.utils
from ..utils import comfy_dir
import comfy_extras.nodes_model_advanced as nodes_model_advanced
import comfy_extras.nodes_upscale_model as nodes_upscale_model

class PrimereSamplers:
    CATEGORY = TREE_DASHBOARD
    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS)
    RETURN_NAMES = ("SAMPLER_NAME", "SCHEDULER_NAME")
    FUNCTION = "get_sampler"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler_name": (comfy.samplers.KSampler.SCHEDULERS,)
            }
        }

    def get_sampler(self, sampler_name, scheduler_name):
        return sampler_name, scheduler_name

class PrimereSamplersSteps:
    CATEGORY = TREE_DASHBOARD
    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT", "FLOAT")
    RETURN_NAMES = ("SAMPLER_NAME", "SCHEDULER_NAME", "STEPS", "CFG")
    FUNCTION = "get_sampler_step"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                "scheduler_name": (comfy.samplers.KSampler.SCHEDULERS,),
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
    RETURN_TYPES = ("CHECKPOINT_NAME", "STRING",)
    RETURN_NAMES = ("MODEL_NAME", "MODEL_VERSION",)
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
        modelname_only = Path(base_model).stem
        model_version = utility.get_value_from_cache('model_version', modelname_only)
        if model_version is None:
            LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, base_model, output_vae=True, output_clip=True)
            model_version = utility.getCheckpointVersion(LOADED_CHECKPOINT[0])
            utility.add_value_to_cache('model_version', modelname_only, model_version)

        return (base_model, model_version)

class PrimereVAELoader:
    RETURN_TYPES = ("VAE",)
    RETURN_NAMES = ("VAE",)
    FUNCTION = "load_primere_vae"
    CATEGORY = TREE_DASHBOARD

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae_name": ("VAE_NAME",)
            },
        }

    def load_primere_vae(self, vae_name, ):
        return nodes.VAELoader.load_vae(self, vae_name)

class PrimereLCMSelector:
    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT", "FLOAT", "STRING")
    RETURN_NAMES = ("SAMPLER_NAME", "SCHEDULER_NAME", "STEPS", "CFG", "MODEL_CONCEPT")
    FUNCTION = "select_lcm_mode"
    CATEGORY = TREE_DASHBOARD

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "use_lcm": ("BOOLEAN", {"default": False}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"forceInput": True, "default": "euler"}),
                "scheduler_name": (comfy.samplers.KSampler.SCHEDULERS, {"forceInput": True, "default": "normal"}),
                "lcm_sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"forceInput": True, "default": "lcm"}),
                "lcm_scheduler_name": (comfy.samplers.KSampler.SCHEDULERS, {"forceInput": True, "default": "sgm_uniform"}),
                "cfg_scale": ('FLOAT', {"forceInput": True, "default": 7}),
                "steps": ('INT', {"forceInput": True, "default": 12}),
                "lcm_cfg_scale": ('FLOAT', {"forceInput": True, "default": 1.2}),
                "lcm_steps": ('INT', {"forceInput": True, "default": 6}),
            },
        }

    def select_lcm_mode(self, use_lcm = False, sampler_name = 'euler', scheduler_name = 'normal', lcm_sampler_name = 'lcm', lcm_scheduler_name = 'sgm_uniform', cfg_scale = 7, steps = 12, lcm_cfg_scale = 1.2, lcm_steps = 6):
        model_concept = "Normal"
        if use_lcm == True:
            sampler_name = lcm_sampler_name
            scheduler_name = lcm_scheduler_name
            steps = lcm_steps
            cfg_scale = lcm_cfg_scale
            model_concept = "LCM"

        return (sampler_name, scheduler_name, steps, cfg_scale, model_concept,)

class PrimereModelConceptSelector:
    RETURN_TYPES = (comfy.samplers.KSampler.SAMPLERS, comfy.samplers.KSampler.SCHEDULERS, "INT", "FLOAT", "STRING", "STRING", "INT", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("SAMPLER_NAME", "SCHEDULER_NAME", "STEPS", "CFG", "MODEL_CONCEPT", "LIGHTNING_SELECTOR", "LIGHTNING_MODEL_STEP", "CASCADE_STAGE_A", "CASCADE_STAGE_B", "CASCADE_STAGE_C", "CASCADE_CLIP")
    FUNCTION = "select_model_concept"
    CATEGORY = TREE_DASHBOARD

    UNETLIST = folder_paths.get_filename_list("unet")
    VAELIST = folder_paths.get_filename_list("vae")
    CLIPLIST = folder_paths.get_filename_list("clip")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "normal_sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"forceInput": True, "default": "euler"}),
                "normal_scheduler_name": (comfy.samplers.KSampler.SCHEDULERS, {"forceInput": True, "default": "normal"}),
                "normal_cfg_scale": ('FLOAT', {"forceInput": True, "default": 7}),
                "normal_steps": ('INT', {"forceInput": True, "default": 12}),

                "model_concept": (["Normal", "LCM", "Turbo", "Cascade", "Lightning"], {"default": "Normal"}),
                "lightning_selector": (["UNET", "LORA", "SAFETENSOR", "CUSTOM"], {"default": "SAFETENSOR"}),
                "lightning_model_step": ([1, 2, 4, 8], {"default": 8}),
                "lightning_sampler": (
                "BOOLEAN", {"default": False, "label_on": "Set by model", "label_off": "Custom (external)"}),

                "cascade_stage_a": (cls.VAELIST,),
                "cascade_stage_b": (cls.UNETLIST,),
                "cascade_stage_c": (cls.UNETLIST,),
                "cascade_clip": (cls.CLIPLIST,),
            },
            "optional": {
                "lcm_sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"forceInput": True, "default": "lcm"}),
                "lcm_scheduler_name": (comfy.samplers.KSampler.SCHEDULERS, {"forceInput": True, "default": "sgm_uniform"}),
                "lcm_cfg_scale": ('FLOAT', {"forceInput": True, "default": 1.2}),
                "lcm_steps": ('INT', {"forceInput": True, "default": 6}),

                "turbo_sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"forceInput": True, "default": "dpmpp_sde"}),
                "turbo_scheduler_name": (comfy.samplers.KSampler.SCHEDULERS, {"forceInput": True, "default": "normal"}),
                "turbo_cfg_scale": ('FLOAT', {"forceInput": True, "default": 1.15}),
                "turbo_steps": ('INT', {"forceInput": True, "default": 2}),

                "cascade_sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"forceInput": True, "default": "euler_ancestral"}),
                "cascade_scheduler_name": (comfy.samplers.KSampler.SCHEDULERS, {"forceInput": True, "default": "simple"}),
                "cascade_cfg_scale": ('FLOAT', {"forceInput": True, "default": 4}),
                "cascade_steps": ('INT', {"forceInput": True, "default": 20}),

                "lightning_sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"forceInput": True, "default": "dpmpp_sde"}),
                "lightning_scheduler_name": (comfy.samplers.KSampler.SCHEDULERS, {"forceInput": True, "default": "simple"}),
                "lightning_cfg_scale": ('FLOAT', {"forceInput": True, "default": 1.2}),
                "lightning_steps": ('INT', {"forceInput": True, "default": 6}),
            }
        }

    def select_model_concept(self, cascade_stage_a, cascade_stage_b, cascade_stage_c, cascade_clip,
                             model_concept = 'Normal', lightning_selector = "SAFETENSOR", lightning_model_step = 8, lightning_sampler = False,
                             normal_sampler_name = 'euler', normal_scheduler_name = 'normal', normal_cfg_scale = 7, normal_steps = 12,
                             lcm_sampler_name = 'lcm', lcm_scheduler_name = 'sgm_uniform', lcm_cfg_scale = 1.2, lcm_steps = 6,
                             turbo_sampler_name = 'dpmpp_sde', turbo_scheduler_name = "karras", turbo_cfg_scale = 1.15, turbo_steps = 2,
                             cascade_sampler_name = 'euler_ancestral', cascade_scheduler_name = "simple", cascade_cfg_scale = 4, cascade_steps = 20,
                             lightning_sampler_name = 'dpmpp_sde', lightning_scheduler_name = "simple", lightning_cfg_scale = 1.2, lightning_steps = 6):

        match model_concept:
            case 'Normal':
                sampler_name = normal_sampler_name
                scheduler_name = normal_scheduler_name
                steps = normal_steps
                cfg_scale = normal_cfg_scale

            case 'LCM':
                sampler_name = lcm_sampler_name
                scheduler_name = lcm_scheduler_name
                steps = lcm_steps
                cfg_scale = lcm_cfg_scale

            case 'Turbo':
                sampler_name = turbo_sampler_name
                scheduler_name = turbo_scheduler_name
                steps = turbo_steps
                cfg_scale = turbo_cfg_scale

            case 'Cascade':
                sampler_name = cascade_sampler_name
                scheduler_name = cascade_scheduler_name
                steps = cascade_steps
                cfg_scale = cascade_cfg_scale

            case 'Lightning':
                if lightning_sampler == False:
                    sampler_name = lightning_sampler_name
                    scheduler_name = lightning_scheduler_name
                    steps = lightning_steps
                    cfg_scale = lightning_cfg_scale
                else:
                    sampler_name = 'euler'
                    scheduler_name = 'sgm_uniform'
                    steps = lightning_model_step
                    cfg_scale = 1

        if model_concept != 'Lightning':
            lightning_selector = None
            lightning_model_step = None

        if model_concept != 'Cascade':
            cascade_stage_a = None
            cascade_stage_b = None
            cascade_stage_c = None
            cascade_clip = None

        return (sampler_name, scheduler_name, steps, round(cfg_scale, 2), model_concept, lightning_selector, lightning_model_step, cascade_stage_a, cascade_stage_b, cascade_stage_c, cascade_clip)

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
                "strength_lcm_model": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
                "strength_lcm_clip": ("FLOAT", {"default": 1.0, "min": -20.0, "max": 20.0, "step": 0.01}),
            },
            "optional": {
                "model_concept": ("STRING", {"default": "Normal", "forceInput": True}),
                "concept_data": ("TUPLE", {"default": None, "forceInput": True}),

                "loaded_model": ('MODEL', {"forceInput": True, "default": None}),
                "loaded_clip": ('CLIP', {"forceInput": True, "default": None}),
                "loaded_vae": ('VAE', {"forceInput": True, "default": None}),
            },
        }

    def load_primere_ckpt(self, ckpt_name, use_yaml, strength_lcm_model, strength_lcm_clip,
                          model_concept = "Normal", concept_data = None,
                          lightning_selector = 'SAFETENSOR', lightning_model_step = 8,
                          cascade_stage_a = None, cascade_stage_b = None, cascade_stage_c = None, cascade_clip = None,
                          loaded_model = None, loaded_clip = None, loaded_vae = None):

        if concept_data is not None:
            if 'lightning_selector' in concept_data:
                lightning_selector = concept_data['lightning_selector']
            if 'lightning_model_step' in concept_data:
                lightning_model_step = concept_data['lightning_model_step']

            if 'cascade_stage_a' in concept_data:
                cascade_stage_a = concept_data['cascade_stage_a']
            if 'cascade_stage_b' in concept_data:
                cascade_stage_b = concept_data['cascade_stage_b']
            if 'cascade_stage_c' in concept_data:
                cascade_stage_c = concept_data['cascade_stage_c']
            if 'cascade_clip' in concept_data:
                cascade_clip = concept_data['cascade_clip']

        if model_concept == "Cascade" and cascade_stage_a is not None and cascade_stage_b is not None and cascade_stage_c is not None and cascade_clip is not None:
            MODEL_VERSION = 'SDXL_2048'
            is_sdxl = 1

            OUTPUT_VAE = nodes.VAELoader.load_vae(self, cascade_stage_a)[0]

            MODEL_B = nodes.UNETLoader.load_unet(self, cascade_stage_b)[0]
            MODEL_C = nodes.UNETLoader.load_unet(self, cascade_stage_c)[0]

            OUTPUT_CLIP = nodes.CLIPLoader.load_clip(self, cascade_clip, 'stable_cascade')[0]
            OUTPUT_MODEL = [MODEL_B, MODEL_C]

            return (OUTPUT_MODEL,) + (OUTPUT_CLIP,) + (OUTPUT_VAE,) + (MODEL_VERSION,)

        ModelConceptChanges = utility.ModelConceptNames(ckpt_name, model_concept, lightning_selector, lightning_model_step)
        ckpt_name = ModelConceptChanges['ckpt_name']
        lora_name = ModelConceptChanges['lora_name']
        unet_name = ModelConceptChanges['unet_name']
        lightningModeValid = ModelConceptChanges['lightningModeValid']

        path = Path(ckpt_name)
        ModelName = path.stem
        ModelConfigPath = path.parent.joinpath(ModelName + '.yaml')
        ModelConfigFullPath = Path(folder_paths.models_dir).joinpath('checkpoints').joinpath(ModelConfigPath)

        if (loaded_model is not None and loaded_clip is not None and loaded_vae is not None):
            LOADED_CHECKPOINT = []
            LOADED_CHECKPOINT.insert(0, loaded_model)
            LOADED_CHECKPOINT.insert(1, loaded_clip)
            LOADED_CHECKPOINT.insert(2, loaded_vae)
        else:
            if (os.path.isfile(ModelConfigFullPath) and use_yaml == True):
                ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
                print(ModelName + '.yaml file found and loading...')
                try:
                    LOADED_CHECKPOINT = comfy.sd.load_checkpoint(ModelConfigFullPath, ckpt_path, True, True, None, None, None)
                except Exception:
                    LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True)
            else:
                LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True)

        OUTPUT_MODEL = LOADED_CHECKPOINT[0]
        OUTPUT_CLIP = LOADED_CHECKPOINT[1]
        MODEL_VERSION = utility.get_value_from_cache('model_version', ModelName)
        if MODEL_VERSION is None:
            MODEL_VERSION = utility.getCheckpointVersion(OUTPUT_MODEL)
            utility.add_value_to_cache('model_version', ModelName, MODEL_VERSION)

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

        is_sdxl = 0
        match MODEL_VERSION:
            case 'SDXL_2048':
                is_sdxl = 1

        if model_concept == "LCM":
            SDXL_LORA = 'https://huggingface.co/latent-consistency/lcm-lora-sdxl/resolve/main/pytorch_lora_weights.safetensors?download=true'
            SD_LORA = 'https://huggingface.co/latent-consistency/lcm-lora-sdv1-5/resolve/main/pytorch_lora_weights.safetensors?download=true'
            DOWNLOADED_SD_LORA = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'lcm_lora_sd.safetensors')
            DOWNLOADED_SDXL_LORA = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'lcm_lora_sdxl.safetensors')

            if os.path.exists(DOWNLOADED_SD_LORA) == False:
                print('Downloading SD LCM LORA....')
                reqsdlcm = requests.get(SD_LORA, allow_redirects=True)
                if reqsdlcm.status_code == 200 and reqsdlcm.ok == True:
                    open(DOWNLOADED_SD_LORA, 'wb').write(reqsdlcm.content)
                else:
                    print('ERROR: Cannot dowload SD LCM Lora')

            if os.path.exists(DOWNLOADED_SDXL_LORA) == False:
                print('Downloading SDXL LCM LORA....')
                reqsdxllcm = requests.get(SDXL_LORA, allow_redirects=True)
                if reqsdxllcm.status_code == 200 and reqsdxllcm.ok == True:
                    open(DOWNLOADED_SDXL_LORA, 'wb').write(reqsdxllcm.content)
                else:
                    print('ERROR: Cannot dowload SDXL LCM Lora')

            if is_sdxl == 0:
                LORA_PATH = DOWNLOADED_SD_LORA
            else:
                LORA_PATH = DOWNLOADED_SDXL_LORA

            if os.path.exists(LORA_PATH) == True:
                if strength_lcm_model > 0 or strength_lcm_clip > 0:
                    print('LCM mode on')
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

                    MODEL_LORA, CLIP_LORA = comfy.sd.load_lora_for_models(OUTPUT_MODEL, OUTPUT_CLIP, lora, strength_lcm_model, strength_lcm_clip)

                    OUTPUT_MODEL = lcm(self, MODEL_LORA, False)
                    OUTPUT_CLIP = CLIP_LORA

        if model_concept == "Lightning" and lightningModeValid == True and loaded_model is None:
            OUTPUT_MODEL = utility.LightningConceptModel(self, model_concept, lightningModeValid, lightning_selector, lightning_model_step, OUTPUT_MODEL, lora_name, unet_name)

        return (OUTPUT_MODEL,) + (OUTPUT_CLIP,) + (LOADED_CHECKPOINT[2],) + (MODEL_VERSION,)

class AnyType(str):
    def __ne__(self, __value: object) -> bool:
        return False

class PrimerePromptSwitch:
    any_typ = AnyType("*")

    RETURN_TYPES = (any_typ, any_typ, any_typ, any_typ, any_typ, "INT")
    RETURN_NAMES = ("PROMPT+", "PROMPT-", "SUBPATH", "MODEL", "ORIENTATION", "SELECTED_INDEX")
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
                "subpath_1": (any_typ,),
                "model_1": (any_typ,),
                "orientation_1": (any_typ,),
            },
        }

    def promptswitch(self, *args, **kwargs):
        selected_index = int(kwargs['select'])
        input_namep = f"prompt_pos_{selected_index}"
        input_namen = f"prompt_neg_{selected_index}"
        input_subpath = f"subpath_{selected_index}"
        input_model = f"model_{selected_index}"
        input_orientation = f"orientation_{selected_index}"

        if input_subpath not in kwargs:
            kwargs[input_subpath] = None

        if input_model not in kwargs:
            kwargs[input_model] = None

        if input_orientation not in kwargs:
            kwargs[input_orientation] = None

        if input_namep in kwargs:
            return (kwargs[input_namep], kwargs[input_namen], kwargs[input_subpath], kwargs[input_model], kwargs[input_orientation], selected_index)
        else:
            print(f"PrimerePromptSwitch: invalid select index (ignored)")
            return (None, None, None, None, None, selected_index)

class PrimereSeed:
  RETURN_TYPES = ("INT",)
  RETURN_NAMES = ("SEED",)
  FUNCTION = "seed"
  CATEGORY = TREE_DASHBOARD

  @classmethod
  def INPUT_TYPES(cls):
    return {
        "required": {
            "seed": ("INT", {"default": -1, "min": -1, "max": 0xffffffffffffffff}),
       },
    }

  def seed(self, seed = 0):
    return (seed,)


class PrimereFractalLatent:
    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls):
        pln = PowerLawNoise('cpu')
        return {
            "required": {
                # "batch_size": ("INT", {"default": 1, "max": 64, "min": 1, "step": 1}),
                "width": ("INT", {"default": 512, "max": 8192, "min": 64, "forceInput": True}),
                "height": ("INT", {"default": 512, "max": 8192, "min": 64, "forceInput": True}),
                # "resampling": (["nearest-exact", "bilinear", "area", "bicubic", "bislerp"],),
                "rand_noise_type": ("BOOLEAN", {"default": False}),
                "noise_type": (pln.get_noise_types(),),
                # "scale": ("FLOAT", {"default": 1.0, "max": 1024.0, "min": 0.01, "step": 0.001}),
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

                "extra_variation": ("BOOLEAN", {"default": False, "label_on": "ON", "label_off": "OFF"}),
                # "variation_seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                # "variation_increment": ("FLOAT", {"default": 0.01, "min": 0.01, "max": 1.0, "step": 0.01}),
                # "variation_strength": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "optional_vae": ("VAE",),
            }
        }

    RETURN_TYPES = ("LATENT", "IMAGE")
    RETURN_NAMES = ("LATENTS", "PREVIEWS")
    FUNCTION = "primere_latent_noise"
    CATEGORY = TREE_DASHBOARD

    def primere_latent_noise(self, width, height, rand_noise_type, noise_type, rand_alpha_exponent, alpha_exponent, alpha_exp_rand_min, alpha_exp_rand_max, rand_modulator, modulator, modulator_rand_min, modulator_rand_max, noise_seed, seed, rand_device, device, optional_vae = None, extra_variation = False):
        if extra_variation == True:
            rand_device = True
            rand_alpha_exponent = True
            rand_modulator = True
            rand_noise_type = True
            alpha_exp_rand_min = -12.00
            alpha_exp_rand_max = 7.00
            modulator_rand_min = 0.10
            modulator_rand_max = 2.00
            noise_seed = seed

        if rand_noise_type == True:
            pln = PowerLawNoise(device)
            noise_type = random.choice(pln.get_noise_types())

        if rand_device == True:
            device = random.choice(["cpu", "cuda"])

        if extra_variation == True and (noise_type == 'white' or noise_type == 'violet'):
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
            return {'samples': latents}, tensors

        encoder = nodes.VAEEncode()
        latents = []
        for tensor in tensors:
            tensor = tensor.unsqueeze(0)
            latents.append(encoder.encode(optional_vae, tensor)[0]['samples'])

        latents = torch.cat(latents)
        return {'samples': latents}, tensors

class PrimereCLIP:
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("COND+", "COND-", "PROMPT+", "PROMPT-", "PROMPT L+", "PROMPT L-")
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

        return {
            "required": {
                "clip": ("CLIP", ),
                "model_version": ("STRING", {"default": 'BaseModel_1024', "forceInput": True}),
                "positive_prompt": ("STRING", {"forceInput": True}),
                "negative_prompt": ("STRING", {"forceInput": True}),
                "negative_strength": ("FLOAT", {"default": 1.2, "min": 0.0, "max": 10.0, "step": 0.01}),
                "use_int_style": ("BOOLEAN", {"default": False}),
                "int_style_pos": (['None'] + sorted(list(cls.default_pos.keys())),),
                "int_style_pos_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "int_style_neg": (['None'] + sorted(list(cls.default_neg.keys())),),
                "int_style_neg_strength": ("FLOAT", {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}),
                "adv_encode": ("BOOLEAN", {"default": False}),
                "token_normalization": (["none", "mean", "length", "length+mean"],),
                "weight_interpretation": (["comfy", "A1111", "compel", "comfy++", "down_weight"],),
                # "affect_pooled": ("BOOLEAN", {"default": False}),
            },
            "optional": {
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
            }
        }

    def clip_encode(self, clip, negative_strength, int_style_pos_strength, int_style_neg_strength, opt_pos_strength, opt_neg_strength, style_pos_strength, style_neg_strength, int_style_pos, int_style_neg, adv_encode, token_normalization, weight_interpretation, sdxl_l_strength, copy_prompt_to_l = True, width = 1024, height = 1024, positive_prompt = "", negative_prompt = "", model_keywords = None, lora_keywords = None, lycoris_keywords = None, embedding_pos = None, embedding_neg = None, opt_pos_prompt = "", opt_neg_prompt = "", style_position = False, style_neg_prompt = "", style_pos_prompt = "", sdxl_positive_l = "", sdxl_negative_l = "", use_int_style = False, model_version = "BaseModel_1024", model_concept = "Normal"):
        is_sdxl = 0
        match model_version:
            case 'SDXL_2048':
                is_sdxl = 1

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

        if (model_version == 'BaseModel_1024'):
            adv_encode = False

        if (adv_encode == True):
            if (is_sdxl == 0):

                if model_concept == 'Cascade':
                    positive_text = utility.clear_cascade(positive_text)
                    negative_text = utility.clear_cascade(negative_text)

                embeddings_final_pos, pooled_pos = advanced_encode(clip, positive_text, token_normalization, weight_interpretation, w_max = 1.0, apply_to_pooled = True)
                embeddings_final_neg, pooled_neg = advanced_encode(clip, negative_text, token_normalization, weight_interpretation, w_max = 1.0, apply_to_pooled = True)

                return ([[embeddings_final_pos, {"pooled_output": pooled_pos}]], [[embeddings_final_neg, {"pooled_output": pooled_neg}]], positive_text, negative_text, "", "")
            else:
                # embeddings_final_pos, pooled_pos = advanced_encode_XL(clip, sdxl_positive_l, positive_text, token_normalization, weight_interpretation, w_max = 1.0, clip_balance = sdxl_balance_l, apply_to_pooled = True)
                # embeddings_final_neg, pooled_neg = advanced_encode_XL(clip, sdxl_negative_l, negative_text, token_normalization, weight_interpretation, w_max = 1.0, clip_balance = sdxl_balance_l, apply_to_pooled = True)
                # return ([[embeddings_final_pos, {"pooled_output": pooled_pos}]],[[embeddings_final_neg, {"pooled_output": pooled_neg}]], positive_text, negative_text, sdxl_positive_l, sdxl_negative_l)
                if model_concept == 'Cascade':
                    positive_text = utility.clear_cascade(positive_text)
                    negative_text = utility.clear_cascade(negative_text)

                tokens_p = clip.tokenize(positive_text)
                if 'l' in clip.tokenize(sdxl_positive_l):
                    tokens_p["l"] = clip.tokenize(sdxl_positive_l)["l"]
                    if len(tokens_p["l"]) != len(tokens_p["g"]):
                        empty = clip.tokenize("")
                        while len(tokens_p["l"]) < len(tokens_p["g"]):
                            tokens_p["l"] += empty["l"]
                        while len(tokens_p["l"]) > len(tokens_p["g"]):
                            tokens_p["g"] += empty["g"]

                tokens_n = clip.tokenize(negative_text)
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

                return ([[cond_p, {"pooled_output": pooled_p, "width": width, "height": height, "crop_w": 0, "crop_h": 0, "target_width": width, "target_height": height}]], [[cond_n, {"pooled_output": pooled_n, "width": width, "height": height, "crop_w": 0, "crop_h": 0, "target_width": width, "target_height": height}]], positive_text, negative_text, sdxl_positive_l, sdxl_negative_l)

        else:
            if model_concept == 'Cascade':
                positive_text = utility.clear_cascade(positive_text)
                negative_text = utility.clear_cascade(negative_text)

            tokens = clip.tokenize(positive_text)
            cond_pos, pooled_pos = clip.encode_from_tokens(tokens, return_pooled = True)

            tokens = clip.tokenize(negative_text)
            cond_neg, pooled_neg = clip.encode_from_tokens(tokens, return_pooled = True)

            return ([[cond_pos, {"pooled_output": pooled_pos}]], [[cond_neg, {"pooled_output": pooled_neg}]], positive_text, negative_text, "", "")

class PrimereResolution:
    RETURN_TYPES = ("INT", "INT", "INT",)
    RETURN_NAMES = ("WIDTH", "HEIGHT", "SQUARE_SHAPE",)
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
                # "force_768_SD1x": ("BOOLEAN", {"default": True}),
                "basemodel_res": ([512, 768, 1024, 1280, 1600, 2048], {"default": 768}),
                "sdxlmodel_res": ([512, 768, 1024, 1280, 1600, 2048], {"default": 1024}),
                "turbo_res": ([512, 768, 1024, 1280, 1600, 2048], {"default": 512}),
                "rnd_orientation": ("BOOLEAN", {"default": False}),
                "orientation": (["Horizontal", "Vertical"], {"default": "Horizontal"}),
                "round_to_standard": ("BOOLEAN", {"default": False}),

                "calculate_by_custom": ("BOOLEAN", {"default": False}),
                "custom_side_a": ("FLOAT", {"default": 1.6, "min": 1.0, "max": 100.0, "step": 0.05}),
                "custom_side_b": ("FLOAT", {"default": 2.8, "min": 1.0, "max": 100.0, "step": 0.05}),
            },
            "optional": {
                "seed": ("INT", {"default": 0, "min": -1, "max": 0xffffffffffffffff, "forceInput": True}),
                "model_version": ("STRING", {"default": 'BaseModel_1024', "forceInput": True}),
                "model_concept": ("STRING", {"default": "Normal", "forceInput": True}),
            }
        }

    def calculate_imagesize(self, ratio: str, basemodel_res: int, sdxlmodel_res: int, turbo_res: int, rnd_orientation: bool, orientation: str, round_to_standard: bool, calculate_by_custom: bool, custom_side_a: float, custom_side_b: float, seed: int = 0, model_version: str = "BaseModel_1024", model_concept = "Normal"):
        if seed < 1:
            seed = random.randint(0, 9)

        if rnd_orientation == True:
            if (seed % 2) == 0:
                orientation = "Horizontal"
            else:
                orientation = "Vertical"

        if model_concept == "Turbo":
            basemodel_res = turbo_res
            sdxlmodel_res = turbo_res

        if model_version != 'SDXL_2048':
            square_shape = basemodel_res
            match basemodel_res:
                case 512:
                    model_version = 'BaseModel_768'
                case 768:
                    model_version = 'BaseModel_1024'
                case 1024:
                    model_version = 'BaseModel_mod_1024'
                case 1280:
                    model_version = 'BaseModel_mod_1280'
                case 1600:
                    model_version = 'BaseModel_mod_1600'
                case 2048:
                    model_version = 'BaseModel_mod_2048'
        else:
            square_shape = sdxlmodel_res
            match sdxlmodel_res:
                case 512:
                    model_version = 'SDXLModel_mod_768'
                case 768:
                    model_version = 'SDXLModel_mod_1024'
                case 1024:
                    model_version = 'SDXL_2048'
                case 1280:
                    model_version = 'SDXLModel_mod_1280'
                case 1600:
                    model_version = 'SDXLModel_mod_1600'
                case 2048:
                    model_version = 'SDXLModel_mod_2048'

        dimensions = utility.calculate_dimensions(self, ratio, orientation, round_to_standard, model_version, calculate_by_custom, custom_side_a, custom_side_b)
        dimension_x = dimensions[0]
        dimension_y = dimensions[1]
        return (dimension_x, dimension_y, square_shape,)

class PrimereResolutionMultiplier:
    RETURN_TYPES = ("INT", "INT", "FLOAT", "IMAGE")
    RETURN_NAMES = ("WIDTH", "HEIGHT", "UPSCALE_RATIO", "IMAGE")
    FUNCTION = "multiply_imagesize"
    CATEGORY = TREE_DASHBOARD
    upscale_methods = ["nearest-exact", "bilinear", "area", "bicubic", "lanczos"]

    @ classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "use_multiplier": ("BOOLEAN", {"default": True}),
                "width": ('INT', {"forceInput": True, "default": 512}),
                "height": ('INT', {"forceInput": True, "default": 512}),
                "multiply_sd": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.02}),
                "multiply_sdxl": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.02}),
                "multiply_turbo": ("FLOAT", {"default": 2.0, "min": 0.1, "max": 8.0, "step": 0.02}),
            },
            "optional": {
                "model_version": ("STRING", {"default": 'BaseModel_1024', "forceInput": True}),
                "model_concept": ("STRING", {"default": "Normal", "forceInput": True}),
                "triggered_prescale": ("BOOLEAN", {"default": False}),
                "image": ("IMAGE", {"forceInput": True}),
                "area_trigger_mpx": ("FLOAT", {"default": 0.60, "min": 0.01, "max": round(pow(utility.MAX_RESOLUTION, 2) / 1000000, 2), "step": 0.01}),
                "area_target_mpx": ("FLOAT", {"default": 1.05, "min": 0.25, "max": round(pow(utility.MAX_RESOLUTION, 2) / 1000000, 2), "step": 0.01}),
                "upscale_model": (['None'] + folder_paths.get_filename_list("upscale_models"), {"default": 'None'}),
                "upscale_method": (cls.upscale_methods, {"default": 'bicubic'}),
            }
        }

    def multiply_imagesize(self, width: int, height: int, use_multiplier: bool, multiply_sd: float, multiply_sdxl: float, multiply_turbo: float, model_version: str = "BaseModel_1024", model_concept: str = "Normal", triggered_prescale = False, image = None, area_trigger_mpx = 0.60, area_target_mpx = 1.05, upscale_model = 'None', upscale_method = 'bicubic'):
        if use_multiplier == False:
            return (width, height, 1, image)

        is_sdxl = 0
        match model_version:
            case 'SDXL_2048':
                is_sdxl = 1

        if image is not None:
            width = image.shape[2]
            height = image.shape[1]

        if (is_sdxl == 1):
            dimension_x = round(width * multiply_sdxl)
            dimension_y = round(height * multiply_sdxl)
            ratio = round(multiply_sdxl, 2)
        else:
            dimension_x = round(width * multiply_sd)
            dimension_y = round(height * multiply_sd)
            ratio = round(multiply_sd, 2)

        if model_concept == "Turbo":
            dimension_x = round(width * multiply_turbo)
            dimension_y = round(height * multiply_turbo)
            ratio = round(multiply_turbo, 2)

        if triggered_prescale == True and use_multiplier == True:
            upscale_to_mpx = (dimension_x * dimension_y) / (1024 * 1024)
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

class PrimereStepsCfg:
  RETURN_TYPES = ("INT", "FLOAT")
  RETURN_NAMES = ("STEPS", "CFG")
  FUNCTION = "steps_cfg"
  CATEGORY = TREE_DASHBOARD

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "steps": ("INT", {"default": 12, "min": 1, "max": 1000, "step": 1}),
        "cfg": ("FLOAT", {"default": 7, "min": 0.1, "max": 100, "step": 0.01}),
      },
    }

  def steps_cfg(self, steps = 12, cfg = 7):
    return (steps, round(cfg, 2),)

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
              positive_prompt = re.sub("(\(" + embedding_name + ":\d+\.\d+\))|(\(" + embedding_name + ":\d+\))|(" + embedding_name + ":\d+\.\d+)|(" + embedding_name + ":\d+)|(" + embedding_name + ":)|(\(" + embedding_name + "\))|(" + embedding_name + ")", "", positive_prompt)
              negative_prompt = re.sub("(\(" + embedding_name + ":\d+\.\d+\))|(\(" + embedding_name + ":\d+\))|(" + embedding_name + ":\d+\.\d+)|(" + embedding_name + ":\d+)|(" + embedding_name + ":)|(\(" + embedding_name + "\))|(" + embedding_name + ")", "", negative_prompt)
              positive_prompt = re.sub(r'(, )\1+', r', ', positive_prompt).strip(', ').replace(' ,', ',')
              negative_prompt = re.sub(r'(, )\1+', r', ', negative_prompt).strip(', ').replace(' ,', ',')

      if len(NETWORK_START) > 0:
         NETWORK_END = ['\n', '>', ' ', ',', '}', ')', '|'] + NETWORK_START
         positive_prompt = utility.clear_prompt(NETWORK_START, NETWORK_END, positive_prompt)
         negative_prompt = utility.clear_prompt(NETWORK_START, NETWORK_END, negative_prompt)

      if model_concept == 'Cascade':
          positive_prompt = utility.clear_cascade(positive_prompt)
          negative_prompt = utility.clear_cascade(negative_prompt)

      return (positive_prompt, negative_prompt,)

class PrimereNetworkTagLoader:
  RETURN_TYPES = ("MODEL", "CLIP", "LORA_STACK", "LYCORIS_STACK", "HYPERNETWORK_STACK", "MODEL_KEYWORD", "MODEL_KEYWORD")
  RETURN_NAMES = ("MODEL", "CLIP", "LORA_STACK", "LYCORIS_STACK", "HYPERNETWORK_STACK", "LORA_KEYWORD", "LYCORIS_KEYWORD")
  FUNCTION = "load_networks"
  CATEGORY = TREE_DASHBOARD
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
          }
      }

  def load_networks(self, model, clip, positive_prompt, process_lora, process_lycoris, process_hypernetwork, copy_weight_to_clip, lora_clip_custom_weight, lycoris_clip_custom_weight, use_lora_keyword, use_lycoris_keyword, lora_keyword_placement, lycoris_keyword_placement, lora_keyword_selection, lycoris_keyword_selection, lora_keywords_num, lycoris_keywords_num, lora_keyword_weight, lycoris_keyword_weight, hypernetwork_safe_load = True):
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
    def INPUT_TYPES(s):
        return {
            "required": {
                "model_name": ('CHECKPOINT_NAME', {"forceInput": True, "default": ""}),
                "use_model_keyword": ("BOOLEAN", {"default": False}),
                "model_keyword_placement": (["First", "Last"], {"default": "Last"}),
                "model_keyword_selection": (["Select in order", "Random select"], {"default": "Select in order"}),
                "model_keywords_num": ("INT", {"default": 1, "min": 1, "max": 50, "step": 1}),
                "model_keyword_weight": ("FLOAT", {"default": 1.0, "min": 0, "max": 10.0, "step": 0.1}),
            },
        }

    def load_ckpt_keyword(self, model_name, use_model_keyword, model_keyword_placement, model_keyword_selection, model_keywords_num, model_keyword_weight):
        model_keyword = [None, None]

        if use_model_keyword == True:
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
                            if model_keyword_selection == 'Select in order':
                                list_of_keyword_items = keyword_list[:model_keywords_num]
                            else:
                                list_of_keyword_items = random.sample(keyword_list, model_keywords_num)
                            keywords = ", ".join(list_of_keyword_items)

                    if (model_keyword_weight != 1):
                        keywords = '(' + keywords + ':' + str(round(model_keyword_weight, 1)) + ')'

                    model_keyword = [keywords, model_keyword_placement]

        return (model_keyword,)

class PrimereConceptDataTuple:
    RETURN_TYPES = ("TUPLE",)
    RETURN_NAMES = ("CONCEPT_DATA",)
    FUNCTION = "load_concept_collector"
    CATEGORY = TREE_DASHBOARD

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "lightning_selector": ("STRING", {"default": "SAFETENSOR", "forceInput": True}),
                "lightning_model_step": ("INT", {"default": 8, "forceInput": True}),

                "cascade_stage_a": ("STRING", {"forceInput": True}),
                "cascade_stage_b": ("STRING", {"forceInput": True}),
                "cascade_stage_c": ("STRING", {"forceInput": True}),
                "cascade_clip": ("STRING", {"forceInput": True}),
            },
        }

    def load_concept_collector(self, **kwargs):
        return (kwargs,)