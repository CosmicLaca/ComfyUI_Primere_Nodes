import os
import torch
import comfy
import comfy.sd
import comfy.utils
import folder_paths
import nodes
import comfy_extras.nodes_sd3 as nodes_sd3
import comfy_extras.nodes_model_advanced as nodes_model_advanced
from comfy import model_management
from pathlib import Path
from .tree import PRIMERE_ROOT
from . import utility
from . import nf4_helper
from . import sana_utils
from .gguf import nodes as gguf_nodes
import difflib
import numpy as np
import pyrallis
from ComfyUI_ExtraModels.PixArt.loader import load_pixart
from ComfyUI_ExtraModels.PixArt.conf import pixart_conf
from diffusers import UNet2DConditionModel, EulerDiscreteScheduler
from .kolors.pipelines.pipeline_stable_diffusion_xl_chatglm_256 import StableDiffusionXLPipeline
from .kolors.models.tokenization_chatglm import ChatGLMTokenizer
from .kolors.models.modeling_chatglm import ChatGLMModel
from ComfyUI_ExtraModels.Sana.conf import sana_conf
from ComfyUI_ExtraModels.Sana.loader import load_sana
from ComfyUI_ExtraModels.VAE.conf import vae_conf
from ComfyUI_ExtraModels.VAE.loader import EXVAE
from ComfyUI_ExtraModels.utils.dtype import string_to_dtype
from transformers import AutoTokenizer, T5Tokenizer, T5EncoderModel, AutoModelForCausalLM, BitsAndBytesConfig
from .sana.diffusion.model.builder import build_model
from .sana.diffusion.model.dc_ae.efficientvit.ae_model_zoo import create_dc_ae_model_cfg
from .sana.diffusion.model.dc_ae.efficientvit.models.efficientvit.dc_ae import DCAE
from .sana.diffusion.utils.config import SanaConfig
from .sana.pipeline.sana_pipeline import SanaPipeline


def resolve_symlink(ckpt_name):
    fullpathFile = folder_paths.get_full_path('checkpoints', ckpt_name)
    if not os.path.islink(str(fullpathFile)):
        return None, None, None
    File_link = Path(str(fullpathFile)).resolve()
    model_ext = os.path.splitext(File_link)[1].lower()
    linkName_U = str(folder_paths.folder_names_and_paths["diffusion_models"][0][0])
    linkName_D = str(folder_paths.folder_names_and_paths["diffusion_models"][0][1])
    linkedFileName = str(File_link).replace(linkName_U + '\\', '').replace(linkName_D + '\\', '')
    if str(Path(linkName_U).stem) in linkedFileName:
        linkedFileName = linkedFileName.split(Path(linkName_U).stem + '\\', 1)[1]
    if str(Path(linkName_D).stem) in linkedFileName:
        linkedFileName = linkedFileName.split(Path(linkName_D).stem + '\\', 1)[1]
    return File_link, linkedFileName, model_ext


def apply_lora(loader_self, model, lora_path, strength):
    if not os.path.exists(lora_path) or strength == 0:
        return model
    lora = None
    if loader_self.loaded_lora is not None:
        if loader_self.loaded_lora[0] == lora_path:
            lora = loader_self.loaded_lora[1]
        else:
            temp = loader_self.loaded_lora
            loader_self.loaded_lora = None
            del temp
    if lora is None:
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        loader_self.loaded_lora = (lora_path, lora)
    return comfy.sd.load_lora_for_models(model, None, lora, strength, 0)[0]


def pick_lora(concept_data):
    if concept_data.get('speed_lora') == True:
        return concept_data.get('speed_lora_name'), concept_data.get('speed_lora_strength', 1)
    if concept_data.get('srpo_lora') == True:
        return concept_data.get('srpo_lora_name'), concept_data.get('srpo_lora_strength', 1)
    return None, 0


def load_sd_model(loader_self, ckpt_name, use_yaml, model_config_full_path, concept_data):
    if os.path.isfile(model_config_full_path) and use_yaml:
        ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)
        try:
            LOADED_CHECKPOINT = comfy.sd.load_checkpoint(model_config_full_path, ckpt_path, True, True, None, None, None)
        except Exception:
            LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(loader_self, ckpt_name)
    else:
        LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(loader_self, ckpt_name)
    OUTPUT_MODEL = LOADED_CHECKPOINT[0]
    OUTPUT_CLIP = LOADED_CHECKPOINT[1]
    vae_selection = concept_data.get('vae_selection', True)
    vae_name = concept_data.get('vae', None)
    if not vae_selection and vae_name:
        OUTPUT_VAE = utility.vae_loader_class.load_vae(vae_name)[0]
    elif len(LOADED_CHECKPOINT) >= 3 and type(LOADED_CHECKPOINT[2]).__name__ == 'VAE':
        OUTPUT_VAE = LOADED_CHECKPOINT[2]
    elif vae_name:
        OUTPUT_VAE = utility.vae_loader_class.load_vae(vae_name)[0]
    else:
        OUTPUT_VAE = LOADED_CHECKPOINT[2]
    return OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE


def load_sd3_model(loader_self, ckpt_name, concept_data):
    LOADED_CHECKPOINT = []
    File_link, linkedFileName, model_ext = resolve_symlink(ckpt_name)
    if File_link and model_ext == '.gguf':
        OUTPUT_MODEL = gguf_nodes.UnetLoaderGGUF.load_unet(loader_self, linkedFileName)[0]
    else:
        LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(loader_self, ckpt_name)
        OUTPUT_MODEL = LOADED_CHECKPOINT[0]
    clip_selection = concept_data.get('clip_selection', True)
    clip_from_ckpt = len(LOADED_CHECKPOINT) >= 2 and type(LOADED_CHECKPOINT[1]).__name__ == 'CLIP'
    if not clip_selection or not clip_from_ckpt:
        OUTPUT_CLIP = nodes_sd3.TripleCLIPLoader.execute(concept_data.get('encoder_3'), concept_data.get('encoder_2'), concept_data.get('encoder_1'))[0]
    else:
        OUTPUT_CLIP = LOADED_CHECKPOINT[1]
    vae_name = concept_data.get('vae', None)
    if len(LOADED_CHECKPOINT) >= 3 and type(LOADED_CHECKPOINT[2]).__name__ == 'VAE':
        OUTPUT_VAE = LOADED_CHECKPOINT[2]
    elif vae_name:
        OUTPUT_VAE = utility.vae_loader_class.load_vae(vae_name)[0]
    else:
        OUTPUT_VAE = LOADED_CHECKPOINT[2]
    lora_name, lora_strength = pick_lora(concept_data)
    if lora_name:
        lora_path = folder_paths.get_full_path('loras', lora_name)
        if lora_path:
            OUTPUT_MODEL = apply_lora(loader_self, OUTPUT_MODEL, lora_path, lora_strength)
    return OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE


def load_stable_cascade_model(loader_self, ckpt_name, concept_data):
    stage_b = concept_data.get('encoder_1', None)
    cascade_clip = concept_data.get('encoder_3', None)
    vae_name = concept_data.get('vae', None)
    OUTPUT_CLIP = nodes.CLIPLoader.load_clip(loader_self, cascade_clip, 'stable_cascade')[0]
    OUTPUT_VAE = utility.vae_loader_class.load_vae(vae_name)[0]
    MODEL_B = nodes.UNETLoader.load_unet(loader_self, stage_b, 'default')[0]
    File_link, linkedFileName, _ = resolve_symlink(ckpt_name)
    if File_link:
        MODEL_C = nodes.UNETLoader.load_unet(loader_self, linkedFileName, 'default')[0]
    else:
        MODEL_C = nodes.UNETLoader.load_unet(loader_self, ckpt_name, 'default')[0]
    return [MODEL_B, MODEL_C], OUTPUT_CLIP, OUTPUT_VAE


def load_zimage_model(loader_self, ckpt_name, concept_data):
    weight_dtype = concept_data.get('weight_dtype', 'default')
    if 'e4m3fn' in ckpt_name:
        weight_dtype = 'fp8_e4m3fn'
    if 'e5m2' in ckpt_name:
        weight_dtype = 'fp8_e5m2'
    File_link, linkedFileName, model_ext = resolve_symlink(ckpt_name)
    if File_link:
        if 'diffusion_models' in str(File_link) or 'unet' in str(File_link):
            if model_ext == '.gguf':
                OUTPUT_MODEL = gguf_nodes.UnetLoaderGGUF.load_unet(loader_self, linkedFileName)[0]
            else:
                try:
                    OUTPUT_MODEL = nodes.UNETLoader.load_unet(loader_self, linkedFileName, weight_dtype)[0]
                except Exception:
                    OUTPUT_MODEL = nf4_helper.UNETLoaderNF4.load_nf4unet(linkedFileName)[0]
        else:
            OUTPUT_MODEL = nodes.CheckpointLoaderSimple.load_checkpoint(loader_self, ckpt_name)[0]
    else:
        OUTPUT_MODEL = nodes.CheckpointLoaderSimple.load_checkpoint(loader_self, ckpt_name)[0]
    encoder_1 = concept_data.get('encoder_1', None)
    clip_ext = os.path.splitext(encoder_1)[1].lower() if encoder_1 else ''
    if clip_ext == '.gguf':
        OUTPUT_CLIP = gguf_nodes.CLIPLoaderGGUF.load_clip(loader_self, encoder_1, 'qwen_image')[0]
    else:
        OUTPUT_CLIP = nodes.CLIPLoader.load_clip(loader_self, encoder_1, 'flux2')[0]
    OUTPUT_VAE = utility.vae_loader_class.load_vae(concept_data.get('vae', None))[0]
    return OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE


def load_flux_model(loader_self, ckpt_name, concept_data):
    weight_dtype = concept_data.get('weight_dtype', 'default')
    is_gguf_model = False
    File_link, linkedFileName, model_ext = resolve_symlink(ckpt_name)
    if File_link:
        if 'diffusion_models' in str(File_link) or 'unet' in str(File_link):
            if model_ext == '.gguf':
                OUTPUT_MODEL = gguf_nodes.UnetLoaderGGUF.load_unet(loader_self, linkedFileName)[0]
                is_gguf_model = True
            else:
                try:
                    OUTPUT_MODEL = nodes.UNETLoader.load_unet(loader_self, linkedFileName, weight_dtype)[0]
                except Exception:
                    OUTPUT_MODEL = nf4_helper.UNETLoaderNF4.load_nf4unet(linkedFileName)[0]
        else:
            OUTPUT_MODEL = nodes.CheckpointLoaderSimple.load_checkpoint(loader_self, linkedFileName)[0]
    else:
        OUTPUT_MODEL = nodes.CheckpointLoaderSimple.load_checkpoint(loader_self, ckpt_name)[0]
    encoder_1 = concept_data.get('encoder_1', None)
    encoder_2 = concept_data.get('encoder_2', None)
    clip_ext_1 = os.path.splitext(encoder_1)[1].lower() if encoder_1 else ''
    clip_ext_2 = os.path.splitext(encoder_2)[1].lower() if encoder_2 else ''
    if is_gguf_model or clip_ext_1 == '.gguf' or clip_ext_2 == '.gguf':
        OUTPUT_CLIP = gguf_nodes.DualCLIPLoaderGGUF.load_clip(loader_self, encoder_2, encoder_1, 'flux')[0]
    else:
        OUTPUT_CLIP = nodes.DualCLIPLoader.load_clip(loader_self, encoder_2, encoder_1, 'flux')[0]
    OUTPUT_VAE = utility.vae_loader_class.load_vae(concept_data.get('vae', None))[0]
    lora_name, lora_strength = pick_lora(concept_data)
    if lora_name:
        lora_path = folder_paths.get_full_path('loras', lora_name)
        if lora_path:
            OUTPUT_MODEL = apply_lora(loader_self, OUTPUT_MODEL, lora_path, lora_strength)
    return OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE


def load_auraflow_model(loader_self, ckpt_name, concept_data):
    File_link, linkedFileName, model_ext = resolve_symlink(ckpt_name)
    if File_link:
        if model_ext == '.gguf':
            OUTPUT_MODEL = gguf_nodes.UnetLoaderGGUF.load_unet(loader_self, linkedFileName)[0]
        else:
            OUTPUT_MODEL = nodes.UNETLoader.load_unet(loader_self, linkedFileName, 'default')[0]
    else:
        OUTPUT_MODEL = nodes.CheckpointLoaderSimple.load_checkpoint(loader_self, ckpt_name)[0]
    encoder_1 = concept_data.get('encoder_1', None)
    OUTPUT_CLIP = nodes.CLIPLoader.load_clip(loader_self, encoder_1, 'stable_diffusion')[0]
    OUTPUT_VAE = utility.vae_loader_class.load_vae(concept_data.get('vae', None))[0]
    return OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE


def load_pixart_model(loader_self, ckpt_name, concept_data):
    ckpt_path = folder_paths.get_full_path("checkpoints", ckpt_name)

    pixart_model_name = Path(ckpt_name).stem
    pixart_types = list(pixart_conf.keys())
    cutoff_list = list(np.around(np.arange(0.1, 1.01, 0.01).tolist(), 2))[::-1]
    is_found = []
    trycut = 0
    for trycut in cutoff_list:
        is_found = difflib.get_close_matches(pixart_model_name, pixart_types, cutoff=trycut)
        if len(is_found) > 0:
            break
    pixart_model_type = 'PixArtMS_Sigma_XL_2' if trycut <= 0.35 else is_found[0]
    model_conf = pixart_conf[pixart_model_type]

    OUTPUT_MODEL_MAIN = load_pixart(model_path=ckpt_path, model_conf=model_conf)

    encoder_1 = concept_data.get('encoder_1', None)
    OUTPUT_CLIP_MAIN = nodes.CLIPLoader.load_clip(loader_self, encoder_1, 'sd3')[0]

    refiner_model = concept_data.get('refiner_model', None)
    OUTPUT_MODEL_REFINER = None
    OUTPUT_CLIP_REFINER = None
    if concept_data.get('refiner') == True and refiner_model and refiner_model != 'None':
        REFINER_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(loader_self, refiner_model)
        OUTPUT_MODEL_REFINER = REFINER_CHECKPOINT[0]
        OUTPUT_CLIP_REFINER = REFINER_CHECKPOINT[1]
        OUTPUT_VAE = REFINER_CHECKPOINT[2]
    else:
        OUTPUT_VAE = utility.vae_loader_class.load_vae(concept_data.get('vae', None))[0]

    return {'main': OUTPUT_MODEL_MAIN, 'refiner': OUTPUT_MODEL_REFINER}, {'main': OUTPUT_CLIP_MAIN, 'refiner': OUTPUT_CLIP_REFINER}, OUTPUT_VAE


def load_playground_model(loader_self, ckpt_name, use_yaml, model_config_full_path, concept_data):
    OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE = load_sd_model(loader_self, ckpt_name, use_yaml, model_config_full_path, concept_data)
    sigma_max = concept_data.get('sigma_max', 120)
    sigma_min = concept_data.get('sigma_min', 0.002)
    OUTPUT_MODEL = nodes_model_advanced.ModelSamplingContinuousEDM.patch(loader_self, OUTPUT_MODEL, 'edm_playground_v2.5', sigma_max, sigma_min)[0]
    return OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE


def load_lightning_hyper_model(loader_self, ckpt_name, concept_data):
    model_concept = concept_data.get('model_concept')
    lora_path = None
    lora_strength = 1.0
    if concept_data.get('speed_lora') == True:
        lora_name = concept_data.get('speed_lora_name')
        if lora_name:
            lora_path = folder_paths.get_full_path('loras', lora_name)
            lora_strength = concept_data.get('speed_lora_strength', 1.0)

    if model_concept == 'Hyper':
        _, unet_name, _ = resolve_symlink(ckpt_name)
        if unet_name is not None:
            checkpoint_result = utility.BDanceConceptHelper(loader_self, model_concept, True, 'UNET', None, None, None, unet_name, None, lora_strength)
            OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE = checkpoint_result[0], checkpoint_result[1], checkpoint_result[2]
        else:
            LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(loader_self, ckpt_name)
            OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE = LOADED_CHECKPOINT[0], LOADED_CHECKPOINT[1], LOADED_CHECKPOINT[2]
        if lora_path:
            OUTPUT_MODEL = utility.BDanceConceptHelper(loader_self, model_concept, True, 'LORA', None, OUTPUT_MODEL, lora_path, None, None, lora_strength)
    else:
        LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(loader_self, ckpt_name)
        OUTPUT_MODEL = LOADED_CHECKPOINT[0]
        OUTPUT_CLIP = LOADED_CHECKPOINT[1]
        OUTPUT_VAE = LOADED_CHECKPOINT[2]
        if lora_path:
            OUTPUT_MODEL = utility.BDanceConceptHelper(loader_self, model_concept, True, 'LORA', None, OUTPUT_MODEL, lora_path, None, None, lora_strength)

    return OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE


def load_lcm_model(loader_self, ckpt_name, concept_data):
    LOADED_CHECKPOINT = nodes.CheckpointLoaderSimple.load_checkpoint(loader_self, ckpt_name)
    OUTPUT_MODEL = LOADED_CHECKPOINT[0]
    OUTPUT_CLIP = LOADED_CHECKPOINT[1]
    OUTPUT_VAE = LOADED_CHECKPOINT[2]

    MODEL_VERSION = utility.getModelType(ckpt_name, 'checkpoints')

    if concept_data.get('lcm_lora') == True:
        lora_name = concept_data.get('lcm_lora_name', None)
        if lora_name:
            lora_path = folder_paths.get_full_path('loras', lora_name)
            if lora_path:
                OUTPUT_MODEL = apply_lora(loader_self, OUTPUT_MODEL, lora_path, concept_data.get('lcm_lora_strength', 1.0))

    class ModelSamplingAdvanced(utility.ModelSamplingDiscreteLCM, nodes_model_advanced.LCM):
        pass

    m = OUTPUT_MODEL.clone()
    m.add_object_patch("model_sampling", ModelSamplingAdvanced())
    OUTPUT_MODEL = m

    return OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE


def load_sana_model(loader_self, ckpt_name, concept_data):
    encoder_path = concept_data.get('encoder_1', 'gemma-2-2b-it')
    weight_dtype_str = concept_data.get('weight_dtype', 'fp16')
    vae_name = concept_data.get('vae')
    precision = concept_data.get('precision', 'fp16')
    scheduler_name = concept_data.get('scheduler_name', 'flow_dpm-solver')

    device = model_management.get_torch_device()
    fullpathFile = folder_paths.get_full_path('checkpoints', ckpt_name)
    if os.path.islink(str(fullpathFile)):
        fullpathFile = Path(str(fullpathFile)).resolve()

    text_encoder_dir = os.path.join(folder_paths.models_dir, 'text_encoders', encoder_path)
    if not os.path.exists(text_encoder_dir):
        text_encoder_dir = os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', 'LLM', encoder_path)

    vae_path = folder_paths.get_full_path("vae", vae_name)
    text_encoder_dtype = model_management.text_encoder_dtype(device)

    if scheduler_name == 'flow_dpm-solver':
        dtype = utility.get_dtype_by_name(weight_dtype_str)
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
            if precision == '8-bit':
                quantization_config = BitsAndBytesConfig(load_in_8bit=True)
            elif precision == '4-bit':
                quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=text_encoder_dtype)
            else:
                quantization_config = None
            if '-4bit' in encoder_path:
                llm_model = AutoModelForCausalLM.from_pretrained(text_encoder_dir, torch_dtype=text_encoder_dtype)
            else:
                llm_model = AutoModelForCausalLM.from_pretrained(text_encoder_dir, quantization_config=quantization_config, torch_dtype=text_encoder_dtype)
            tokenizer.padding_side = "right"
            text_encoder = llm_model.get_decoder()

        text_encoder.to(device)
        state_dict = comfy.utils.load_torch_file(str(fullpathFile), safe_load=True)
        is_1600M = state_dict['final_layer.scale_shift_table'].shape[1] == 2240
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
            "caption_channels": text_encoder.config.hidden_size,
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
        unet.load_state_dict(state_dict, strict=False)
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
            'device': device,
        }
        SANA_VAE = sana_utils.first_stage_model(vae)
        SANA_CLIP = sana_utils.cond_stage_model(tokenizer, text_encoder)
    else:
        model_keys = list(sana_conf.keys())
        model_conf = sana_conf[model_keys[1]]
        SANA_MODEL = load_sana(model_path=str(fullpathFile), model_conf=model_conf)

        vae_config = vae_conf['dcae-f32c32-sana-1.0']
        SANA_VAE = EXVAE(vae_path, vae_config, string_to_dtype(weight_dtype_str.upper(), "vae"))

        tokenizer = AutoTokenizer.from_pretrained(text_encoder_dir)
        text_encoder_model = AutoModelForCausalLM.from_pretrained(text_encoder_dir, torch_dtype=text_encoder_dtype)
        tokenizer.padding_side = "right"
        text_encoder = text_encoder_model.get_decoder()
        if device != "cpu":
            text_encoder = text_encoder.to(device)

        SANA_CLIP = {
            "tokenizer": tokenizer,
            "text_encoder": text_encoder,
            "text_encoder_model": text_encoder_model,
        }

    return SANA_MODEL, SANA_CLIP, SANA_VAE


def load_kolors_model(loader_self, ckpt_name, concept_data):
    weight_dtype_str = concept_data.get('weight_dtype', 'fp16')
    precision = concept_data.get('precision', 'quant8')
    vae_name = concept_data.get('vae')

    dtype_map = {'bf16': torch.bfloat16, 'fp16': torch.float16, 'fp32': torch.float32}
    dtype = dtype_map.get(weight_dtype_str, torch.float16)

    model_name = Path(ckpt_name).stem
    fullpathFile = folder_paths.get_full_path('checkpoints', ckpt_name)
    if os.path.islink(str(fullpathFile)):
        link_path = Path(str(fullpathFile)).resolve()
        model_name = Path(link_path.parent.parent).stem

    model_path = os.path.join(folder_paths.models_dir, "diffusers", model_name)
    pbar = comfy.utils.ProgressBar(4)
    scheduler = EulerDiscreteScheduler.from_pretrained(model_path, subfolder='scheduler')
    unet = UNet2DConditionModel.from_pretrained(model_path, subfolder='unet', variant="fp16", revision=None, low_cpu_mem_usage=True).to(dtype).eval()
    pipeline = StableDiffusionXLPipeline(unet=unet, scheduler=scheduler)
    KOLORS_MODEL = {'pipeline': pipeline, 'dtype': dtype}
    pbar.update(1)

    pbar = comfy.utils.ProgressBar(2)
    text_encoder_path = os.path.join(model_path, "text_encoder")
    text_encoder = ChatGLMModel.from_pretrained(text_encoder_path, torch_dtype=torch.float16)
    if precision == 'quant8':
        try:
            text_encoder.quantize(8)
        except Exception:
            print('Quantization 8 failed...')
    elif precision == 'quant4':
        try:
            text_encoder.quantize(4)
        except Exception:
            print('Quantization 4 failed...')
    tokenizer = ChatGLMTokenizer.from_pretrained(text_encoder_path)
    pbar.update(1)
    CHATGLM3_MODEL = {'text_encoder': text_encoder, 'tokenizer': tokenizer}

    if not vae_name:
        raise ValueError("KwaiKolors requires an explicit VAE. Set 'vae' in the concept data (e.g. 'kolors\\diffusion_pytorch_model.fp16.safetensors').")
    OUTPUT_VAE = utility.vae_loader_class.load_vae(vae_name)[0]

    return KOLORS_MODEL, CHATGLM3_MODEL, OUTPUT_VAE
