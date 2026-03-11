import os
import comfy
import comfy.sd
import comfy.utils
import folder_paths
import nodes
import comfy_extras.nodes_sd3 as nodes_sd3
from pathlib import Path
from .tree import PRIMERE_ROOT
from . import utility
from . import nf4_helper
from .gguf import nodes as gguf_nodes


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
        OUTPUT_MODEL = apply_lora(loader_self, OUTPUT_MODEL, os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', lora_name), lora_strength)
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
        OUTPUT_MODEL = apply_lora(loader_self, OUTPUT_MODEL, os.path.join(PRIMERE_ROOT, 'Nodes', 'Downloads', lora_name), lora_strength)
    return OUTPUT_MODEL, OUTPUT_CLIP, OUTPUT_VAE
