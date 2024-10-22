import math
import comfy.model_sampling
import torch
from dynamicprompts.generators import RandomPromptGenerator
import hashlib
import chardet
import pandas
import re
from pathlib import Path
import difflib
# from ..utils import cache_file
import os
import json
import numpy as np
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
from urllib.parse import urlparse
import requests
import folder_paths
import comfy_extras.nodes_model_advanced as nodes_model_advanced
import nodes
# from ..utils import comfy_dir
import collections
import pytorch_lightning as pl
import torch.nn as nn
from PIL import Image, ImageOps, ImageSequence

from comfy.k_diffusion.sampling import default_noise_sampler
from comfy.ldm.modules.diffusionmodules.util import make_beta_schedule
from comfy.model_sampling import EPS
from comfy.samplers import KSAMPLER, calculate_sigmas
from comfy_extras.nodes_model_advanced import ModelSamplingDiscreteDistilled
from tqdm.auto import trange
import comfy.model_detection as model_detection

here = Path(__file__).parent.parent.absolute()
comfy_dir = str(here.parent.parent)
cache_dir = os.path.join(here, 'Nodes', '.cache')
cache_file = os.path.join(cache_dir, '.cache.json')

SUPPORTED_FORMATS = [".png", ".jpg", ".jpeg", ".webp", ".preview.png", ".preview.jpg", ".preview.jpeg",]
STANDARD_SIDES = np.arange(64, 2049, 16).tolist()
CASCADE_SIDES = np.arange(64, 2049, 16).tolist()
MAX_RESOLUTION = 8192
VALID_SHAPES = np.arange(512, 2049, 256).tolist()
PREVIEW_ROOT = os.path.join(comfy_dir, "web", "extensions", "PrimerePreviews", "images")
SUPPORTED_MODELS = ["SD1", "SD2", "SDXL", "SD3", "StableCascade", "Turbo", "Flux", "KwaiKolors", "Hunyuan", "Playground", "Pony", "LCM", "Lightning", "Hyper", "SSD", "SegmindVega", "KOALA", "StableZero", "SV3D", "AuraFlow", "SD09", "StableAudio"]
CONCEPT_RESOLUTIONS = {
                        "512": ['SD09', "Turbo"],
                        "768": ['SD1', 'SD2'],
                        "1024": ["SDXL", "SD3", "StableCascade", "Flux", "KwaiKolors", "Hunyuan", "Playground", "Pony", "LCM", "Lightning", "Hyper"]
                      }

PREVIEW_PATH_BY_TYPE = {
    "Checkpoint": os.path.join(PREVIEW_ROOT, "checkpoints"),
    "CSV Prompt": os.path.join(PREVIEW_ROOT, "styles"),
    "Lora": os.path.join(PREVIEW_ROOT, "loras"),
    "Lycoris": os.path.join(PREVIEW_ROOT, "lycoris"),
    "Hypernetwork": os.path.join(PREVIEW_ROOT, "hypernetworks"),
    "Embedding": os.path.join(PREVIEW_ROOT, "embeddings"),
}

WORKFLOW_SORT_LIST = ['exif_status', 'exif_data_count', 'meta_source', 'pic2story', 'positive', 'positive_l', 'positive_r', 'negative', 'negative_l', 'negative_r', 'prompt_state', 'decoded_positive', 'decoded_negative', 'pic2story_positive',
                      'model', 'model_concept', 'concept_data', 'model_version', 'is_sdxl', 'model_hash', 'vae', 'vae_hash', 'vae_name_sd', 'vae_name_sdxl', 'sampler', 'scheduler', 'steps',
                      'cfg', 'seed', 'width', 'height', 'size_string', 'preferred', 'saved_image_width', 'saved_image_heigth', 'upscaler_ratio',
                      'vae_name_sd', 'vae_name_sdxl']

def merge_str_to_tuple(item1, item2):
    if not isinstance(item1, tuple):
        item1 = (item1,)
    if not isinstance(item2, tuple):
        item2 = (item2,)
    return item1 + item2

def merge_dict(dict1, dict2):
    dict3 = dict1.copy()
    for k, v in dict2.items():
        dict3[k] = merge_str_to_tuple(v, dict3[k]) if k in dict3 else v
    return dict3

def remove_quotes(string):
    return str(string).replace('"', "").replace("'", "")

def add_quotes(string):
    return '"' + str(string) + '"'

def get_square_shape(shape_a, shape_b):
    area = shape_a * shape_b
    square = math.sqrt(area)
    standard_square = min(VALID_SHAPES, key=lambda x: abs(square - x))
    return standard_square

def get_dimensions_by_shape(self, rationame: str, square: int, orientation:str = 'Vertical', round_to_standard: bool = False, calculate_by_custom: bool = False, custom_side_a: float = 1, custom_side_b: float = 1, standard:str = 'STANDARD'):
    def calculate_dim(ratio_1: float, ratio_2: float, square: int):
        FullPixels = square ** 2
        ratio = ratio_2 / ratio_1
        side_a = math.sqrt(FullPixels * ratio)
        side_b = side_a /  ratio

        if round_to_standard == True:
            STANDARD_LIST = STANDARD_SIDES
            if standard == 'CASCADE':
                STANDARD_LIST = CASCADE_SIDES
            side_a = min(STANDARD_LIST, key=lambda x: abs(side_a - x))
            side_b = round(FullPixels / side_a)
            side_b = min(STANDARD_LIST, key=lambda x: abs(x - side_b))

        side_a = round(side_a)
        side_b = round(side_b)

        return sorted([side_a, side_b], reverse=True)

    if (calculate_by_custom == True and isinstance(custom_side_a, (int, float)) and isinstance(custom_side_b, (int, float)) and custom_side_a >= 1 and custom_side_b >= 1):
        ratio_x = custom_side_a
        ratio_y = custom_side_b
    else:
        RatioLabel = self.ratioNames[rationame]
        ratio_x = self.sd_ratios[RatioLabel]['side_x']
        ratio_y = self.sd_ratios[RatioLabel]['side_y']

    dimensions = calculate_dim(ratio_x, ratio_y, square)
    if (orientation == 'Vertical'):
        dimensions = sorted(dimensions)

    return dimensions

def clear_prompt(NETWORK_START, NETWORK_END, promptstring):
    promptstring_temp = promptstring

    for LABEL in NETWORK_START:
        if LABEL in promptstring:
            LabelStartIndexes = [n for n in range(len(promptstring)) if promptstring.find(LABEL, n) == n]

            for LabelStartIndex in LabelStartIndexes:
                Matches = []
                for endString in NETWORK_END:
                    Match = promptstring.find(endString, (LabelStartIndex + 1))
                    if (Match > 0):
                        Matches.append(Match)

                LabelEndIndex = sorted(Matches)[0]
                MatchedString = promptstring[LabelStartIndex:(LabelEndIndex + 1)]
                if len(MatchedString) > 0:
                    if '<' in MatchedString:
                        endString = '>'
                        Match = promptstring.find(endString, (LabelStartIndex + 1))
                        if (Match > 0):
                            LabelEndIndex = Match
                            MatchedString = promptstring[LabelStartIndex:(LabelEndIndex + 1)]
                            promptstring_temp = promptstring_temp.replace(MatchedString, "")

                    if '{' in MatchedString:
                        endString = '}'
                        Match = promptstring.find(endString, (LabelStartIndex + 1))
                        if (Match > 0):
                            LabelEndIndex = Match
                            MatchedString = promptstring[LabelStartIndex:(LabelEndIndex + 1)]
                            promptstring_temp = promptstring_temp.replace(MatchedString, "")

                    if ')' in MatchedString:
                        MatchedString = promptstring[(LabelStartIndex - 1):(LabelEndIndex + 1)]
                        promptstring_temp = promptstring_temp.replace(MatchedString, "")

                    promptstring_temp = promptstring_temp.replace(MatchedString, "")

    return promptstring_temp.replace('()', '').replace(' , ,', ',').replace('||', '').replace('{,', '').replace('  ', ' ').replace(', ,', ',').strip(', ')

def clear_cascade(prompt):
    return re.sub("(\.d+)|(:d+)|[()]|BREAK|break", "", prompt).replace('  ', ' ')

def clear_hunyuan(prompt, length = 0):
    cleanPrompt = re.sub("(\.\d+)|(:\d+)|[()]|BREAK|break", "", prompt).replace('  ', ' ')
    if length > 0:
        cleanPrompt = cleanPrompt[:length].rsplit(' ', 1)[0]
    return cleanPrompt

def get_networks_prompt(NETWORK_START, NETWORK_END, promptstring):
    valid_networks = []

    for LABEL in NETWORK_START:
        if LABEL in promptstring:
            LabelStartIndexes = [n for n in range(len(promptstring)) if promptstring.find(LABEL, n) == n]

            for LabelStartIndex in LabelStartIndexes:
                Matches = []
                for endString in NETWORK_END:
                    Match = promptstring.find(endString, (LabelStartIndex + 1))
                    if (Match > 0):
                        Matches.append(Match)

                LabelEndIndex = sorted(Matches)[0]
                MatchedString = promptstring[(LabelStartIndex + len(LABEL)):(LabelEndIndex)]
                if len(MatchedString) > 0:
                    networkdata = MatchedString.split(":")
                    if len(networkdata) == 1:
                        networkdata.append('1')
                    if LABEL == '<lora:':
                        networkdata.append('LORA')
                    if LABEL == '<lyco:':
                        networkdata.append('LYCORIS')
                    if LABEL == '<hypernet:':
                        networkdata.append('HYPERNET')

                    valid_networks.append(networkdata)

    return valid_networks

class ModelSamplingDiscreteLCM(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.sigma_data = 1.0
        timesteps = 1000
        beta_start = 0.00085
        beta_end = 0.012

        betas = torch.linspace(beta_start**0.5, beta_end**0.5, timesteps, dtype=torch.float32) ** 2
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)

        original_timesteps = 50
        self.skip_steps = timesteps // original_timesteps

        alphas_cumprod_valid = torch.zeros((original_timesteps), dtype=torch.float32)
        for x in range(original_timesteps):
            alphas_cumprod_valid[original_timesteps - 1 - x] = alphas_cumprod[timesteps - 1 - x * self.skip_steps]

        sigmas = ((1 - alphas_cumprod_valid) / alphas_cumprod_valid) ** 0.5
        self.set_sigmas(sigmas)

    def set_sigmas(self, sigmas):
        self.register_buffer('sigmas', sigmas)
        self.register_buffer('log_sigmas', sigmas.log())

    @property
    def sigma_min(self):
        return self.sigmas[0]

    @property
    def sigma_max(self):
        return self.sigmas[-1]

    def timestep(self, sigma):
        log_sigma = sigma.log()
        dists = log_sigma.to(self.log_sigmas.device) - self.log_sigmas[:, None]
        return dists.abs().argmin(dim=0).view(sigma.shape) * self.skip_steps + (self.skip_steps - 1)

    def sigma(self, timestep):
        t = torch.clamp(((timestep - (self.skip_steps - 1)) / self.skip_steps).float(), min=0, max=(len(self.sigmas) - 1))
        low_idx = t.floor().long()
        high_idx = t.ceil().long()
        w = t.frac()
        log_sigma = (1 - w) * self.log_sigmas[low_idx] + w * self.log_sigmas[high_idx]
        return log_sigma.exp()

    def percent_to_sigma(self, percent):
        return self.sigma(torch.tensor(percent * 999.0))

def DynPromptDecoder(self, dyn_prompt, seed):
    prompt_generator = RandomPromptGenerator(
        self._wildcard_manager,
        seed = seed,
        parser_config = self._parser_config,
        unlink_seed_from_prompt = False,
        ignore_whitespace = False
    )

    dyn_type = type(dyn_prompt).__name__
    if (dyn_type != 'str'):
        dyn_prompt = ''

    try:
        all_prompts = prompt_generator.generate(dyn_prompt, 1) or [""]
    except Exception:
        all_prompts = [""]

    prompt = all_prompts[0]
    return prompt

'''def ModelObjectParser(modelobject):
    for key in modelobject:
        Suboject_1 = modelobject[key]
        Suboject_2 = Suboject_1._modules
        for key1 in Suboject_2:
            sub_2_typename = type(Suboject_2[key1]).__name__
            if sub_2_typename == 'SpatialTransformer':
                VersionObject = Suboject_2[key1]._modules['transformer_blocks']._modules['0']._modules['attn2']._modules['to_k'].in_features

                if VersionObject <= 768:
                    VersionObject = 768
                if 1024 >= VersionObject > 768:
                    VersionObject = 1024
                if VersionObject > 1024:
                    VersionObject = 2048

                return VersionObject
def getCheckpointVersion(modelobject):
    ckpt_type = type(modelobject.__dict__['model']).__name__
    try:
        ModelVersion = ModelObjectParser(modelobject.model._modules['diffusion_model']._modules['input_blocks']._modules)
    except:
        ModelVersion = 1024

    return ckpt_type + '_' + str(ModelVersion)'''

def getResolutionByType(model_type):
    for res_key in CONCEPT_RESOLUTIONS:
        if model_type in CONCEPT_RESOLUTIONS[res_key]:
            return int(res_key)

def getModelType(base_model, model_type):
    LYCO_DIR = os.path.join(comfy_dir, 'models', 'lycoris')
    folder_paths.add_model_folder_path("lycoris", LYCO_DIR)

    ckpt_path = folder_paths.get_full_path(model_type, base_model)
    model_version = None

    if ckpt_path is not None:
        if model_type != 'checkpoints':
            try:
                safetensors_header = comfy.utils.safetensors_header(ckpt_path)
                if safetensors_header is not None:
                    header_json = json.loads(safetensors_header)
                    if (model_type == 'loras' or model_type == 'lycoris') and '__metadata__' in header_json and 'ss_base_model_version' in header_json['__metadata__']:
                        model_version = header_json['__metadata__']['ss_base_model_version']

                    elif (model_type == 'loras' or model_type == 'lycoris') and '__metadata__' in header_json and 'ss_resolution' in header_json['__metadata__']:
                        model_version = header_json['__metadata__']['ss_resolution']
                        if model_version is not None and model_version != 'NoneType':
                            lora_resolution = (model_version.replace("(", "").replace(")", "").split(", "))
                            if len(lora_resolution) == 2:
                                model_version_res = (int(lora_resolution[0]) * int(lora_resolution[1]))
                                if model_version_res < 1000 * 1000:
                                    model_version = 'SD1'
                                else:
                                    model_version = 'SDXL'

                    elif (model_type == 'loras' or model_type == 'lycoris') and '__metadata__' in header_json and 'ss_datasets' in header_json['__metadata__']:
                        dataset = header_json['__metadata__']['ss_datasets']
                        header_json_dataset = json.loads(dataset)
                        if len(header_json_dataset) > 0 and 'resolution' in header_json_dataset[0]:
                            dataset_resolution = header_json_dataset[0]['resolution']
                            if len(dataset_resolution) == 2:
                                model_version_res = (int(dataset_resolution[0]) * int(dataset_resolution[1]))
                                if model_version_res < 1000 * 1000:
                                    model_version = 'SD1'
                                else:
                                    model_version = 'SDXL'

                    else:
                        if '__metadata__' in header_json and 'modelspec.architecture' in header_json['__metadata__']:
                            model_version = header_json['__metadata__']['modelspec.architecture']
                else:
                    return False
            except:
                try:
                    sd = comfy.utils.load_torch_file(ckpt_path)
                    diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
                    model_config = model_detection.model_config_from_unet(sd, diffusion_model_prefix)
                    model_version = type(model_config).__name__
                except:
                    return False
        else:
            try:
                sd = comfy.utils.load_torch_file(ckpt_path)
                diffusion_model_prefix = model_detection.unet_prefix_from_state_dict(sd)
                model_config = model_detection.model_config_from_unet(sd, diffusion_model_prefix)
                model_version = type(model_config).__name__
            except:
                return False

    else:
        return False

    if model_version is not None and model_version != 'NoneType' and (model_type == 'checkpoints' or model_type == 'loras' or model_type == 'lycoris'):
        model_version = model_version.replace("_", "").replace(".", "").replace("sdv1", "SD1")
        res = [ele for ele in SUPPORTED_MODELS if (ele.lower() in model_version.lower())]
        if len(res) > 0:
            model_version = res[0]

    return model_version

def get_model_hash(filename):
    try:
        with open(filename, "rb") as file:
            m = hashlib.sha256()
            file.seek(0x100000)
            m.update(file.read(0x10000))
            hash = m.hexdigest()[0:8]
            return hash
    except FileNotFoundError:
        return None

def load_external_csv(csv_full_path: str, header_cols: int):
    fileTest = open(csv_full_path, 'rb').readline()
    result = chardet.detect(fileTest)
    ENCODING = result['encoding']
    if ENCODING == 'ascii':
        ENCODING = 'UTF-8'

    with open(csv_full_path, "r", newline = '', encoding = ENCODING) as csv_file:
        try:
            return pandas.read_csv(csv_file, header = header_cols, index_col = False, skipinitialspace = True)
        except pandas.errors.ParserError as e:
            errorstring = repr(e)
            matchre = re.compile('Expected (d+) fields in line (d+), saw (d+)')
            (expected, line, saw) = map(int, matchre.search(errorstring).groups())
            print(f'Error at line {line}. Fields added : {saw - expected}.')
            return None

def get_model_keywords(filename, modelhash, model_name):
    keywords = load_external_csv(filename, 3)
    if keywords is not None:
        selected_kv = keywords[keywords['#model_hash'] == modelhash]['keyword'].values
        if (len(selected_kv) > 1):
            selected_ckpt = keywords[keywords['#model_hash'] == modelhash]['filename.ckpt'].values
            basename = Path(model_name).stem

            cutoff_list = list(np.around(np.arange(0.1, 1.05, 0.05).tolist(), 2))[::-1]
            is_found = []
            model_name_kw = None

            for trycut in cutoff_list:
                is_found = difflib.get_close_matches(basename, selected_ckpt, cutoff=trycut)
                if len(is_found) >= 1:
                    model_name_kw = is_found[0]
                    break

            if len(is_found) >= 0:
                if model_name_kw is not None:
                    selected_kv = keywords[keywords['filename.ckpt'] == model_name_kw]['keyword'].values

        if (len(selected_kv) > 0):
            return selected_kv[0]
        else:
            return None
    else:
        return None

def get_closest_element(value, netlist):
    cutoff_list = list(np.around(np.arange(0.1, 1.05, 0.05).tolist(), 2))[::-1]
    is_found = None

    for trycut in cutoff_list:
        is_found = difflib.get_close_matches(value, netlist, cutoff=trycut)
        if len(is_found) >= 1:
            return is_found[0]

    return is_found

def get_category_from_cache(category):
    ifCacheExist = os.path.isfile(cache_file)
    if ifCacheExist == True:
        with open(cache_file, 'r') as openfile:
            try:
                saved_cache = json.load(openfile)
                try:
                    return saved_cache[category]
                except Exception:
                    return None
            except ValueError as e:
                return None
    else:
        return None
def get_value_from_cache(category, key):
    ifCacheExist = os.path.isfile(cache_file)
    if ifCacheExist == True:
        with open(cache_file, 'r') as openfile:
            try:
                saved_cache = json.load(openfile)
                try:
                    return saved_cache[category][key]
                except Exception:
                    return None
            except ValueError as e:
                return None
    else:
        return None

def update_value_in_cache(category, key, value):
    cacheData = {category: {key: value}}
    json_object = json.dumps(cacheData, indent=4)
    ifCacheExist = os.path.isfile(cache_file)
    if ifCacheExist == True:
        with open(cache_file, 'r') as openfile:
            try:
                saved_cache = json.load(openfile)
                if category in saved_cache and key in saved_cache[category]:
                    saved_cache[category][key] = value
                else:
                    saved_cache.update(cacheData)

                newJsonObject = json.dumps(saved_cache, indent=4)
                with open(cache_file, "w", encoding='utf-8') as outfile:
                    outfile.write(newJsonObject)
                return True

            except ValueError as e:
                return None
    else:
        with open(cache_file, "w", encoding='utf-8') as outfile:
            outfile.write(json_object)
        return True

def add_value_to_cache(category, key, value):
    cacheData = {category: {key: value}}
    json_object = json.dumps(cacheData, indent=4)
    ifCacheExist = os.path.isfile(cache_file)

    if ifCacheExist == True:
        with open(cache_file, 'r') as openfile:
            try:
                saved_cache = json.load(openfile)
                if category in saved_cache:
                    saved_cache[category][key] = value
                else:
                    saved_cache.update(cacheData)

                newJsonObject = json.dumps(saved_cache, indent=4)
                with open(cache_file, "w", encoding='utf-8') as outfile:
                    outfile.write(newJsonObject)
                return True

            except ValueError as e:
                return False
    else:
        with open(cache_file, "w", encoding='utf-8') as outfile:
            outfile.write(json_object)
        return True

def getLoraVersion(modelobject):
    VersionKeysBlock = [
        'lora_unet_down_blocks_0_attentions_0_proj_in.lora_up.weight',
        'lora_unet_down_blocks_2_attentions_0_transformer_blocks_0_attn1_to_k.lora_up.weight',
        'lora_te_text_model_encoder_layers_0_mlp_fc1.lora_up.weight',
        'lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_ff_net_0_proj.lora_up.weight',
        'lora_unet_input_blocks_4_1_transformer_blocks_0_ff_net_0_proj.lora_up.weight',
        'lora_unet_input_blocks_4_1_transformer_blocks_0_ff_net_0_proj.hada_w1_a',
        'lora_unet_down_blocks_1_attentions_0_transformer_blocks_0_ff_net_0_proj.hada_w1_a',
    ]
    VersionHelper = 'Unknown'

    for index, value in modelobject.items():
        try:
            type(value.item()).__name__
        except Exception:
            if index in VersionKeysBlock:
                VersionHelper = len(value)
                break

    return VersionHelper

def pil2numpy(image: Image.Image):
    return np.array(image).astype(np.float32) / 255.0

def numpy2pil(image: np.ndarray, mode=None):
    return Image.fromarray(np.clip(255.0 * image, 0, 255).astype(np.uint8), mode)

def pil2tensor(image: Image.Image):
    return torch.from_numpy(pil2numpy(image)).unsqueeze(0)

def tensor2pil(image: torch.Tensor, mode=None):
    return numpy2pil(image.cpu().numpy().squeeze(), mode=mode)

def image_scale_down(images, width, height, crop):
    if crop == "center":
        old_width = images.shape[2]
        old_height = images.shape[1]
        old_aspect = old_width / old_height
        new_aspect = width / height
        x = 0
        y = 0
        if old_aspect > new_aspect:
            x = round((old_width - old_width * (new_aspect / old_aspect)) / 2)
        elif old_aspect < new_aspect:
            y = round((old_height - old_height * (old_aspect / new_aspect)) / 2)
        s = images[:, y : old_height - y, x : old_width - x, :]
    else:
        s = images

    results = []
    for image in s:
        img = tensor2pil(image).convert("RGB")
        img = img.resize((width, height), Image.LANCZOS)
        results.append(pil2tensor(img))

    return (torch.cat(results, dim=0),)

def image_scale_down_by_scale(images, scale_by):
    width = images.shape[2]
    height = images.shape[1]
    new_width = int(width * scale_by)
    new_height = int(height * scale_by)
    return image_scale_down(images, new_width, new_height, "center")

def image_scale_down_by_dim(images, new_width, new_height):
    return image_scale_down(images, new_width, new_height, "center")

def img_resizer(image: torch.Tensor, width: int, height: int, interpolation_mode: str):
    assert isinstance(image, torch.Tensor)
    assert isinstance(height, int)
    assert isinstance(width, int)
    assert isinstance(interpolation_mode, str)

    interpolation_mode = interpolation_mode.upper().replace(" ", "_")
    interpolation_mode = getattr(InterpolationMode, interpolation_mode)

    image = image.permute(0, 3, 1, 2)
    image = F.resize(image, (height, width), interpolation=interpolation_mode, antialias=True)
    image = image.permute(0, 2, 3, 1)
    return image

def apply_variation_noise(latent_image, noise_device, variation_seed, variation_strength, mask=None):
    latent_size = latent_image.size()
    latent_size_1batch = [1, latent_size[1], latent_size[2], latent_size[3]]

    if noise_device == "cpu":
        variation_generator = torch.manual_seed(variation_seed)
    else:
        torch.cuda.manual_seed(variation_seed)
        variation_generator = None

    variation_latent = torch.randn(latent_size_1batch, dtype=latent_image.dtype, layout=latent_image.layout, generator=variation_generator, device=noise_device)
    variation_noise = variation_latent.expand(latent_image.size()[0], -1, -1, -1)

    if variation_strength == 0:
        return latent_image
    elif mask is None:
        result = (1 - variation_strength) * latent_image + variation_strength * variation_noise
    else:
        # this seems precision is not enough when variation_strength is 0.0
        result = (mask == 1).float() * ((1 - variation_strength) * latent_image + variation_strength * variation_noise * mask) + (mask == 0).float() * latent_image

    return result

def prepare_noise(latent_image, seed, noise_inds=None, noise_device="cpu", incremental_seed_mode="comfy", variation_seed=None, variation_strength=None):
    latent_size = latent_image.size()
    latent_size_1batch = [1, latent_size[1], latent_size[2], latent_size[3]]

    if variation_strength is not None and variation_strength > 0 or incremental_seed_mode.startswith("variation str inc"):
        if noise_device == "cpu":
            variation_generator = torch.manual_seed(variation_seed)
        else:
            torch.cuda.manual_seed(variation_seed)
            variation_generator = None

        variation_latent = torch.randn(latent_size_1batch, dtype=latent_image.dtype, layout=latent_image.layout, generator=variation_generator, device=noise_device)
    else:
        variation_latent = None

    def apply_variation(input_latent, strength_up=None):
        if variation_latent is None:
            return input_latent
        else:
            strength = variation_strength

            if strength_up is not None:
                strength += strength_up

            variation_noise = variation_latent.expand(input_latent.size()[0], -1, -1, -1)
            result = (1 - strength) * input_latent + strength * variation_noise
            return result

    # method: incremental seed batch noise
    if noise_inds is None and incremental_seed_mode == "incremental":
        batch_cnt = latent_size[0]

        latents = None
        for i in range(batch_cnt):
            if noise_device == "cpu":
                generator = torch.manual_seed(seed+i)
            else:
                torch.cuda.manual_seed(seed+i)
                generator = None

            latent = torch.randn(latent_size_1batch, dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device=noise_device)
            latent = apply_variation(latent)

            if latents is None:
                latents = latent
            else:
                latents = torch.cat((latents, latent), dim=0)

        return latents

    # method: incremental variation batch noise
    elif noise_inds is None and incremental_seed_mode.startswith("variation str inc"):
        batch_cnt = latent_size[0]

        latents = None
        for i in range(batch_cnt):
            if noise_device == "cpu":
                generator = torch.manual_seed(seed)
            else:
                torch.cuda.manual_seed(seed)
                generator = None

            latent = torch.randn(latent_size_1batch, dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device=noise_device)

            step = float(incremental_seed_mode[18:])
            latent = apply_variation(latent, step*i)

            if latents is None:
                latents = latent
            else:
                latents = torch.cat((latents, latent), dim=0)

        return latents

    # method: comfy batch noise
    if noise_device == "cpu":
        generator = torch.manual_seed(seed)
    else:
        torch.cuda.manual_seed(seed)
        generator = None

    if noise_inds is None:
        latents = torch.randn(latent_image.size(), dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device=noise_device)
        latents = apply_variation(latents)
        return latents

    unique_inds, inverse = np.unique(noise_inds, return_inverse=True)
    noises = []
    for i in range(unique_inds[-1] + 1):
        noise = torch.randn([1] + list(latent_image.size())[1:], dtype=latent_image.dtype, layout=latent_image.layout, generator=generator, device=noise_device)
        if i in unique_inds:
            noises.append(noise)

    noises = [noises[i] for i in inverse]
    noises = torch.cat(noises, axis=0)
    return noises


def hf_downloader(repo_id, model_local_dir):
    from huggingface_hub import snapshot_download
    model_path = f"{model_local_dir}/{repo_id.split('/')[-1]}"
    snapshot_download(repo_id=repo_id, local_dir=model_path, local_dir_use_symlinks=True, max_workers=1)
    return model_path

def ModelConceptNames(ckpt_name, model_concept, lightning_selector, lightning_model_step, hypersd_selector, hypersd_model_step, model_version = 'SDXL'):
    lora_name = None
    unet_name = None
    lightningModeValid = False
    hyperModeValid = False

    LoraList = getDownloadedFiles()

    if model_concept == 'Lightning':
        if lightning_selector == 'SAFETENSOR':
            allCheckpoints = folder_paths.get_filename_list("checkpoints")
            allLightning = list(filter(lambda a: 'sdxl_lightning_'.casefold() in a.casefold(), allCheckpoints))
            if len(allLightning) > 0:
                finalLightning = list(filter(lambda a: str(lightning_model_step) + 'step'.casefold() in a.casefold(), allLightning))
                if len(finalLightning) > 0:
                    lightningModeValid = True
                    ckpt_name = finalLightning[0]

        if lightning_selector == 'LORA':
            # LoraList = folder_paths.get_filename_list("loras")
            if len(LoraList) > 0:
                allLoraLightning = list(filter(lambda a: 'sdxl_lightning_'.casefold() in a.casefold(), LoraList))
                if len(allLoraLightning) > 0:
                    finalLightning = list(filter(lambda a: str(lightning_model_step) + 'step'.casefold() in a.casefold(), allLoraLightning))
                    if len(finalLightning) > 0:
                        lightningModeValid = True
                        lora_name = finalLightning[0]

        if lightning_selector == 'UNET':
            UnetList = folder_paths.get_filename_list("unet")
            if len(UnetList) > 0:
                allUnetLightning = list(filter(lambda a: 'sdxl_lightning_'.casefold() in a.casefold(), UnetList))
                if len(allUnetLightning) > 0:
                    finalLightning = list(filter(lambda a: str(lightning_model_step) + 'step'.casefold() in a.casefold(), allUnetLightning))
                    if len(finalLightning) > 0:
                        lightningModeValid = True
                        unet_name = finalLightning[0]

    if model_concept == 'Hyper':
        print('Hyper lora check:')
        if hypersd_selector == 'LORA':
            if len(LoraList) > 0:
                if model_version == 'SDXL':
                    allLoraHyper = list(filter(lambda a: 'Hyper-SDXL-'.casefold() in a.casefold(), LoraList))
                else:
                    allLoraHyper = list(filter(lambda a: 'Hyper-SD15-'.casefold() in a.casefold(), LoraList))
                if len(allLoraHyper) > 0:
                    pluralString = ''
                    if hypersd_model_step > 1:
                        pluralString = 's'

                    finalHyper = list(filter(lambda a: str(hypersd_model_step) + 'step' + pluralString + '-lora'.casefold() in a.casefold() or str(hypersd_model_step) + 'step' + pluralString + '-CFG-lora'.casefold() in a.casefold(), allLoraHyper))
                    if len(finalHyper) > 0:
                        hyperModeValid = True
                        lora_name = finalHyper[0]

        if hypersd_selector == 'UNET':
            UnetList = folder_paths.get_filename_list("unet")
            if len(UnetList) > 0:
                allUnetHyper = list(filter(lambda a: 'Hyper-SDXL-1step-Unet-Comfyui'.casefold() in a.casefold(), UnetList))
                if len(allUnetHyper) > 0:
                    hyperModeValid = True
                    unet_name = allUnetHyper[0]

    return {'ckpt_name': ckpt_name, 'lora_name': lora_name, 'unet_name': unet_name, 'lightningModeValid': lightningModeValid, 'hyperModeValid': hyperModeValid}

def BDanceConceptHelper(self, model_concept, lightningModeValid, lightning_selector, lightning_model_step, OUTPUT_MODEL, lora_name, unet_name, ckpt_name, lora_model_strength = 1):
    if model_concept == 'Lightning' and lightningModeValid == True and lightning_selector == 'LORA' and lora_name is not None:
        if lora_model_strength != 0:
            lora = None
            if self.loaded_lora is not None:
                if self.loaded_lora[0] == lora_name:
                    lora = self.loaded_lora[1]
                else:
                    temp = self.loaded_lora
                    self.loaded_lora = None
                    del temp

            if lora is None:
                lora = comfy.utils.load_torch_file(lora_name, safe_load=True)
                self.loaded_lora = (lora_name, lora)

            print(lora_name)
            OUTPUT_MODEL = comfy.sd.load_lora_for_models(OUTPUT_MODEL, None, lora, lora_model_strength, 0)[0]

    if model_concept == 'Lightning' and lightningModeValid == True and lightning_selector == 'UNET' and unet_name is not None:
        OUTPUT_MODEL = nodes.UNETLoader.load_unet(self, unet_name, "default")[0]

    if model_concept == 'Lightning' and lightningModeValid == True and lightning_selector == 'SAFETENSOR' and ckpt_name is not None:
        OUTPUT_MODEL = nodes.CheckpointLoaderSimple.load_checkpoint(self, ckpt_name)[0]

    if model_concept == 'Lightning' and lightning_model_step == 1 and lightningModeValid == True:
        OUTPUT_MODEL = nodes_model_advanced.ModelSamplingDiscrete.patch(self, OUTPUT_MODEL, "x0", False)[0]

    if model_concept == 'Hyper' and lightningModeValid == True and lightning_selector == 'LORA' and lora_name is not None:
        if lora_model_strength != 0:
            lora = None
            if self.loaded_lora is not None:
                if self.loaded_lora[0] == lora_name:
                    lora = self.loaded_lora[1]
                else:
                    temp = self.loaded_lora
                    self.loaded_lora = None
                    del temp

            if lora is None:
                lora = comfy.utils.load_torch_file(lora_name, safe_load=True)
                self.loaded_lora = (lora_name, lora)

            print(lora_name)
            OUTPUT_MODEL = comfy.sd.load_lora_for_models(OUTPUT_MODEL, None, lora, lora_model_strength, 0)[0]

    if model_concept == 'Hyper' and lightningModeValid == True and lightning_selector == 'UNET' and unet_name is not None:
        unet_path = folder_paths.get_full_path("unet", unet_name)
        OUTPUT_MODEL = comfy.sd.load_checkpoint_guess_config(unet_path)

    return OUTPUT_MODEL

def get_hypersd_sigmas(model):
    timesteps = torch.tensor([800])
    sigmas = model.model.model_sampling.sigma(timesteps)
    sigmas = torch.cat([sigmas, sigmas.new_zeros([1])])
    return (sigmas,)

class ModelSamplingDiscreteDistilledTCD(ModelSamplingDiscreteDistilled, EPS):
    def __init__(self, model_config=None):
        super().__init__(model_config)
        sampling_settings = model_config.sampling_settings if model_config is not None else {}

        beta_schedule = sampling_settings.get("beta_schedule", "linear")
        linear_start = sampling_settings.get("linear_start", 0.00085)
        linear_end = sampling_settings.get("linear_end", 0.012)

        betas = make_beta_schedule(
            beta_schedule, n_timestep=1000, linear_start=linear_start, linear_end=linear_end, cosine_s=8e-3
        )
        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0, dtype=torch.float32)
        self.register_buffer("alphas_cumprod", alphas_cumprod.clone().detach())

@torch.no_grad()
def sample_tcd(model, x, sigmas, extra_args=None, callback=None, disable=None, noise_sampler=None, eta=0.3, alpha_prod_s: torch.Tensor = None):
    extra_args = {} if extra_args is None else extra_args
    noise_sampler = default_noise_sampler(x) if noise_sampler is None else noise_sampler
    s_in = x.new_ones([x.shape[0]])
    beta_prod_s = 1 - alpha_prod_s
    for i in trange(len(sigmas) - 1, disable=disable):
        denoised = model(x, sigmas[i] * s_in, **extra_args)
        eps = (x - denoised) / sigmas[i]
        denoised = alpha_prod_s[i + 1].sqrt() * denoised + beta_prod_s[i + 1].sqrt() * eps
        if callback is not None:
            callback({"x": x, "i": i, "sigma": sigmas[i], "sigma_hat": sigmas[i], "denoised": denoised})
        x = denoised
        if eta > 0 and sigmas[i + 1] > 0:
            noise = noise_sampler(sigmas[i], sigmas[i + 1])
            x = x / alpha_prod_s[i+1].sqrt() + noise * (sigmas[i+1]**2 + 1 - 1/alpha_prod_s[i+1]).sqrt()
    return x
def TCDModelSamplingDiscrete(self, model, steps=4, scheduler="simple", denoise=1.0, eta=0.3):
    m = model.clone()
    ms = ModelSamplingDiscreteDistilledTCD(model.model.model_config)

    total_steps = steps
    if denoise <= 0.0:
        # raise error ?
        sigmas = torch.FloatTensor([])
    elif denoise <= 1.0:
        total_steps = int(steps / denoise)
        sigmas = calculate_sigmas(ms, scheduler, total_steps).cpu()
        sigmas = sigmas[-(steps + 1) :]
    m.add_object_patch("model_sampling", ms)

    timesteps_s = torch.floor((1 - eta) * ms.timestep(sigmas)).to(dtype=torch.long).detach()
    timesteps_s[-1] = 0
    alpha_prod_s = ms.alphas_cumprod[timesteps_s]
    sampler = KSAMPLER(sample_tcd, extra_options={"eta": eta, "alpha_prod_s": alpha_prod_s}, inpaint_options={})
    return (m, sampler, sigmas)

def getLatentSize(samples):
    for tensor in samples['samples'][0]:
        if isinstance(tensor, torch.Tensor):
            shape = tensor.shape
            tensor_height = shape[-2]
            tensor_width = shape[-1]
            return (tensor_width, tensor_height)
        else:
            return (None, None)

    return (None, None)

def MatchDimensions(width_1, height_1, width_2, height_2, axis_value):
    if axis_value == 1:
        rate = height_2 / height_1
        new_heigth = height_1
        new_width = width_2 / rate
    else:
        rate = width_2 / width_1
        new_width = width_1
        new_heigth = height_2 / rate

    return [round(new_width), round(new_heigth)]

def ImageConcat(image1, image2, axis_value):
    image1_size = image1.size
    image2_size = image2.size
    new_image_dim = MatchDimensions(image1_size[0], image1_size[1], image2_size[0], image2_size[1], axis_value)
    img2_resized = image2.resize(new_image_dim)

    if axis_value == 1:
        new_image = Image.new('RGB', (new_image_dim[0] + image1_size[0], image1_size[1]), (250, 250, 250))
        new_image.paste(image1, (0, 0))
        new_image.paste(img2_resized, (image1_size[0], 0))
    else:
        new_image = Image.new('RGB', (image1_size[0], new_image_dim[1] + image1_size[1]), (250, 250, 250))
        new_image.paste(image1, (0, 0))
        new_image.paste(img2_resized, (0, image1_size[1]))

    return new_image

def getDataFromWorkflowById(workflow, nodeName, dataIndex):
    result = None

    for NODE_ITEMS in workflow:
        if 'type' in NODE_ITEMS:
            ITEM_TYPE = NODE_ITEMS['type']
            if ITEM_TYPE == nodeName:
                if 'widgets_values' in NODE_ITEMS:
                    ITEM_VALUES = NODE_ITEMS['widgets_values']
                    if len(ITEM_VALUES) >= dataIndex + 1:
                        result = ITEM_VALUES[dataIndex]

    return result

def getDataFromWorkflowByName(workflow, nodeName, inputName, prompt):
    results = None

    for node in workflow:
        node_id = None
        name = node["type"]
        if "properties" in node:
            if "Node name for S&R" in node["properties"]:
                name = node["properties"]["Node name for S&R"]
        if name == nodeName:
            node_id = node["id"]
        else:
            if "title" in node:
                name = node["title"]
            if name == nodeName:
                node_id = node["id"]
        if node_id is None:
            continue
        if str(node_id) in prompt:
            values = prompt[str(node_id)]
            if "inputs" in values and inputName in values["inputs"]:
                v = values["inputs"][inputName]
                return v

    return results

def collect_state(extra_pnginfo, prompt):
    workflow = extra_pnginfo["workflow"]
    results = {}
    if "links" in workflow:
        results["__links"] = workflow["links"]
    for node in workflow["nodes"]:
        node_id = str(node["id"])
        name = node["type"]
        if "Debug" in name or "Show" in name or "Function" in name or "Evaluate" in name:
            continue

        if "widgets_values" in node and "inputs" not in node:
            results[node_id] = node["widgets_values"]
        elif node_id in prompt:
            values = prompt[node_id]
            if "inputs" in values:
                results[node_id] = {}
                for widget in values["inputs"].items():
                    (n, v) = widget
                    if type(v) is not str and isinstance(v, collections.abc.Sequence):
                        continue
                    results[node_id][n] = v
        elif "widgets_values" in node:
            results[node_id] = node["widgets_values"]

    result = json.dumps(results, sort_keys=True)
    return hashlib.sha256(result.encode()).hexdigest()

class MLP(pl.LightningModule):
    def __init__(self, input_size, xcol='emb', ycol='avg_rating'):
        super().__init__()
        self.input_size = input_size
        self.xcol = xcol
        self.ycol = ycol
        self.layers = nn.Sequential(
            nn.Linear(self.input_size, 1024),
            nn.Dropout(0.2),
            nn.Linear(1024, 128),
            nn.Dropout(0.2),
            nn.Linear(128, 64),
            nn.Dropout(0.1),
            nn.Linear(64, 16),
            nn.Linear(16, 1)
        )

    def forward(self, x):
        return self.layers(x)
    def training_step(self, batch, batch_idx):
            x = batch[self.xcol]
            y = batch[self.ycol].reshape(-1, 1)
            x_hat = self.layers(x)
            loss = F.mse_loss(x_hat, y)
            return loss
    def validation_step(self, batch, batch_idx):
        x = batch[self.xcol]
        y = batch[self.ycol].reshape(-1, 1)
        x_hat = self.layers(x)
        loss = F.mse_loss(x_hat, y)
        return loss
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-3)
        return optimizer
def normalized(a, axis=-1, order=2):
    import numpy as np  # pylint: disable=import-outside-toplevel
    l2 = np.atleast_1d(np.linalg.norm(a, order, axis))
    l2[l2 == 0] = 1
    return a / np.expand_dims(l2, axis)

def ImageLoaderFromPath(ImgPath, new_width = None, new_height = None):
    output_image = None

    if Path(ImgPath).is_file() == True:
        loaded_img = Image.open(ImgPath)
        output_images = []
        for i in ImageSequence.Iterator(loaded_img):
            i = ImageOps.exif_transpose(i)
            if i.mode == 'I':
                i = i.point(lambda i: i * (1 / 255))
            image = i.convert("RGB")
            if new_width is not None and new_height is not None:
                newsize = (new_width, new_height)
                image = image.resize(newsize)
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image)[None,]
            output_images.append(image)
        if len(output_images) > 1:
            output_image = torch.cat(output_images, dim=0)
        else:
            output_image = output_images[0]

    return output_image

def tensor_to_image(tensor):
    tensor = tensor.cpu()
    image_np = tensor.squeeze().mul(255).clamp(0, 255).byte().numpy()
    image = Image.fromarray(image_np, mode='RGB')
    return image

def Pic2Story(repo_id, img, prompts, special_tokens_skip = True, clean_same_result = True):
    story_out = None
    from transformers import BlipProcessor, BlipForConditionalGeneration
    import torch

    os.environ['TRANSFORMERS_OFFLINE'] = "1"
    processor = BlipProcessor.from_pretrained(repo_id)
    pil_image = tensor_to_image(img)

    try:
        model = BlipForConditionalGeneration.from_pretrained(repo_id, torch_dtype=torch.float16).to("cuda")
        if type(prompts) == str:
            inputs = processor(pil_image, prompts, return_tensors="pt").to("cuda", torch.float16)
            out = model.generate(**inputs)
            story_out = processor.decode(out[0], skip_special_tokens=special_tokens_skip)

        elif type(prompts).__name__ == 'list':
            for prompt in prompts:
                inputs = processor(pil_image, prompt, return_tensors="pt").to("cuda", torch.float16)
                out = model.generate(**inputs)
                Processed = processor.decode(out[0], skip_special_tokens=special_tokens_skip) + ', '
                if story_out is not None:
                    story_out = story_out + Processed
                else:
                    story_out = Processed

    except Exception:
        print('Pic2Story Float 16 failed')

    if type(story_out) != str:
        try:
            model = BlipForConditionalGeneration.from_pretrained(repo_id).to("cuda")
            if type(prompts) == str:
                inputs = processor(pil_image, prompts, return_tensors="pt").to("cuda")
                out = model.generate(**inputs)
                story_out = processor.decode(out[0], skip_special_tokens=special_tokens_skip)

            elif type(prompts).__name__ == 'list':
                for prompt in prompts:
                    inputs = processor(pil_image, prompt, return_tensors="pt").to("cuda")
                    out = model.generate(**inputs)
                    Processed = processor.decode(out[0], skip_special_tokens=special_tokens_skip) + ', '
                    if story_out is not None:
                        story_out = story_out + Processed
                    else:
                        story_out = Processed

        except Exception:
            print('Pic2Story GPU failed')

    if type(story_out) != str:
        try:
            model = BlipForConditionalGeneration.from_pretrained(repo_id)
            if type(prompts) == str:
                inputs = processor(pil_image, prompts, return_tensors="pt")
                out = model.generate(**inputs)
                story_out = processor.decode(out[0], skip_special_tokens=special_tokens_skip)

            elif type(prompts).__name__ == 'list':
                for prompt in prompts:
                    inputs = processor(pil_image, prompt, return_tensors="pt")
                    out = model.generate(**inputs)
                    Processed = processor.decode(out[0], skip_special_tokens=special_tokens_skip) + ', '
                    if story_out is not None:
                        story_out = story_out + Processed
                    else:
                        story_out = Processed

        except Exception:
            print('Pic2Story CPU failed')

    if type(story_out) == str:
        if clean_same_result == True:
            story_out = ' '.join(dict.fromkeys(story_out.split()))
        return story_out.rstrip(', ').replace(' and ', ' ').replace(' an ', ' ').replace(' is ', ' ').replace(' are ', ' ')
    else:
        return story_out

def getDownloadedFiles():
    DOWNLOAD_DIR = os.path.join(here, 'Nodes', 'Downloads')
    folder_paths.add_model_folder_path("primere_downloads", DOWNLOAD_DIR)
    downloaded_filelist = folder_paths.get_filename_list("primere_downloads")
    downloaded_filelist_filtered = folder_paths.filter_files_extensions(downloaded_filelist, ['.ckpt', '.safetensors'])
    return downloaded_filelist_filtered

def downloader(from_url, to_path):
    if os.path.isfile(to_path) == False:
        pathparser = urlparse(from_url)
        TargetFilename = os.path.basename(pathparser.path)
        print('Downloading: ' + TargetFilename)
        Request = requests.get(from_url, allow_redirects=True)
        if Request.status_code == 200 and Request.ok == True:
            open(to_path, 'wb').write(Request.content)
            print('DOWNLOADED: ' + to_path)
            return True
        else:
            print('ERROR: Cannot download ' + TargetFilename)
            return False
def fileDownloader(targetFILE, sourceURL):
    if os.path.exists(targetFILE) == False:
        print('Downloading from: ' + sourceURL + ' to: ' + str(targetFILE))
        reqsdlcm = requests.get(sourceURL, allow_redirects=True)
        if reqsdlcm.status_code == 200 and reqsdlcm.ok == True:
            open(targetFILE, 'wb').write(reqsdlcm.content)
            return True
        else:
            print('ERROR: Cannot dowload required file to: ' + str(targetFILE))
            return False
    return True