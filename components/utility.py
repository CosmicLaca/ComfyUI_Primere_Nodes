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
from ..utils import cache_file
import os
import json
import numpy as np
from PIL import Image
from torchvision.transforms import InterpolationMode
import torchvision.transforms.functional as F
from urllib.parse import urlparse
import requests
import folder_paths
import comfy_extras.nodes_model_advanced as nodes_model_advanced
import nodes
from ..utils import comfy_dir

SUPPORTED_FORMATS = [".png", ".jpg", ".jpeg", ".webp"]
STANDARD_SIDES = np.arange(64, 2049, 16).tolist()
CASCADE_SIDES = np.arange(64, 2049, 16).tolist()
MAX_RESOLUTION = 8192
VALID_SHAPES = np.arange(512, 2049, 256).tolist()
PREVIEW_ROOT = os.path.join(comfy_dir, "web", "extensions", "Primere", "images")

PREVIEW_PATH_BY_TYPE = {
    "Checkpoint": os.path.join(PREVIEW_ROOT, "checkpoints"),
    "CSV Prompt": os.path.join(PREVIEW_ROOT, "styles"),
    "Lora": os.path.join(PREVIEW_ROOT, "loras"),
    "Lycoris": os.path.join(PREVIEW_ROOT, "lycoris"),
    "Hypernetwork": os.path.join(PREVIEW_ROOT, "hypernetworks"),
    "Embedding": os.path.join(PREVIEW_ROOT, "embeddings"),
}

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
    return re.sub("(\.\d+)|(:\d+)|[()]|BREAK|break", "", prompt).replace('  ', ' ')

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

def ModelObjectParser(modelobject):
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

    return ckpt_type + '_' + str(ModelVersion)

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
            matchre = re.compile('Expected (\d+) fields in line (\d+), saw (\d+)')
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

def ModelConceptNames(ckpt_name, model_concept, lightning_selector, lightning_model_step):
    lora_name = None
    unet_name = None
    lightningModeValid = False

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
            LoraList = folder_paths.get_filename_list("loras")
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

    return {'ckpt_name': ckpt_name, 'lora_name': lora_name, 'unet_name': unet_name, 'lightningModeValid': lightningModeValid}

def LightningConceptModel(self, model_concept, lightningModeValid, lightning_selector, lightning_model_step, OUTPUT_MODEL, lora_name, unet_name):
    if model_concept == 'Lightning' and lightningModeValid == True and lightning_selector == 'LORA' and lora_name is not None:
        OUTPUT_MODEL = nodes.LoraLoader.load_lora(self, OUTPUT_MODEL, None, lora_name, 1, 0)[0]

    if model_concept == 'Lightning' and lightningModeValid == True and lightning_selector == 'UNET' and unet_name is not None:
        OUTPUT_MODEL = nodes.UNETLoader.load_unet(self, unet_name)[0]

    if model_concept == 'Lightning' and lightning_model_step == 1 and lightningModeValid == True:
        OUTPUT_MODEL = nodes_model_advanced.ModelSamplingDiscrete.patch(self, OUTPUT_MODEL, "x0", False)[0]

    return OUTPUT_MODEL

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
