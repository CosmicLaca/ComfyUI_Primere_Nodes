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

SUPPORTED_FORMATS = [".png", ".jpg", ".jpeg", ".webp"]
STANDARD_SIDES = [64, 80, 96, 128, 144, 160, 192, 256, 320, 368, 400, 480, 512, 560, 640, 704, 768, 832, 896, 960, 1024, 1088, 1152, 1216, 1280, 1344, 1408, 1472, 1536, 1600, 1664, 1728, 1792, 1856, 1920, 1984, 2048]

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

def calculate_dimensions(self, ratio: str, orientation: str, round_to_standard: bool, model_version: str, calculate_by_custom: bool, custom_side_a: float, custom_side_b: float):
    SD_1 = 512
    SD_2 = 768
    SD_1024 = 1024
    SD_1280 = 1280
    SDXL_1 = 1024
    DEFAULT_RES = SD_2

    match model_version:
        case 'BaseModel_768':
            DEFAULT_RES = SD_1
        case 'BaseModel_1024':
            DEFAULT_RES = SD_2
        case 'BaseModel_mod_1024':
            DEFAULT_RES = SD_1024
        case 'BaseModel_mod_1280':
            DEFAULT_RES = SD_1280
        case 'SDXL_2048':
            DEFAULT_RES = SDXL_1

    def calculate(ratio_1: float, ratio_2: float, side: int):
        FullPixels = side ** 2
        result_x = FullPixels / ratio_2
        result_y = result_x / ratio_1
        side_base = round(math.sqrt(result_y))
        side_a = round(ratio_1 * side_base)
        if round_to_standard == True:
            side_a = min(STANDARD_SIDES, key=lambda x: abs(side_a - x))
        side_b = round(FullPixels / side_a)
        return sorted([side_a, side_b], reverse=True)

    if (calculate_by_custom == True and isinstance(custom_side_a, (int, float)) and isinstance(custom_side_b, (int, float)) and custom_side_a >= 1 and custom_side_b >= 1):
        ratio_x = custom_side_a
        ratio_y = custom_side_b
    else:
        RatioLabel = self.ratioNames[ratio]
        ratio_x = self.sd_ratios[RatioLabel]['side_x']
        ratio_y = self.sd_ratios[RatioLabel]['side_y']

    dimensions = calculate(ratio_x, ratio_y, DEFAULT_RES)
    if (orientation == 'Vertical'):
        dimensions = sorted(dimensions)

    dimension_x = dimensions[0]
    dimension_y = dimensions[1]
    return (dimension_x, dimension_y,)

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

def rescale_zero_terminal_snr_sigmas(sigmas):
    alphas_cumprod = 1 / ((sigmas * sigmas) + 1)
    alphas_bar_sqrt = alphas_cumprod.sqrt()
    alphas_bar_sqrt_0 = alphas_bar_sqrt[0].clone()
    alphas_bar_sqrt_T = alphas_bar_sqrt[-1].clone()
    alphas_bar_sqrt -= (alphas_bar_sqrt_T)
    alphas_bar_sqrt *= alphas_bar_sqrt_0 / (alphas_bar_sqrt_0 - alphas_bar_sqrt_T)
    alphas_bar = alphas_bar_sqrt ** 2  # Revert sqrt
    alphas_bar[-1] = 4.8973451890853435e-08
    return ((1 - alphas_bar) / alphas_bar) ** 0.5

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

class LCM(comfy.model_sampling.EPS):
    def calculate_denoised(self, sigma, model_output, model_input):
        timestep = self.timestep(sigma).view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        sigma = sigma.view(sigma.shape[:1] + (1,) * (model_output.ndim - 1))
        x0 = model_input - model_output * sigma

        sigma_data = 0.5
        scaled_timestep = timestep * 10.0 #timestep_scaling

        c_skip = sigma_data**2 / (scaled_timestep**2 + sigma_data**2)
        c_out = scaled_timestep / (scaled_timestep**2 + sigma_data**2) ** 0.5

        return c_out * x0 + c_skip * model_input

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

            cutoff_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
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

def get_closest_element(value, list):
    cutoff_list = [1, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]
    is_found = None

    for trycut in cutoff_list:
        is_found = difflib.get_close_matches(value, list, cutoff=trycut)
        if len(is_found) >= 1:
            return is_found[0]

    return is_found