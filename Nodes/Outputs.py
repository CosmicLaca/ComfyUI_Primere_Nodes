from ..components.tree import TREE_OUTPUTS
import os
import folder_paths
import re
import json
import time
import numpy as np
import pyexiv2
from PIL.PngImagePlugin import PngInfo
from PIL import Image, PngImagePlugin
from pathlib import Path
import datetime
import comfy.samplers
import random
import nodes
import comfy_extras.nodes_custom_sampler as nodes_custom_sampler
import comfy_extras.nodes_stable_cascade as nodes_stable_cascade
import torch
from ..components import utility
from ..components import latentnoise
from itertools import compress
from server import PromptServer
from ..utils import comfy_dir
import clip
from ..components.tree import PRIMERE_ROOT

ALLOWED_EXT = ('.jpeg', '.jpg', '.png', '.tiff', '.gif', '.bmp', '.webp')

class PrimereMetaSave:
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("SAVED_INFO",)
    FUNCTION = "save_images_meta"
    OUTPUT_NODE = True
    CATEGORY = TREE_OUTPUTS

    NODE_FILE = os.path.abspath(__file__)
    NODE_ROOT = os.path.dirname(NODE_FILE)

    def __init__(self):
        self.output_dir = folder_paths.output_directory
        self.type = 'output'

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "save_image": ("BOOLEAN", {"default": True}),
                "images": ("IMAGE",),
                "output_path": ("STRING", {"default": '[time(%Y-%m-%d)]', "multiline": False}),
                "subpath": (["None", "Dev", "Test", "Production", "Preview", "NewModel", "Project", "Portfolio", "Character", "Style", "Product", "Fun", "SFW", "NSFW"], {"default": "Project"}),
                "add_modelname_to_path": ("BOOLEAN", {"default": False}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "filename_delimiter": ("STRING", {"default": "_"}),
                "add_date_to_filename": ("BOOLEAN", {"default": True}),
                "add_time_to_filename": ("BOOLEAN", {"default": True}),
                "add_seed_to_filename": ("BOOLEAN", {"default": True}),
                "add_size_to_filename": ("BOOLEAN", {"default": True}),
                "filename_number_padding": ("INT", {"default": 2, "min": 1, "max": 9, "step": 1}),
                "filename_number_start": ("BOOLEAN", {"default":False}),
                "extension": (['png', 'jpeg', 'jpg', 'gif', 'tiff', 'webp'], {"default": "jpg"}),
                "png_embed_workflow": ("BOOLEAN", {"default":False}),
                "image_embed_exif": ("BOOLEAN", {"default":False}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "overwrite_mode": (["false", "prefix_as_filename"],),
                "save_mata_to_json": ("BOOLEAN", {"default": False}),
                "save_info_to_txt": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image_metadata": ('TUPLE', {"forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT", "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    def save_images_meta(self, images, add_date_to_filename, add_time_to_filename, add_seed_to_filename, add_size_to_filename, save_mata_to_json, save_info_to_txt, image_metadata=None,
                         output_path='[time(%Y-%m-%d)]', subpath='Project', add_modelname_to_path = False, filename_prefix="ComfyUI", filename_delimiter='_',
                         extension='jpg', quality=95, prompt=None, extra_pnginfo=None,
                         overwrite_mode='false', filename_number_padding=2, filename_number_start=False,
                         png_embed_workflow=False, image_embed_exif=False, save_image=True):

        if save_image == False:
            saved_info = "*** Image saver switched OFF, image not saved. ***"
            return saved_info, {"ui": {"images": []}}

        delimiter = filename_delimiter
        number_padding = filename_number_padding
        tokens = TextTokens()

        original_output = self.output_dir
        filename_prefix = tokens.parseTokens(filename_prefix)
        nowdate = datetime.datetime.now()
        if image_metadata is None:
            image_metadata = {}

        if len(images) < 1:
            return

        image_metadata['saved_image_width'] = images[0].shape[1]
        image_metadata['saved_image_heigth'] = images[0].shape[0]
        # image_metadata['upscaler_ratio'] = round(image_metadata['saved_image_width'] / image_metadata['width'], 2)
        if 'width' in image_metadata and 'height' in image_metadata:
            image_metadata['upscaler_ratio'] = 'From: ' + str(image_metadata['width']) + 'x' + str(image_metadata['height']) + ' to: '  + str(image_metadata['saved_image_width']) + 'x' + str(image_metadata['saved_image_heigth']) + ' Ratio: ' + str(round(round(image_metadata['saved_image_width'] / image_metadata['width'] / 0.05) * 0.05, 2))

        if add_date_to_filename:
            filename_prefix = filename_prefix + '_' + nowdate.strftime("%Y%d%m")
        if add_time_to_filename:
            filename_prefix = filename_prefix + '_' + nowdate.strftime("%H%M%S")
        if add_seed_to_filename:
            if 'seed' in image_metadata:
                filename_prefix = filename_prefix + '_' + str(image_metadata['seed'])
        if add_size_to_filename:
            if 'width' in image_metadata:
                filename_prefix = filename_prefix + '_' + str(image_metadata['saved_image_width']) + 'x' + str(image_metadata['saved_image_heigth'])

        if output_path in [None, '', "none", "."]:
            output_path = self.output_dir
        else:
            output_path = tokens.parseTokens(output_path)
        if not os.path.isabs(output_path):
            output_path = os.path.join(self.output_dir, output_path)
        base_output = os.path.basename(output_path)
        if output_path.endswith("ComfyUI/output") or output_path.endswith("ComfyUI\output"):
            base_output = ""

        if add_modelname_to_path == True and 'model' in image_metadata:
            path = Path(output_path)
            ModelStartPath = output_path.replace(path.stem, '')
            ModelPath = Path(image_metadata['model'])
            # if preferred_subpath is not None and len(preferred_subpath.strip()) > 0:
            #    subpath = preferred_subpath
            if 'preferred' in image_metadata and type(image_metadata['preferred']).__name__ == 'dict' and len(image_metadata['preferred']) > 0 and 'subpath' in image_metadata['preferred'] and image_metadata['preferred']['subpath'] is not None and len(image_metadata['preferred']['subpath'].strip()) > 0:
                subpath = image_metadata['preferred']['subpath']

            if subpath is not None and subpath != 'None' and len(subpath.strip()) > 0:
                output_path = ModelStartPath + ModelPath.stem.upper() + os.sep + subpath + os.sep + path.stem
            else:
                output_path = ModelStartPath + ModelPath.stem.upper() + os.sep + path.stem
        else:
            if 'preferred' in image_metadata and type(image_metadata['preferred']).__name__ == 'dict' and len(image_metadata['preferred']) > 0 and 'subpath' in image_metadata['preferred'] and image_metadata['preferred']['subpath'] is not None and len(image_metadata['preferred']['subpath'].strip()) > 0:
                path = Path(output_path)
                ModelStartPath = output_path.replace(path.stem, '')
                subpath = image_metadata['preferred']['subpath']
                output_path = ModelStartPath + os.sep + subpath + os.sep + path.stem

        if output_path.strip() != '':
            if not os.path.isabs(output_path):
                output_path = os.path.join(folder_paths.output_directory, output_path)
            if not os.path.exists(output_path.strip()):
                print(f'The path `{output_path.strip()}` specified doesn\'t exist! Creating directory.')
                os.makedirs(output_path, exist_ok=True)

        if filename_number_start == 'true':
            pattern = f"(\\d{{{filename_number_padding}}}){re.escape(delimiter)}{re.escape(filename_prefix)}"
        else:
            pattern = f"{re.escape(filename_prefix)}{re.escape(delimiter)}(\\d{{{filename_number_padding}}})"
        existing_counters = [int(re.search(pattern, filename).group(1)) for filename in os.listdir(output_path) if re.match(pattern, os.path.basename(filename))]
        existing_counters.sort(reverse=True)

        if existing_counters:
            counter = existing_counters[0] + 1
        else:
            counter = 1

        file_extension = '.' + extension
        if file_extension not in ALLOWED_EXT:
            # print(f"The extension `{extension}` is not valid. The valid formats are: {', '.join(sorted(ALLOWED_EXT))}")
            file_extension = "jpg"

        results = list()
        # for image in images:
        image = images[0]
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        metadata = PngInfo()
        if png_embed_workflow == 'true':
            if prompt is not None:
                metadata.add_text("prompt", json.dumps(prompt))
            if extra_pnginfo is not None:
                for x in extra_pnginfo:
                    metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        if overwrite_mode == 'prefix_as_filename':
            file = f"{filename_prefix}{file_extension}"
        else:
            if filename_number_start == 'true':
                file = f"{counter:0{number_padding}}{delimiter}{filename_prefix}{file_extension}"
            else:
                file = f"{filename_prefix}{delimiter}{counter:0{number_padding}}{file_extension}"
            if os.path.exists(os.path.join(output_path, file)):
                counter += 1

        try:
            output_file = os.path.abspath(os.path.join(output_path, file))

#            exif_metadata_A11 = None
#             if 'positive' in image_metadata and 'negative' in image_metadata:
#                a11samplername = exif_data_checker.comfy_samplers2a11(image_metadata['sampler'], image_metadata['scheduler'])
#                exif_metadata_A11 = f"""{image_metadata['positive']}
# Negative prompt: {image_metadata['negative']}
# Steps: {str(image_metadata['steps'])}, Sampler: {a11samplername}, CFG scale: {str(image_metadata['cfg'])}, Seed: {str(image_metadata['seed'])}, Size: {str(image_metadata['width'])}x{str(image_metadata['height'])}, Model hash: {image_metadata['model_hash']}, Model: {image_metadata['model']}, VAE: {image_metadata['vae']}"""

            exif_metadata_json = image_metadata

            if extension == 'png':
                img.save(output_file, pnginfo=metadata, optimize=True)
            elif extension == 'webp':
                img.save(output_file, quality=quality, exif=metadata)
            else:
                img.save(output_file, quality=quality, optimize=True)
                if image_embed_exif == True:
                    metadata = pyexiv2.Image(output_file)
#                     if exif_metadata_A11 is not None:
#                        metadata.modify_exif({'Exif.Photo.UserComment': 'charset=Unicode ' + exif_metadata_A11})
                    metadata.modify_exif({'Exif.Image.ImageDescription': json.dumps(exif_metadata_json)})
                    print(f"Image file saved with exif: {output_file}")
                else:
                    if extension == 'webp':
                        img.save(output_file, quality=quality, exif=metadata)
                    else:
                        img.save(output_file, quality=quality, optimize=True)
                        print(f"Image file saved without exif: {output_file}")

            if save_mata_to_json:
                jsonfile = os.path.splitext(output_file)[0] + '.json'
                with open(jsonfile, 'w', encoding='utf-8') as jf:
                    json.dump(exif_metadata_json, jf, ensure_ascii=False, indent=4)

        except OSError as e:
            print(f'Unable to save file to: {output_file}')
            print(e)
        except Exception as e:
            print('Unable to save file due to the to the following error:')
            print(e)

        if overwrite_mode == 'false':
            counter += 1

        filtered_paths = []

        if filtered_paths:
            for image_path in filtered_paths:
                subfolder = self.get_subfolder_path(image_path, self.output_dir)
                image_data = {
                    "filename": os.path.basename(image_path),
                    "subfolder": subfolder,
                    "type": self.type
                }
                results.append(image_data)

        metastring = ""
        if image_metadata is not None and len(image_metadata) > 0:
            for key, val in image_metadata.items():
                if len(str(val).strip( '"')) > 0:
                    metastring = metastring + ':: ' + key.upper() + ': ' + str(val).strip( '"') + '\n'

        saved_info = f""":: Time to save: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
:: Output file: {output_file}

:: PROCESS INFO ::
------------------
{metastring}"""

        if save_info_to_txt:
            infofile = os.path.splitext(output_file)[0] + '.txt'
            with open(infofile, 'w', encoding='utf-8', newline="") as infofile:
                infofile.write(saved_info)

        return saved_info, {"ui": {"images": []}}

    def get_subfolder_path(self, image_path, output_path):
        output_parts = output_path.strip(os.sep).split(os.sep)
        image_parts = image_path.strip(os.sep).split(os.sep)
        common_parts = os.path.commonprefix([output_parts, image_parts])
        subfolder_parts = image_parts[len(common_parts):]
        subfolder_path = os.sep.join(subfolder_parts[:-1])
        return subfolder_path

class TextTokens:
    def __init__(self):
        self.tokens = {
            '[time]': str(time.time()).replace('.', '_')
        }
        if '.' in self.tokens['[time]']: self.tokens['[time]'] = self.tokens['[time]'].split('.')[0]

    def format_time(self, format_code):
        return time.strftime(format_code, time.localtime(time.time()))

    def parseTokens(self, text):
        tokens = self.tokens.copy()

        # Update time
        tokens['[time]'] = str(time.time())
        if '.' in tokens['[time]']:
            tokens['[time]'] = tokens['[time]'].split('.')[0]

        for token, value in tokens.items():
            if token.startswith('[time('):
                continue
            text = text.replace(token, value)

        def replace_custom_time(match):
            format_code = match.group(1)
            return self.format_time(format_code)

        text = re.sub(r'\[time\((.*?)\)\]', replace_custom_time, text)
        return text

class AnyType(str):
  def __ne__(self, __value: object) -> bool:
    return False

any = AnyType("*")

class PrimereAnyOutput:
  RETURN_TYPES = ()
  FUNCTION = "show_output"
  OUTPUT_NODE = True
  CATEGORY = TREE_OUTPUTS

  @classmethod
  def INPUT_TYPES(cls):
    return {
      "required": {
        "input": (any, {}),
      },
    }

  def show_output(self, input = None):
    value = 'None'
    if input is not None:
      try:
        value = json.dumps(input, indent=4)
      except Exception:
        try:
          value = str(input)
        except Exception:
          value = 'Input data exists, but could not be serialized.'

    return {"ui": {"text": (value.strip( '"'),)}}

class PrimereTextOutput:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"forceInput": True}),
            },
        }

    INPUT_IS_LIST = True
    RETURN_TYPES = ()
    FUNCTION = "notify"
    OUTPUT_NODE = True
    OUTPUT_IS_LIST = (True,)
    CATEGORY = TREE_OUTPUTS

    def notify(self, text):
        return {"ui": {"text": text}}

class PrimereMetaCollector:
    CATEGORY = TREE_OUTPUTS
    RETURN_TYPES = ("TUPLE",)
    RETURN_NAMES = ("METADATA",)
    FUNCTION = "load_process_meta"

    INPUT_DICT = {
        "required": {
                "positive": ('STRING', {"forceInput": True, "default": "Red sportcar racing"}),
                "negative": ('STRING', {"forceInput": True, "default": "Cute cat, nsfw, nude, nudity, porn"})
            }, "optional": {
                # "seed": ('INT', {"forceInput": True, "default": 1}),
                "positive_l": ('STRING', {"forceInput": True, "default": None}),
                "negative_l": ('STRING', {"forceInput": True, "default": None}),
                "positive_r": ('STRING', {"forceInput": True, "default": None}),
                "negative_r": ('STRING', {"forceInput": True, "default": None}),
                "model": ('CHECKPOINT_NAME', {"forceInput": True, "default": None}),
                "model_version": ("STRING", {"default": 'BaseModel_1024', "forceInput": True}),
                "model_concept": ("STRING", {"default": "Normal", "forceInput": True}),
                "concept_data": ("TUPLE", {"default": None, "forceInput": True}),
                "sampler": (comfy.samplers.KSampler.SAMPLERS, {"forceInput": True, "default": "euler"}),
                "scheduler": (comfy.samplers.KSampler.SCHEDULERS, {"forceInput": True, "default": "normal"}),
                "width": ('INT', {"forceInput": True, "default": 512}),
                "height": ('INT', {"forceInput": True, "default": 512}),
                "model_shapes": ('TUPLE', {"forceInput": True, "default": None}),
                "cfg": ('FLOAT', {"forceInput": True, "default": 7}),
                "steps": ('INT', {"forceInput": True, "default": 12}),
                "vae_name_sd": ('VAE_NAME', {"forceInput": True, "default": None}),
                "vae_name_sdxl": ('VAE_NAME', {"forceInput": True, "default": None}),
                "preferred": ("TUPLE", {"default": None, "forceInput": True})
            },
        }

    @classmethod
    def INPUT_TYPES(cls):
        return cls.INPUT_DICT

    def load_process_meta(self,  *args, **kwargs):
        data_json = {}

        for key, value in self.INPUT_DICT.items():
            for key_l2, value_l2 in value.items():
                if 'default' in value_l2[1]:
                    default_value = value_l2[1]['default']
                else:
                    default_value = None

                if key_l2 not in kwargs:
                    data_json[key_l2] = default_value
                else:
                    data_json[key_l2] = kwargs[key_l2]

        return (data_json,)

class PrimereKSampler:
    CATEGORY = TREE_OUTPUTS
    RETURN_TYPES =("LATENT",)
    RETURN_NAMES = ("LATENT",)
    FUNCTION = "pk_sampler"

    def __init__(self):
        self.state_hash = False
        self.count = 0

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        if kwargs['variation_extender'] > 0 or kwargs['device'] != 'DEFAULT' or kwargs['variation_batch_step'] > 0:
            return float("NaN")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                    "model": ("MODEL", {"forceInput": True}),
                    "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                    "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                    "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                    "sampler_name": (comfy.samplers.KSampler.SAMPLERS, ),
                    "scheduler_name": (comfy.samplers.KSampler.SCHEDULERS, ),
                    "positive": ("CONDITIONING", ),
                    "negative": ("CONDITIONING", ),
                    "latent_image": ("LATENT", ),
                    "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "variation_extender": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                    "variation_batch_step": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01}),
                    "device": (["DEFAULT", "GPU", "CPU"], {"default": 'DEFAULT'}),
                },
                "optional": {
                    "model_concept": ("STRING", {"default": "Normal", "forceInput": True}),
                },
                "hidden": {
                    "extra_pnginfo": "EXTRA_PNGINFO",
                    "prompt": "PROMPT"
                }
            }

    def pk_sampler(self, model, seed, steps, cfg, sampler_name, scheduler_name, positive, negative, latent_image, extra_pnginfo, prompt, model_concept = "Normal", denoise=1.0, variation_extender = 0, variation_batch_step = 0, device = 'DEFAULT'):
        samples = latent_image
        variation_extender_original = variation_extender
        variation_batch_step_original = variation_batch_step

        def check_state(self, extra_pnginfo, prompt):
            old = self.state_hash
            self.state_hash = utility.collect_state(extra_pnginfo, prompt)
            if self.state_hash == old:
                self.count += 1
                return self.count
            self.count = 0
            return self.count

        batch_counter = int(check_state(self, extra_pnginfo, prompt)) + 1

        match model_concept:
            case "Turbo":
                sigmas = nodes_custom_sampler.SDTurboScheduler().get_sigmas(model, steps, denoise)
                sampler = comfy.samplers.sampler_object(sampler_name)
                turbo_samples = nodes_custom_sampler.SamplerCustom().sample(model, True, seed, cfg, positive, negative, sampler, sigmas[0], latent_image)
                samples = (turbo_samples[0],)
                # return samples

            case "Cascade":
                if type(model).__name__ == 'list':
                    latent_size = utility.getLatentSize(latent_image)
                    if (latent_size[0] < latent_size[1]):
                        orientation = 'Vertical'
                    else:
                        orientation = 'Horizontal'

                    dimensions = utility.get_dimensions_by_shape(self, 'Square [1:1]', 1024, orientation, True, True, latent_size[0], latent_size[1], 'CASCADE')
                    dimension_x = dimensions[0]
                    dimension_y = dimensions[1]

                    height = dimension_y
                    width = dimension_x
                    compression = 42
                    if type(model[0]).__name__ == 'ModelPatcher' and type(model[1]).__name__ == 'ModelPatcher':
                        c_latent = {"samples": torch.zeros([1, 16, height // compression, width // compression])}
                        b_latent = {"samples": torch.zeros([1, 4, height // 4, width // 4])}
                        samples_c = nodes.KSampler.sample(self, model[1], seed, steps, cfg, sampler_name, scheduler_name, positive, negative, c_latent, denoise=denoise)[0]
                        conditining_c = nodes_stable_cascade.StableCascade_StageB_Conditioning.set_prior(self, positive, samples_c)[0]
                        samples = nodes.KSampler.sample(self, model[0], seed, 10, 1.00, sampler_name, scheduler_name, conditining_c, negative, b_latent, denoise=denoise)
                        # return samples
            case _:
                if variation_batch_step_original > 0:
                    if batch_counter > 0:
                        variation_batch_step = variation_batch_step_original * batch_counter

                    variation_extender = round(variation_extender_original + variation_batch_step, 2)

                if variation_extender_original > 0 or device != 'DEFAULT' or variation_batch_step_original > 0:
                    if (variation_extender > 1):
                        random.seed(batch_counter)
                        variation_extender = round(random.uniform(0.01, 1.00), 2)
                    if variation_batch_step == 0:
                        variation_seed = batch_counter + seed
                    else:
                        variation_seed = seed
                    samples = latentnoise.noisy_samples(model, device, steps, cfg, sampler_name, scheduler_name, positive, negative, latent_image, denoise, variation_seed, variation_extender)
                else:
                    samples = nodes.KSampler.sample(self, model, seed, steps, cfg, sampler_name, scheduler_name, positive, negative, latent_image, denoise=denoise)
                    # return samples

        return samples

    '''
        if variation_batch_step > 0:
            variation_extender = variation_extender + (batch_counter * variation_batch_step)

        if variation_extender > 0 or device != 'DEFAULT':
            variation_seed = batch_counter + seed
            samples = latentnoise.noisy_samples(model, device, steps, cfg, sampler_name, scheduler_name, positive, negative, samples, denoise, variation_seed, variation_extender)[0]
            # return samples    
    '''

class PrimerePreviewImage():
    CATEGORY = TREE_OUTPUTS
    RETURN_TYPES = ()
    OUTPUT_NODE = True
    FUNCTION = "preview_img_saver"

    image_path = folder_paths.get_output_directory()

    def __init__(self):
        self.output_dir = folder_paths.get_temp_directory()
        self.type = "temp"
        self.prefix_append = "_temp_" + ''.join(random.choice("abcdefghijklmnopqrstupvxyz") for x in range(5))
        self.compress_level = 1

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image_save_as": ("BOOLEAN", {"default": False, "label_on": "Save as preview", "label_off": "Save as any..."}),
                "image_type": (['jpeg', 'png', 'webp'], {"default": "jpeg"}),
                "image_resize": ("INT", {"default": 0, "min": 0, "max": utility.MAX_RESOLUTION, "step": 64}),
                "image_quality": ("INT",  {"default": 95,"min": 10, "max": 100, "step": 5}),
                "preview_target": (['Checkpoint', 'CSV Prompt', 'Lora', 'Lycoris', 'Hypernetwork', 'Embedding'],),
                "preview_save_mode": (['Overwrite', 'Keep', 'Join horizontal', 'Join vertical'], {"default": "Overwrite"}),

                "images": ("IMAGE", ),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "image_path": (cls.image_path,),
                "id": "UNIQUE_ID",
            },
        }

    def preview_img_saver(self, images, *args, **kwargs):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"
        self.compress_level = 4

        VISUAL_NODE_NAMES = ['PrimereVisualCKPT', 'PrimereVisualLORA', 'PrimereVisualEmbedding', 'PrimereVisualHypernetwork', 'PrimereVisualStyle', 'PrimereVisualLYCORIS']
        VISUAL_NODE_FILENAMES = ['PrimereVisualCKPT', 'PrimereVisualLORA', 'PrimereVisualEmbedding', 'PrimereVisualHypernetwork', 'PrimereVisualLYCORIS']
        WIDGET_DATA = {
            "PrimereVisualCKPT": [0],
            "PrimereVisualStyle": [0],
            "PrimereVisualLORA": [6, 10, 14, 18, 22, 26],
            "PrimereVisualEmbedding": [5, 9, 13, 17, 21, 25],
            "PrimereVisualHypernetwork": [6, 9, 12, 15, 18, 21],
            "PrimereVisualLYCORIS": [6, 10, 14, 18, 22, 26],
        }

        WORKFLOWDATA = kwargs['extra_pnginfo']['workflow']['nodes']
        VISUAL_DATA = {}
        for NODE_ITEMS in WORKFLOWDATA:
            ITEM_TYPE = NODE_ITEMS['type']
            if ITEM_TYPE in VISUAL_NODE_NAMES:
                ITEM_VALUES = NODE_ITEMS['widgets_values']
                if ITEM_TYPE in WIDGET_DATA:
                    REQUIRED_DATA_LISTINDEX = WIDGET_DATA[ITEM_TYPE]
                    WIDGET_STATES = [True]
                    if len(REQUIRED_DATA_LISTINDEX) > 1:
                        SWITCH_LIST = list(map(lambda x: x + -1, REQUIRED_DATA_LISTINDEX))
                        WIDGET_STATES = list(map(ITEM_VALUES.__getitem__, SWITCH_LIST))

                    VALID_WIDGET_VALUES = list(map(ITEM_VALUES.__getitem__, REQUIRED_DATA_LISTINDEX))
                    REUIRED_WIDGETS = list(compress(VALID_WIDGET_VALUES, WIDGET_STATES))
                    REPLACED_WIDGETS = [widg.replace(' ', '_') for widg in REUIRED_WIDGETS]
                    if ITEM_TYPE in VISUAL_NODE_FILENAMES:
                        REPLACED_WIDGETS = [Path(widg).stem for widg in REPLACED_WIDGETS]

                    if ITEM_TYPE in VISUAL_DATA.keys():
                        VISUAL_DATA[ITEM_TYPE] = VISUAL_DATA[ITEM_TYPE] + REPLACED_WIDGETS
                        VISUAL_DATA[ITEM_TYPE + '_ORIGINAL'] = VISUAL_DATA[ITEM_TYPE + '_ORIGINAL'] + REUIRED_WIDGETS
                    else:
                        VISUAL_DATA[ITEM_TYPE] = REPLACED_WIDGETS
                        VISUAL_DATA[ITEM_TYPE + '_ORIGINAL'] = REUIRED_WIDGETS

                    VISUAL_DATA[ITEM_TYPE] = [i for n, i in enumerate(VISUAL_DATA[ITEM_TYPE]) if i not in VISUAL_DATA[ITEM_TYPE][:n]]
                    VISUAL_DATA[ITEM_TYPE + '_ORIGINAL'] = [i for n, i in enumerate(VISUAL_DATA[ITEM_TYPE + '_ORIGINAL']) if i not in VISUAL_DATA[ITEM_TYPE + '_ORIGINAL'][:n]]

        PromptServer.instance.send_sync("getVisualTargets", VISUAL_DATA)

        results = nodes.SaveImage.save_images(self, images, filename_prefix = "ComfyUI", prompt = None, extra_pnginfo = None)
        return results

class PrimereAestheticCKPTScorer():
    CATEGORY = TREE_OUTPUTS
    RETURN_TYPES = ("INT",)
    RETURN_NAMES = ("SCORE",)
    OUTPUT_NODE = True
    FUNCTION = "aesthetic_scorer"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "get_aesthetic_score": ("BOOLEAN", {"default": False}),
                "add_to_checkpoint": ("BOOLEAN", {"default": False}),
                "add_to_saved_prompt": ("BOOLEAN", {"default": False}),
                "image": ("IMAGE", ),
            },
            "optional": {
                "workflow_data": ('TUPLE', {"forceInput": True}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
            },
        }

    def aesthetic_scorer(self, image, get_aesthetic_score, add_to_checkpoint, add_to_saved_prompt, workflow_data = None, **kwargs):
        final_prediction = '*** Aesthetic scorer off ***'

        if (get_aesthetic_score == True):
            AESTHETIC_PATH = os.path.join(comfy_dir, 'models', 'aesthetic')
            folder_paths.add_model_folder_path("aesthetic", AESTHETIC_PATH)
            if os.path.exists(AESTHETIC_PATH) == False:
                Path(AESTHETIC_PATH).mkdir(parents=True, exist_ok=True)
            AESTH_FULL_LIST = folder_paths.get_filename_list("aesthetic")
            aestheticFiles = folder_paths.filter_files_extensions(AESTH_FULL_LIST, ['.pth'])

            if 'chadscorer.pth' not in aestheticFiles:
                FileUrl = 'https://huggingface.co/primerecomfydev/chadscorer/resolve/main/chadscorer.pth?download=true'
                FullFilePath = os.path.join(AESTHETIC_PATH, 'chadscorer.pth')
                ModelDownload = utility.downloader(FileUrl, FullFilePath)
                if (ModelDownload == True):
                    AESTH_FULL_LIST = folder_paths.get_filename_list("aesthetic")
                    aestheticFiles = folder_paths.filter_files_extensions(AESTH_FULL_LIST, ['.pth'])

            if 'chadscorer.pth' in aestheticFiles:
                folder_paths.folder_names_and_paths["aesthetic"] = ([os.path.join(folder_paths.models_dir, "aesthetic")], folder_paths.supported_pt_extensions)
                m_path = folder_paths.folder_names_and_paths["aesthetic"][0]
                aesthetic_model = os.path.join(m_path[0], 'chadscorer.pth')
                model = utility.MLP(768)
                s = torch.load(aesthetic_model)
                model.load_state_dict(s)
                model.to("cuda")
                model.eval()
                device = "cuda"
                model2, preprocess = clip.load("ViT-L/14", device=device)  # RN50x64
                tensor_image = image[0]
                img = (tensor_image * 255).to(torch.uint8).numpy()
                pil_image = Image.fromarray(img, mode='RGB')
                image2 = preprocess(pil_image).unsqueeze(0).to(device)
                with torch.no_grad():
                    image_features = model2.encode_image(image2)
                    pass
                im_emb_arr = utility.normalized(image_features.cpu().detach().numpy())
                prediction = model(torch.from_numpy(im_emb_arr).to(device).type(torch.cuda.FloatTensor))
                final_prediction = int(float(prediction[0]) * 100)
                del model

                if (type(final_prediction) != 'str'):
                    final_prediction = str(final_prediction)

                if workflow_data is not None:
                    if add_to_checkpoint == True:
                        if 'model' in workflow_data:
                            selected_model = workflow_data['model']
                            modelname_only = Path(selected_model).stem
                            model_ascore = utility.get_value_from_cache('model_ascores', modelname_only)
                            if model_ascore is None:
                                utility.add_value_to_cache('model_ascores', modelname_only, '1|' + final_prediction)
                            else:
                                model_ascore_list = model_ascore.split("|")
                                counter = str(int(model_ascore_list[0]) + 1)
                                score = str(int(model_ascore_list[1]) + int(final_prediction))
                                utility.add_value_to_cache('model_ascores', modelname_only, counter + '|' + score)

                    if (add_to_saved_prompt == True):
                        if 'positive' in workflow_data:
                            WORKFLOWDATA = kwargs['extra_pnginfo']['workflow']['nodes']
                            selectedStyle = utility.getDataFromWorkflow(WORKFLOWDATA, 'PrimereVisualStyle', 0)
                            if (selectedStyle is None):
                                selectedStyle = utility.getDataFromWorkflow(WORKFLOWDATA, 'PrimereStyleLoader', 0)

                            if selectedStyle is not None:
                                STYLE_DIR = os.path.join(PRIMERE_ROOT, 'stylecsv')
                                STYLE_FILE = os.path.join(STYLE_DIR, "styles.csv")
                                STYLE_FILE_EXAMPLE = os.path.join(STYLE_DIR, "styles.example.csv")
                                if Path(STYLE_FILE).is_file() == True:
                                    STYLE_SOURCE = STYLE_FILE
                                else:
                                    STYLE_SOURCE = STYLE_FILE_EXAMPLE
                                style_data = utility.load_external_csv(STYLE_SOURCE, 0)
                                positive_prompt = style_data[style_data['name'] == selectedStyle]['prompt'].values[0]
                                if (positive_prompt is not None):
                                    if len(positive_prompt) > 100:
                                        positive_prompt = positive_prompt[:100]
                                    if positive_prompt in workflow_data['positive']:
                                        style_ascore = utility.get_value_from_cache('style_ascores', selectedStyle)
                                        if style_ascore is None:
                                            utility.add_value_to_cache('style_ascores', selectedStyle, '1|' + final_prediction)
                                        else:
                                            style_ascore_list = style_ascore.split("|")
                                            counter = str(int(style_ascore_list[0]) + 1)
                                            score = str(int(style_ascore_list[1]) + int(final_prediction))
                                            utility.add_value_to_cache('style_ascores', selectedStyle, counter + '|' + score)

        return {"ui": {"text": [final_prediction]}, "result": (final_prediction,)}
