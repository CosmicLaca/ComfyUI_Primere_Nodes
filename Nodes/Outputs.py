from ..components.tree import TREE_OUTPUTS
import os
import folder_paths
import re
import json
import time
import numpy as np
import pyexiv2
from PIL.PngImagePlugin import PngInfo
from PIL import Image
from pathlib import Path
import datetime
import comfy.samplers
from comfy import model_management
import random
import nodes
import comfy_extras.nodes_flux as nodes_flux
import torch
from ..components import utility
from ..components import primeresamplers
from ..components import file_output
from server import PromptServer
from ..components.tree import PRIMERE_ROOT
from comfy.cli_args import args
from .modules import exif_data_checker
from ..Nodes.Visuals import PrimereVisualCKPT
from ..Nodes.Visuals import PrimereVisualStyle
from transformers import pipeline
from torchvision.transforms import functional as TF
import comfy_extras.nodes_model_advanced as nodes_model_advanced

ALLOWED_EXT = file_output.ALLOWED_EXT

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
                "aesthetic_trigger": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                "images": ("IMAGE",),
                "output_path": ("STRING", {"default": '[time(%Y-%m-%d)]', "multiline": False}),
                "subpath": (["None", "Dev", "Test", "Serie", "Production", "Preview", "NewModel", "Project", "Portfolio", "Civitai", "Behance", "Facebook", "Instagram", "Character", "Style", "Product", "Fun", "SFW", "NSFW"], {"default": "Project"}),
                "subpath_priority": ("BOOLEAN", {"default": False, "label_on": "Preferred", "label_off": "Selected subpath"}),
                "add_modelname_to_path": ("BOOLEAN", {"default": False}),
                "add_concept_to_path": ("BOOLEAN", {"default": False}),
                "filename_prefix": ("STRING", {"default": "ComfyUI"}),
                "filename_delimiter": ("STRING", {"default": "_"}),
                "add_date_to_filename": ("BOOLEAN", {"default": True}),
                "add_time_to_filename": ("BOOLEAN", {"default": True}),
                "add_seed_to_filename": ("BOOLEAN", {"default": True}),
                "add_size_to_filename": ("BOOLEAN", {"default": True}),
                "add_ascore_to_filename": ("BOOLEAN", {"default": True}),
                "filename_number_padding": ("INT", {"default": 2, "min": 1, "max": 9, "step": 1}),
                "filename_number_start": ("BOOLEAN", {"default":False}),
                "extension": (['png', 'jpeg', 'jpg', 'gif', 'tiff', 'webp'], {"default": "jpg"}),
                "png_embed_workflow": ("BOOLEAN", {"default": False}),
                "png_embed_data": ("BOOLEAN", {"default": False}),
                "image_embed_exif": ("BOOLEAN", {"default": False}),
                "a1111_civitai_meta": ("BOOLEAN", {"default": False}),
                "quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
                "overwrite_mode": (["false", "prefix_as_filename"],),
                "save_meta_to_json": ("BOOLEAN", {"default": False}),
                "save_info_to_txt": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "image_metadata": ('TUPLE', {"forceInput": True}),
            },
            "hidden": {
                "prompt": "PROMPT",
                "extra_pnginfo": "EXTRA_PNGINFO"
            },
        }

    def save_images_meta(self, images, add_date_to_filename, add_time_to_filename, add_seed_to_filename, add_size_to_filename, add_ascore_to_filename, save_meta_to_json, save_info_to_txt, image_metadata=None,
                         output_path='[time(%Y-%m-%d)]', subpath='Project', subpath_priority = False, add_modelname_to_path = False, add_concept_to_path = False, filename_prefix="ComfyUI", filename_delimiter='_',
                         extension='jpg', quality=95, prompt=None, extra_pnginfo=None,
                         overwrite_mode='false', filename_number_padding=2, filename_number_start=False,
                         png_embed_workflow=False, png_embed_data=False, image_embed_exif=False, a1111_civitai_meta=False, save_image=True, aesthetic_trigger = 0):

        if save_image == False:
            saved_info = "*** Image saver switched OFF, image not saved. ***"
            return saved_info, {"ui": {"images": []}}

        if 'aesthetic_score' in image_metadata:
            if (type(image_metadata['aesthetic_score']).__name__ == 'int'):
                image_metadata['aesthetic_score'] = str(image_metadata['aesthetic_score'])

            if (image_metadata['aesthetic_score'].isdigit()) and int(image_metadata['aesthetic_score']) > 0:
                if aesthetic_trigger > int(image_metadata['aesthetic_score']):
                    saved_info = "*** Image ignored because aesthetic score: [" + str(image_metadata['aesthetic_score']) + "] less than trigger setting: [" + str(aesthetic_trigger) + "]. ***"
                    return saved_info, {"ui": {"images": []}}

        delimiter = filename_delimiter
        number_padding = filename_number_padding
        tokens = file_output.TextTokens()

        original_output = self.output_dir
        filename_prefix = file_output.sanitize_path_part(tokens.parseTokens(filename_prefix))
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

        if add_ascore_to_filename:
            if 'aesthetic_score' in image_metadata:
                if (image_metadata['aesthetic_score'].isdigit()) and int(image_metadata['aesthetic_score']) > 0:
                    filename_prefix = filename_prefix + '_A' + str(image_metadata['aesthetic_score'])
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

        output_path = file_output.parse_output_path_base(output_path, self.output_dir)
        base_output = os.path.basename(output_path)
        if output_path.endswith("ComfyUI/output") or output_path.endswith("ComfyUI\output"):
            base_output = ""

        subdirs = []

        if add_concept_to_path == True and 'model_concept' in image_metadata:
            concept_name = image_metadata['model_concept']
            if concept_name == 'Auto' and 'model_version' in image_metadata:
                match image_metadata['model_version']:
                    case 'SDXL_2048':
                        concept_name = 'SDXL'
                    case 'BaseModel_768':
                        concept_name = 'SD1'
                    case 'SD3_1024':
                        concept_name = 'SD3'
                    case 'Stable_Zero123_768':
                        concept_name = 'Stable_Zero'
            subdirs.append(file_output.sanitize_path_part(Path(concept_name).stem.upper()))

        if add_modelname_to_path == True and 'model' in image_metadata:
            if image_metadata.get('model_name'):
                image_metadata['model'] = image_metadata['model_name']
            subdirs.append(file_output.sanitize_path_part(Path(image_metadata['model']).stem.upper()))

        if subpath_priority == True and 'preferred' in image_metadata and type(image_metadata['preferred']).__name__ == 'dict' and len(image_metadata['preferred']) > 0 and 'subpath' in image_metadata['preferred']:
            if image_metadata['preferred']['subpath'] is not None and len(image_metadata['preferred']['subpath'].strip()) > 0:
                subpath = image_metadata['preferred']['subpath']
            subdirs.append(file_output.sanitize_path_part(subpath))
        elif subpath_priority == False and subpath is not None and subpath != 'None' and len(subpath.strip()) > 0:
            subdirs.append(file_output.sanitize_path_part(subpath))

        output_path = file_output.append_subdirs_before_stem(output_path, subdirs)
        output_path = file_output.ensure_output_dir(output_path)

        file, counter = file_output.build_filename_and_counter(
            output_path=output_path,
            prefix=filename_prefix,
            delimiter=delimiter,
            number_padding=number_padding,
            number_start=filename_number_start,
            extension=extension,
            overwrite_mode=overwrite_mode,
        )

        results = list()
        # for image in images:
        image = images[0]
        i = 255. * image.cpu().numpy()
        img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))

        metadata = None
        if not args.disable_metadata:
            metadata = PngInfo()
            if png_embed_workflow == True:
                if prompt is not None:
                    metadata.add_text("prompt", json.dumps(prompt))
                if extra_pnginfo is not None:
                    for x in extra_pnginfo:
                        metadata.add_text(x, json.dumps(extra_pnginfo[x]))

        exif_metadata_A11 = None
        try:
            if 'positive' in image_metadata and 'negative' in image_metadata:
                a11samplername = exif_data_checker.comfy_samplers2a11(image_metadata['sampler'], image_metadata['scheduler'])
                image_metadata['vae'] = 'Baked VAE'
                if 'model_hash' not in image_metadata:
                    image_metadata['model_hash'] = 'unknown'
                    if 'model' in image_metadata:
                        checkpointpaths = folder_paths.get_folder_paths("checkpoints")[0]
                        model_full_path = checkpointpaths + os.sep + image_metadata['model']
                        is_link = os.path.islink(str(model_full_path))
                        if is_link == False:
                            image_metadata['model_hash'] = exif_data_checker.get_model_hash(model_full_path)

                if 'is_sdxl' not in image_metadata:
                    image_metadata['vae'] = 'Baked VAE'
                else:
                    if image_metadata['is_sdxl'] == 1:
                        image_metadata['vae'] = image_metadata['vae_name_sdxl']
                    else:
                        image_metadata['vae'] = image_metadata['vae_name_sd']
                    if image_metadata['vae'] is None:
                        image_metadata['vae'] = 'Baked VAE'

                exif_metadata_A11 = f"""{image_metadata['positive']}
Negative prompt: {image_metadata['negative']}
Steps: {str(image_metadata['steps'])}, Sampler: {a11samplername}, CFG scale: {str(image_metadata['cfg'])}, Seed: {str(image_metadata['seed'])}, Size: {str(image_metadata['width'])}x{str(image_metadata['height'])}, Model hash: {image_metadata['model_hash']}, Model: {Path((image_metadata['model'])).stem}, VAE: {image_metadata['vae']}"""
        except Exception:
            print('Cannot save A1111 compatible data')

        try:
            output_file = os.path.abspath(os.path.join(output_path, file))
            exif_metadata_json = image_metadata
            if extension == 'png':
                if png_embed_data == True:
                    metadata.add_text("gendata", json.dumps(exif_metadata_json))
                    print(f"{extension} Image file saved with description info: {output_file}")
                #img.save(output_file, pnginfo=metadata, optimize=True)

                if a1111_civitai_meta == True:
                    if exif_metadata_A11:
                        metadata.add_text("parameters", exif_metadata_A11)
                        print(f"{extension} Image file saved with A1111 info: {output_file}")
                img.save(output_file, pnginfo=metadata, optimize=True)
                print(f"{extension} Image file saved with exif: {output_file}")

            elif extension == 'webp':
                img.save(output_file, quality=quality, exif=metadata)
                print(f"{extension} Image file saved with exif: {output_file}")
            else:
                img.save(output_file, quality=quality, optimize=True)
                metadata = pyexiv2.Image(output_file)
                if image_embed_exif == True:
                    metadata.modify_exif({'Exif.Image.ImageDescription': json.dumps(exif_metadata_json)})
                    print(f"{extension} Image file saved with description exif: {output_file}")
                if a1111_civitai_meta == True and exif_metadata_A11:
                    metadata.modify_exif({'Exif.Photo.UserComment': 'charset=Unicode ' + exif_metadata_A11})
                    print(f"{extension} Image file saved with A1111 exif: {output_file}")

                if a1111_civitai_meta == False and image_embed_exif == False:
                    if extension == 'webp':
                        img.save(output_file, quality=quality, exif=metadata)
                        print(f"{extension} Image file saved without exif: {output_file}")
                    else:
                        img.save(output_file, quality=quality, optimize=True)
                        print(f"{extension} Image file saved without exif: {output_file}")

            if save_meta_to_json:
                jsonfile = os.path.splitext(output_file)[0] + '.json'
                with open(jsonfile, 'w', encoding='utf-8') as jf:
                    json.dump(exif_metadata_json, jf, ensure_ascii=False, indent=4)
                    print(f"JSON file saved with generation data: {jsonfile}")

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
            with open(infofile, 'w', encoding='utf-8', newline="") as inf:
                inf.write(saved_info)
                print(f"TXT file saved with generation data: {infofile}")

        return saved_info, {"ui": {"images": []}}

    def get_subfolder_path(self, image_path, output_path):
        output_parts = output_path.strip(os.sep).split(os.sep)
        image_parts = image_path.strip(os.sep).split(os.sep)
        common_parts = os.path.commonprefix([output_parts, image_parts])
        subfolder_parts = image_parts[len(common_parts):]
        subfolder_path = os.sep.join(subfolder_parts[:-1])
        return subfolder_path

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
            "positive": ('STRING', {"forceInput": True, "default": "Red sportcar racing on the street of metropolis"}),
            "negative": ('STRING', {"forceInput": True, "default": "Cute cat, nsfw, nude, nudity, porn"})
        }, "optional": {
            "positive_decoded": ('STRING', {"forceInput": True, "default": "Red sportcar racing on the street of metropolis"}),
            "negative_decoded": ('STRING', {"forceInput": True, "default": "Cute cat, nsfw, nude, nudity, porn"}),
            "control_data": ("TUPLE", {"default": None, "forceInput": True}),
            "seed": ('INT', {"forceInput": True, "default": 1}),
            "t5_xxl_prompt": ('STRING', {"forceInput": True}),
            "positive_l": ('STRING', {"forceInput": True}),
            "negative_l": ('STRING', {"forceInput": True}),
            "width": ('INT', {"forceInput": True, "default": 512}),
            "height": ('INT', {"forceInput": True, "default": 512}),
            "preferred": ("TUPLE", {"default": None, "forceInput": True}),
            "aesthetic_score": ('INT', {"forceInput": True, "default": 0})
        },
    }

    @classmethod
    def INPUT_TYPES(cls):
        return cls.INPUT_DICT

    def load_process_meta(self, *args, **kwargs):
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

        nested_control = data_json.pop('control_data', None)
        if nested_control and isinstance(nested_control, dict):
            data_json.update(nested_control)

        return (data_json,)

class PrimereKSampler:
    CATEGORY = TREE_OUTPUTS
    RETURN_TYPES = ("LATENT", "TUPLE")
    RETURN_NAMES = ("LATENT", "CONTROL_DATA")
    FUNCTION = "pk_sampler"

    def __init__(self):
        self.state_hash = False
        self.count = 0
        self.noise_base = 0

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL", {"forceInput": True}),
                "steps": ("INT", {"default": 20, "min": 1, "max": 10000, "forceInput": True}),
                "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0, "forceInput": True}),
                "sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"forceInput": True}),
                "scheduler_name": (comfy.samplers.KSampler.SCHEDULERS, {"forceInput": True}),
                "seed": ("INT", {"default": 42, "min": 0, "max": utility.MAX_SEED}),
                "positive": ("CONDITIONING", ),
                "negative": ("CONDITIONING", ),
                "latent_image": ("LATENT", ),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "variation_extender": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "variation_batch_step": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.5, "step": 0.01}),
                "variation_level": ("BOOLEAN", {"default": False, "label_on": "Maximize", "label_off": "Off"}),
                "device": (["DEFAULT", "GPU", "CPU"], {"default": 'DEFAULT'})
            },
            "optional": {
                "model_concept": ("STRING", {"default": "Auto", "forceInput": True}),
                "control_data": ("TUPLE", {"default": None}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT"
            }
        }

    @classmethod
    def IS_CHANGED(self, **kwargs):
        if kwargs['variation_extender'] > 0 or kwargs['device'] != 'DEFAULT' or kwargs['variation_batch_step'] > 0 or kwargs['variation_level'] == True:
            return float("NaN")

    def pk_sampler(self, model, seed, steps, cfg, sampler_name, scheduler_name, positive, negative, latent_image, extra_pnginfo, prompt, model_concept = "Auto", control_data = None, denoise=1.0, variation_extender = 0, variation_batch_step = 0, variation_level = False, model_sampling = 2.5, device = 'DEFAULT', align_your_steps = False):
        timestamp_start = time.time()
        if control_data is not None:
            align_your_steps = control_data.get('align_your_steps', align_your_steps)
            model_sampling = control_data.get('model_sampling', model_sampling)
        if control_data is not None and len(control_data) > 0 and 'exif_status' in control_data and control_data['exif_status'] == 'SUCCEED':
            if 'sampler_settings' in control_data and len(control_data['sampler_settings']) > 0 and 'setup_states' in control_data and 'sampler_setup' in control_data['setup_states']:
                if control_data['setup_states']['sampler_setup'] == True:
                    variation_batch_step = 0
                    variation_level = False
                    denoise = control_data['sampler_settings']['denoise']
                    device = control_data['sampler_settings']['device']
                    align_your_steps = control_data['sampler_settings']['align_your_steps']
                    model_sampling = control_data['sampler_settings']['model_sampling']
                    if control_data['sampler_settings']['variation_level'] == True:
                        variation_extender = control_data['sampler_settings']['noise_constant']
                    else:
                        variation_extender = control_data['sampler_settings']['variation_extender_original']

        if control_data is not None and 'refiner' in control_data and control_data['refiner'] == True and 'refiner_sampling_denoise' in control_data:
            denoise = control_data['refiner_sampling_denoise']

        # samples_out = latent_image
        # out = latent_image.copy()
        variation_extender_original = variation_extender
        variation_batch_step_original = variation_batch_step
        variation_limit = 0.12

        def check_state(self, extra_pnginfo, prompt):
            old = self.state_hash
            self.state_hash = utility.collect_state(extra_pnginfo, prompt)
            if self.state_hash == old:
                self.count += 1
                return self.count
            self.count = 0
            return self.count
        batch_counter = int(check_state(self, extra_pnginfo, prompt)) + 1

        def new_state():
            random.seed(datetime.datetime.now().timestamp())
            return random.randint(1000, utility.MAX_SEED)

        def get_noise_extender(variation_limit, state_random):
            random.seed(state_random)
            noise_extender_low = round(random.uniform(0.00, variation_limit), 2)
            noise_extender_high = round(random.uniform((1 - variation_limit), 1), 2)
            noise_extender = random.choice([noise_extender_low, noise_extender_high])
            return noise_extender

        state_random = int(new_state())
        noise_extender_ksampler = get_noise_extender(variation_limit, state_random)

        random.seed(state_random)
        noise_extender_cascade = round(random.uniform((1 - (variation_limit + 0.4)), 1), 2)

        if variation_batch_step_original > 0:
            if batch_counter > 0:
                variation_batch_step = variation_batch_step_original * batch_counter
            variation_extender = round(variation_extender_original + variation_batch_step, 2)
            noise_extender_ksampler = variation_extender
            noise_extender_cascade = variation_extender

        elif variation_batch_step_original == 0 and variation_extender_original > 0:
            variation_extender = variation_extender_original
            if batch_counter > 1:
                variation_extender = variation_extender_original + (batch_counter / 100)
            noise_extender_ksampler = variation_extender
            noise_extender_cascade = variation_extender

        if variation_extender > 1:
            random.seed(batch_counter)
            variation_extender = round(random.uniform((1 - variation_limit), 1), 2)
            noise_extender_ksampler = variation_extender
            noise_extender_cascade = variation_extender

        noise_constant = noise_extender_ksampler
        WORKFLOWDATA = extra_pnginfo['workflow']['nodes']

        refiner_model_data = None
        if isinstance(model, dict) and 'main' in model:
            refiner_model_data = model.get('refiner')
            model = model['main']
        refiner_cond_pos = None
        refiner_cond_neg = None
        if isinstance(positive, dict) and 'main' in positive:
            refiner_cond_pos = positive.get('refiner')
            positive = positive['main']
        if isinstance(negative, dict) and 'main' in negative:
            refiner_cond_neg = negative.get('refiner')
            negative = negative['main']

        match model_concept:
            case 'SANA1024' | 'SANA512':
                if scheduler_name == 'flow_dpm-solver':
                    device = model['device']
                    samples_out = primeresamplers.PSamplerSana(self, device, seed, model,
                                                               steps, cfg, sampler_name, scheduler_name,
                                                               positive, negative,
                                                               latent_image, denoise,
                                                               variation_extender, variation_batch_step_original,
                                                               batch_counter, variation_extender_original,
                                                               variation_batch_step, variation_level, variation_limit,
                                                               align_your_steps, noise_extender_ksampler, WORKFLOWDATA, prompt)[0]
                else:
                    if device == 'DEFAULT':
                        device = model_management.get_torch_device()
                    latentWidth, latentHeigth = utility.getLatentSize(latent_image)

                    latent = torch.zeros([1, 32, (latentHeigth * 8) // 32, (latentWidth * 8) // 32], device=device)
                    latent_image = {"samples": latent}
                    samples_out = primeresamplers.PKSampler(self, device, seed, model,
                                        steps, cfg, sampler_name, scheduler_name,
                                        positive, negative,
                                        latent_image, denoise,
                                        variation_extender, variation_batch_step_original, batch_counter, variation_extender_original, variation_batch_step, variation_level, variation_limit,
                                        align_your_steps, noise_extender_ksampler, None)[0]

                    try:
                        comfy.model_management.soft_empty_cache()
                        comfy.model_management.cleanup_models(True)
                    except Exception:
                        print('No need to clear cache...')

            case "PixartSigma":
                samples_out = primeresamplers.PSamplerPixart(self, device, seed, model,
                                                             steps, cfg, sampler_name, scheduler_name,
                                                             positive, negative,
                                                             latent_image, denoise,
                                                             variation_extender, variation_batch_step_original,
                                                             batch_counter, variation_extender_original,
                                                             variation_batch_step, variation_level, variation_limit,
                                                             align_your_steps, noise_extender_ksampler, control_data)[0]
            case "KwaiKolors":
                samples_out = primeresamplers.PSamplerKOROLS(self, model, seed, cfg, positive, negative, latent_image, steps, denoise, sampler_name, scheduler_name, model_sampling, 1000)[0]
            case "SD3":
                samples_out = primeresamplers.PSamplerSD3(self, model, seed, cfg, positive, negative, latent_image, steps, denoise, sampler_name, scheduler_name, model_sampling, 1000)[0]
            case "Turbo":
                samples_out = primeresamplers.PTurboSampler(model, seed, cfg, positive, negative, latent_image, steps, denoise, sampler_name)[0]
            case "StableCascade":
                align_your_steps = False
                noise_constant = noise_extender_cascade
                samples_out = primeresamplers.PCascadeSampler(self, model, seed, steps, cfg, sampler_name, scheduler_name, positive, negative, latent_image, denoise, device, variation_level, variation_limit, variation_extender_original, variation_batch_step_original, variation_extender, variation_batch_step, batch_counter, noise_extender_cascade)[0]
            case "Hyper":
                CONCEPT_SELECTOR = control_data.get('model_concept') if control_data else None
                OriginalBaseModel = control_data.get('model_name') if control_data else None
                fullpathFile = folder_paths.get_full_path('checkpoints', OriginalBaseModel)
                is_link = os.path.islink(str(fullpathFile))
                HYPERSD_SELECTOR = 'UNET' if is_link else 'LORA'
                HYPERSD_SAMPLER = True

                if model_concept == 'Hyper' and (CONCEPT_SELECTOR == 'Hyper' or CONCEPT_SELECTOR == 'Auto') and steps == 12 and HYPERSD_SELECTOR == 'LORA' and HYPERSD_SAMPLER == True:
                    cfg = float(control_data.get('cfg', 3.80))
                    scheduler_name = control_data.get('scheduler_name', "normal")
                samples_out = primeresamplers.PSamplerHyper(self, extra_pnginfo, model, seed, steps, cfg, positive, negative, sampler_name, scheduler_name, latent_image, denoise, prompt, control_data)[0]

            case 'QwenGen' | 'QwenEdit':
                align_your_steps = False
                samples_out = primeresamplers.PKSampler(self, device, seed, model,
                                                        steps, cfg, sampler_name, scheduler_name,
                                                        positive, negative,
                                                        latent_image, denoise,
                                                        0, variation_batch_step_original, batch_counter, variation_extender_original, 0, False, variation_limit,
                                                        align_your_steps, noise_extender_ksampler, None)[0]

            case 'Chroma':
                align_your_steps = False
                samples_out = primeresamplers.PSamplerChroma(self, model, seed, cfg, positive, negative, scheduler_name, sampler_name, steps, denoise, latent_image)[0]

            case 'Z-Image':
                align_your_steps = False
                model = nodes_model_advanced.ModelSamplingSD3.patch(self, model, 2.8, 1.0)[0]
                negative = nodes.ConditioningZeroOut.zero_out(self, negative)[0]
                samples_out = primeresamplers.PKSampler(self, device, seed, model,
                                                        steps, cfg, sampler_name, scheduler_name,
                                                        positive, negative,
                                                        latent_image, denoise,
                                                        variation_extender, variation_batch_step_original, batch_counter, variation_extender_original, variation_batch_step, variation_level, variation_limit,
                                                        align_your_steps, noise_extender_ksampler, None)[0]

            case 'Flux':
                FLUX_SAMPLER = control_data.get('sampler', 'ksampler') if control_data else 'ksampler'
                FLUX_GUIDANCE = float(control_data.get('guidance', 3.5)) if control_data else 3.5
                align_your_steps = False
                if FLUX_SAMPLER == 'custom_advanced':
                    samples_out = primeresamplers.PSamplerAdvanced(self, model, seed, FLUX_GUIDANCE, positive, scheduler_name, sampler_name, steps, denoise, latent_image)[0]
                elif FLUX_SAMPLER == 'ksampler':
                    CONDITIONING_POS = nodes_flux.FluxGuidance.execute(positive, FLUX_GUIDANCE)[0] if FLUX_GUIDANCE > 0 else positive
                    if control_data is not None and float(control_data.get('cfg', 2.0)) < 1.2:
                        CONDITIONING_NEG = CONDITIONING_POS
                    else:
                        CONDITIONING_NEG = nodes_flux.FluxGuidance.execute(negative, FLUX_GUIDANCE)[0]
                    samples_out = primeresamplers.PKSampler(self, device, seed, model,
                                                            steps, cfg, sampler_name, scheduler_name,
                                                            CONDITIONING_POS, CONDITIONING_NEG,
                                                            latent_image, denoise,
                                                            variation_extender, variation_batch_step_original, batch_counter, variation_extender_original, variation_batch_step, variation_level, variation_limit,
                                                            align_your_steps, noise_extender_ksampler, None, control_data)[0]
                else:
                    samples_out = primeresamplers.PKSampler(self, device, seed, model,
                                                            steps, cfg, sampler_name, scheduler_name,
                                                            positive, negative,
                                                            latent_image, denoise,
                                                            variation_extender, variation_batch_step_original, batch_counter, variation_extender_original, variation_batch_step, variation_level, variation_limit,
                                                            align_your_steps, noise_extender_ksampler, None, control_data)[0]

            case _:
                if model_concept == 'AuraFlow' and model_sampling is not None and model_sampling > 0:
                    model = nodes_model_advanced.ModelSamplingSD3.patch(self, model, model_sampling, 1.0)[0]
                samples_out = primeresamplers.PKSampler(self, device, seed, model,
                                                        steps, cfg, sampler_name, scheduler_name,
                                                        positive, negative,
                                                        latent_image, denoise,
                                                        variation_extender, variation_batch_step_original, batch_counter, variation_extender_original, variation_batch_step, variation_level, variation_limit,
                                                        align_your_steps, noise_extender_ksampler, None, control_data)[0]

        if refiner_model_data is not None:
            samples_out = primeresamplers.run_refiner_pass(self, refiner_model_data, refiner_cond_pos, refiner_cond_neg, samples_out, control_data, seed)[0]

        if control_data is not None:
            control_data['sampler_settings'] = {}
            control_data['sampler_settings']['denoise'] = denoise
            control_data['sampler_settings']['variation_extender_original'] = variation_extender_original
            control_data['sampler_settings']['variation_batch_step_original'] = variation_batch_step_original
            control_data['sampler_settings']['variation_level'] = variation_level
            control_data['sampler_settings']['device'] = device
            control_data['sampler_settings']['align_your_steps'] = align_your_steps
            control_data['sampler_settings']['noise_constant'] = noise_constant
            control_data['sampler_settings']['variation_seed'] = seed
            control_data['sampler_settings']['batch_counter'] = batch_counter
            control_data['sampler_settings']['model_sampling'] = model_sampling

        timestamp_diff = int(time.time() - timestamp_start)
        is_random_model = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereVisualCKPT', 'random_model', prompt)
        selected_model = control_data.get('model_name') if control_data else utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereVisualCKPT', 'base_model', prompt)
        if is_random_model == True:
            fullSource = PrimereVisualCKPT.allModels
            slashIndex = selected_model.find('\\')
            if slashIndex > 0:
                subdirType = selected_model[0: slashIndex] + '\\'
                models_by_path = list(filter(lambda x: x.startswith(subdirType), fullSource))
                random.seed(seed)
                selected_model = random.choice(models_by_path)

        if selected_model is not None:
            modelname_only = Path(selected_model).stem
            model_samplingtime = utility.get_value_from_cache('model_samplingtime', modelname_only)
            if model_samplingtime is None:
                utility.add_value_to_cache('model_samplingtime', modelname_only, '1|' + str(timestamp_diff))
            else:
                model_samplingtime_list = model_samplingtime.split("|")
                counter = str(int(model_samplingtime_list[0]) + 1)
                diffvalue = str(int(model_samplingtime_list[1]) + timestamp_diff)
                utility.add_value_to_cache('model_samplingtime', modelname_only, counter + '|' + diffvalue)

        return (samples_out, control_data)

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
                "image_quality": ("INT",  {"default": 95, "min": 10, "max": 100, "step": 5}),
                "preview_target": (['Checkpoint', 'CSV Prompt', 'Lora', 'Lycoris', 'Hypernetwork', 'Embedding'],),
                "preview_save_mode": (['Overwrite', 'Keep', 'Join horizontal', 'Join vertical'], {"default": "Overwrite"}),
                "embed_metadata": ("BOOLEAN", {"default": False, "label_on": "Embed metadata", "label_off": "No metadata"}),
                "auto_save_path": ("BOOLEAN", {"default": False, "label_on": "Comfy output folder", "label_off": "Temp folder, will be deleted"}),
            },
            "optional": {
                "images": ("IMAGE", {"default": None}),
                "image_metadata": ('TUPLE', {"forceInput": True}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT",
                "image_path": (cls.image_path,),
                "id": "UNIQUE_ID",
            },
        }

    def preview_img_saver(self, image_save_as, image_type, image_resize, image_quality, preview_target, preview_save_mode, embed_metadata, auto_save_path = False, images=None, image_metadata=None, **kwargs):
        if auto_save_path:
            self.output_dir = folder_paths.get_output_directory()
            self.type = "output"

        if images is None or type(images).__name__ != "Tensor":
            INVALID_IMAGE_PATH = os.path.join(PRIMERE_ROOT, 'front_end', 'images')
            INVALID_IMAGE = os.path.join(INVALID_IMAGE_PATH, "invalid.jpg")
            images = utility.ImageLoaderFromPath(INVALID_IMAGE)

        VISUAL_NODE_NAMES = ['PrimereVisualCKPT', 'PrimereVisualLORA', 'PrimereVisualEmbedding', 'PrimereVisualHypernetwork', 'PrimereVisualLYCORIS', 'PrimereVisualStyle']
        VISUAL_NODE_FILENAMES = ['PrimereVisualCKPT', 'PrimereVisualLORA', 'PrimereVisualEmbedding', 'PrimereVisualHypernetwork', 'PrimereVisualLYCORIS']
        WIDGET_DATA = {
            "PrimereVisualCKPT": ['base_model'],
            "PrimereVisualStyle": ['styles'],
            "PrimereVisualLORA": ['lora_1', 'lora_2', 'lora_3', 'lora_4', 'lora_5', 'lora_6'],
            "PrimereVisualEmbedding": ['embedding_1', 'embedding_2', 'embedding_3', 'embedding_4', 'embedding_5', 'embedding_6'],
            "PrimereVisualHypernetwork": ['hypernetwork_1', 'hypernetwork_2', 'hypernetwork_3', 'hypernetwork_4', 'hypernetwork_5', 'hypernetwork_6'],
            "PrimereVisualLYCORIS": ['lycoris_1', 'lycoris_2', 'lycoris_3', 'lycoris_4', 'lycoris_5', 'lycoris_6'],
        }

        WORKFLOWDATA = kwargs['extra_pnginfo']['workflow']['nodes']
        VISUAL_DATA = {}
        for NODE_ITEMS in WORKFLOWDATA:
            ITEM_TYPE = NODE_ITEMS['type']
            if ITEM_TYPE in VISUAL_NODE_NAMES and ITEM_TYPE in WIDGET_DATA:
                REQUIRED_DATA_NAMES = WIDGET_DATA[ITEM_TYPE]
                if len(REQUIRED_DATA_NAMES) > 0:
                    VALUE_LIST = []
                    VALUE_LIST_ORIGINAL = []
                    WIDGET_STATE = True
                    for DATA_NAME in REQUIRED_DATA_NAMES:
                        if type(DATA_NAME).__name__ == 'str':
                            WIDGET_VALUE_ORIGINAL = utility.getDataFromWorkflowByName(WORKFLOWDATA, ITEM_TYPE, DATA_NAME, kwargs['prompt'])
                            if DATA_NAME[-1].isdigit():
                                USE_WIDGET_NAME = 'use_' + DATA_NAME
                                WIDGET_STATE = utility.getDataFromWorkflowByName(WORKFLOWDATA, ITEM_TYPE, USE_WIDGET_NAME, kwargs['prompt'])

                            if ITEM_TYPE in VISUAL_NODE_FILENAMES:
                                REPLACED_WIDGETS = Path(WIDGET_VALUE_ORIGINAL).stem.replace(' ', '_')
                            else:
                                REPLACED_WIDGETS = WIDGET_VALUE_ORIGINAL.replace(' ', '_')

                            if WIDGET_STATE == True and REPLACED_WIDGETS not in VALUE_LIST:
                                VALUE_LIST.append(REPLACED_WIDGETS)
                                VALUE_LIST_ORIGINAL.append(WIDGET_VALUE_ORIGINAL)

                    VISUAL_DATA[ITEM_TYPE] = VALUE_LIST
                    VISUAL_DATA[ITEM_TYPE + '_ORIGINAL'] = VALUE_LIST_ORIGINAL

        extra_pnginfo_embed = None
        prompt_embed = None
        if embed_metadata and not args.disable_metadata:
            prompt_embed = kwargs.get('prompt')
            if image_metadata is not None and isinstance(image_metadata, dict):
                try:
                    extra_pnginfo_embed = {"gendata": image_metadata}
                except Exception:
                    pass
            elif image_metadata is None and kwargs.get('extra_pnginfo') is not None:
                extra_pnginfo_embed = kwargs.get('extra_pnginfo')

        if auto_save_path:
            results = nodes.SaveImage.save_images(self, images, filename_prefix="Primere_ComfyUI", prompt=prompt_embed, extra_pnginfo=extra_pnginfo_embed)
        else:
            r1 = random.randint(1000, 9999)
            temp_filename = f"Primere_ComfyUI_{r1}.png"
            os.makedirs(folder_paths.get_temp_directory(), exist_ok=True)
            TEMP_FILE = os.path.join(folder_paths.get_temp_directory(), temp_filename)
            utility.tensor_to_image(images[0]).save(TEMP_FILE)
            results = {"ui": {"images": [{"filename": temp_filename, "subfolder": "", "type": "temp"}]}, "result": ()}

        ui_images = results.get('ui', {}).get('images', [])
        VISUAL_DATA['SaveImages'] = ui_images
        VISUAL_DATA['node_id'] = kwargs.get('id')
        VISUAL_DATA['ImagePath'] = folder_paths.get_output_directory() if auto_save_path else folder_paths.get_temp_directory()
        PromptServer.instance.send_sync("getVisualTargets", VISUAL_DATA)

        return results

class PrimereAestheticCKPTScorer:
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
                "control_data": ('TUPLE', {"forceInput": True}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT"
            },
        }

    def aesthetic_scorer(self, image, get_aesthetic_score, add_to_checkpoint, add_to_saved_prompt, prompt, dual_mode = True, control_data = None, **kwargs):
        final_prediction = '*** Aesthetic scorer off ***'
        models = []
        AE_SCORE_MIN = None
        AE_SCORE_MAX = None
        if 'extra_pnginfo' in kwargs:
            WORKFLOWDATA = kwargs['extra_pnginfo']['workflow']['nodes']
            AE_SCORE_MIN = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereVisualCKPT', 'aescore_percent_min', prompt)
            AE_SCORE_MAX = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereVisualCKPT', 'aescore_percent_max', prompt)

        def pipe(model):
            return pipeline(task="image-classification", model=model, device=model_management.get_torch_device())

        if (get_aesthetic_score == True):
            AE_MODEL_ROOT = os.path.join(folder_paths.models_dir, 'aesthetic')
            AEMODELS_ENCODERS_PATHS = utility.getValidAscorerPaths(AE_MODEL_ROOT)
            if len(AEMODELS_ENCODERS_PATHS) > 0:
                if 'cafe_aesthetic' not in AEMODELS_ENCODERS_PATHS:
                    final_prediction = '*** Missing aesthetic model ***'
                if 'cafe_style' not in AEMODELS_ENCODERS_PATHS:
                    final_prediction = '*** Missing style model ***'
                if 'cafe_style' in AEMODELS_ENCODERS_PATHS and 'cafe_aesthetic' in AEMODELS_ENCODERS_PATHS:
                    ae_model_access = os.path.join(AE_MODEL_ROOT, 'cafe_aesthetic')
                    style_model_access = os.path.join(AE_MODEL_ROOT, 'cafe_style')
                    if os.path.isdir(ae_model_access) == True and os.path.isdir(style_model_access) == True:
                        if dual_mode == True:
                            models.append({"pipe": pipe(ae_model_access), "weights": [0.0, 1.0], })
                            models.append({"pipe": pipe(style_model_access), "weights": [1.0, 0.75, 0.5, 0.0, 0.0], })
                            final_divider = 2
                        else:
                            models.append({"pipe": pipe(ae_model_access), "weights": [0.0, 1.0], })
                            final_divider = 1
                        try:
                            count = 1
                            pil_images = image.permute(0, 3, 1, 2)
                            pil_images = torch.clamp(pil_images * 255, 0, 255)
                            pil_images = pil_images.to("cpu", torch.uint8)
                            pil_images = [TF.to_pil_image(i) for i in pil_images]
                            scores = {i: 0.0 for i in range(image.shape[0])}
                            for model in models:
                                pipe = model["pipe"]
                                weights = model["weights"]
                                labels = pipe.model.config.id2label
                                w_len = len(weights)
                                w_sum = sum(weights)
                                w_map = {labels[i]: weights[i] for i in range(w_len)}
                                values = pipe(pil_images, top_k=w_len)
                                for index, value in enumerate(values):
                                    score = [v["score"] * w_map[v["label"]] for v in value]
                                    scores[index] += sum(score) / w_sum
                            scores = sorted(scores.items(), key=lambda k: k[1], reverse=True)[:count]
                            final_score = ", ".join([f"{v:.3f}" for k, v in scores])
                            final_prediction = int((float(final_score) * 1000) / final_divider)
                        except Exception:
                            final_prediction = '*** Invalid input image ***'
                    else:
                        final_prediction = '*** No aesthetic models downloaded ***'
            else:
                final_prediction = '*** No aesthetic models downloaded ***'

            if type(final_prediction) != str:
                final_prediction = str(final_prediction)

            if control_data is not None and final_prediction.isdigit():
                if add_to_checkpoint == True and (control_data['model_concept']):
                    if 'model_name' in control_data:
                        AE_SCORE_MIN = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereVisualCKPT', 'aescore_percent_min', prompt)
                        AE_SCORE_MAX = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereVisualCKPT', 'aescore_percent_max', prompt)
                        selected_model = control_data['model_name']
                        modelname_only = Path(selected_model).stem
                        model_ascore = utility.get_value_from_cache('model_ascores', modelname_only)
                        if model_ascore is None:
                            utility.add_value_to_cache('model_ascores', modelname_only, '1|' + final_prediction)
                        else:
                            model_ascore_list = model_ascore.split("|")
                            counter = str(int(model_ascore_list[0]) + 1)
                            score = str(int(model_ascore_list[1]) + int(final_prediction))
                            utility.add_value_to_cache('model_ascores', modelname_only, counter + '|' + score)

                if add_to_saved_prompt == True and final_prediction.isdigit():
                    if 'positive' in control_data:
                        selectedStyle = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereVisualStyle', 'styles', prompt)
                        if selectedStyle is None:
                            selectedStyle = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereStyleLoader', 'styles', prompt)

                        is_random_style = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereVisualStyle', 'random_prompt', prompt)
                        if is_random_style == True:
                            styles_csv = PrimereVisualStyle.styles_csv
                            seed = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereFastSeed', 'seed', prompt)
                            random.seed(seed)
                            styleKey = styles_csv['name'] == selectedStyle
                            try:
                                preferred_subpath = styles_csv[styleKey]['preferred_subpath'].values[0]
                            except Exception:
                                preferred_subpath = ''
                            if str(preferred_subpath) == "nan":
                                resultsBySubpath = styles_csv[styles_csv['preferred_subpath'].isnull()]
                            else:
                                resultsBySubpath = styles_csv[styles_csv['preferred_subpath'] == preferred_subpath]
                            selectedStyle = random.choice(list(resultsBySubpath['name']))

                        if selectedStyle is not None:
                            STYLE_DIR = os.path.join(PRIMERE_ROOT, 'stylecsv')
                            STYLE_FILE = os.path.join(STYLE_DIR, "styles.csv")
                            try:
                                STYLE_FILE_EXAMPLE = os.path.join(STYLE_DIR, "styles.example.csv")
                            except Exception:
                                STYLE_FILE_EXAMPLE = STYLE_FILE

                            if Path(STYLE_FILE).is_file() == True:
                                STYLE_SOURCE = STYLE_FILE
                            else:
                                STYLE_SOURCE = STYLE_FILE_EXAMPLE
                            style_data = utility.load_external_csv(STYLE_SOURCE, 0)
                            positive_prompt = style_data[style_data['name'] == selectedStyle]['prompt'].values[0]
                            if (positive_prompt is not None):
                                if len(positive_prompt) > 100:
                                    positive_prompt = positive_prompt[:100]
                                if positive_prompt in control_data['positive']:
                                    AE_SCORE_MIN = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereVisualStyle', 'aescore_percent_min', prompt)
                                    AE_SCORE_MAX = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereVisualStyle', 'aescore_percent_max', prompt)
                                    style_ascore = utility.get_value_from_cache('styles_ascores', selectedStyle)
                                    if style_ascore is None:
                                        utility.add_value_to_cache('styles_ascores', selectedStyle, '1|' + final_prediction)
                                    else:
                                        style_ascore_list = style_ascore.split("|")
                                        counter = str(int(style_ascore_list[0]) + 1)
                                        score = str(int(style_ascore_list[1]) + int(final_prediction))
                                        utility.add_value_to_cache('styles_ascores', selectedStyle, counter + '|' + score)

        result_int = int(final_prediction) if final_prediction.isdigit() else 0
        if isinstance(final_prediction, str) and final_prediction.isdigit():
            if AE_SCORE_MIN is not None and AE_SCORE_MAX is not None:
                final_prediction = max(0, min(100, int(((int(final_prediction) - AE_SCORE_MIN) / (AE_SCORE_MAX - AE_SCORE_MIN)) * 100)))
            else:
                final_prediction = int(final_prediction)
        else:
            final_prediction = '*** Aesthetic scorer error ***'

        return {"ui": {"text": [f'{result_int} / {final_prediction}%']}, "result": (result_int,)}

class DebugToFile():
    CATEGORY = TREE_OUTPUTS
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("PROMPT+",)
    FUNCTION = "debug_to_file"

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "input_prompt": ("STRING", {"default": "", "forceInput": True}),
                "enhanced_prompt": ("STRING", {"default": "", "forceInput": True}),
                "seed": ("INT", {"default": 1, "forceInput": True}),
            },
            "hidden": {
                "extra_pnginfo": "EXTRA_PNGINFO",
                "prompt": "PROMPT"
            },
        }

    def debug_to_file(self, **kwargs):
        WORKFLOWDATA = kwargs['extra_pnginfo']['workflow']['nodes']
        prompt = kwargs['prompt']

        LLM_NAME = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereLLMEnhancer', 'llm_model_path', prompt)
        LLM_CONF = utility.getDataFromWorkflowByName(WORKFLOWDATA, 'PrimereLLMEnhancer', 'configurator', prompt)

        json_dir = os.path.join(PRIMERE_ROOT, 'json')
        json_file = os.path.join(json_dir, 'llm_autotest.json')

        cacheData = {LLM_NAME: {LLM_CONF: [{"input_prompt": kwargs['input_prompt'], "enhanced_prompt": kwargs['enhanced_prompt'], "seed": kwargs['seed']}]}}
        json_object = json.dumps(cacheData, indent=4)
        ifJsonExist = os.path.isfile(json_file)
        if ifJsonExist == True:
            with open(json_file, 'r') as openfile:
                saved_cache = json.load(openfile)
                if LLM_NAME in saved_cache and LLM_CONF in saved_cache[LLM_NAME]:
                    cacheData = [{"input_prompt": kwargs['input_prompt'], "enhanced_prompt": kwargs['enhanced_prompt'], "seed": kwargs['seed']}]
                    saved_cache[LLM_NAME][LLM_CONF].append(cacheData)
                elif LLM_NAME in saved_cache and LLM_CONF not in saved_cache[LLM_NAME]:
                    cacheData = {LLM_CONF: [{"input_prompt": kwargs['input_prompt'], "enhanced_prompt": kwargs['enhanced_prompt'], "seed": kwargs['seed']}]}
                    saved_cache[LLM_NAME].update(cacheData)
                else:
                    saved_cache.update(cacheData)
                newJsonObject = json.dumps(saved_cache, indent=4)
                with open(json_file, "w", encoding='utf-8') as outfile:
                    outfile.write(newJsonObject)
        else:
            with open(json_file, "w", encoding='utf-8') as outfile:
                outfile.write(json_object)

        return (LLM_NAME,)
