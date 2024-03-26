import json

import piexif
#import pyexiv2
import piexif.helper
from PIL import Image

from .exif.automatic1111 import Automatic1111
from .exif.primere import Primere
from .exif.comfyui import ComfyUI
from .exif_data_checker import check_sampler_from_exif
from .exif_data_checker import check_model_from_exif
from codecs import encode, decode

class ImageExifReader:
    def __init__(self, file):
        self._raw = ""
        self._parser = {}
        self._parameter = {}
        self._original = ""
        self._tool = ""
        self.read_data(file)

    def read_data(self, file):
        def is_json(jsoninput):
            try:
                json.loads(jsoninput)
            except ValueError as e:
                return False
            return True

def read_data(self, file):
    def is_json(jsoninput):
        try:
            json.loads(jsoninput)
        except ValueError as e:
            return False
        return True

    with Image.open(file) as f:
        # Remove pyexiv2 usage
        # p2metadata = pyexiv2.Image(file)
        # is_primere = p2metadata.read_exif()
        # if 'Exif.Image.ImageDescription' in is_primere:
        #     primere_exif_string = is_primere.get('Exif.Image.ImageDescription').strip()
        #     self._original = primere_exif_string
        #     if is_json(primere_exif_string) == True:
        #         json_object = json.loads(primere_exif_string)
        #         self._tool = "Primere"
        #         self._parser = Primere(info=json_object)

        # Use piexif to check for Primere metadata
        exif_dict = piexif.load(f.info.get("exif", b""))
        user_comment = piexif.helper.UserComment.load(exif_dict.get("Exif", {}).get(piexif.ExifIFD.UserComment, b""))
        if user_comment:
            self._original = user_comment.decode()
            if is_json(self._original):
                json_object = json.loads(self._original)
                self._tool = "Primere"
                self._parser = Primere(info=json_object)

        else:
            if f.format == "PNG":
                self._original = f.info
                if "parameters" in f.info:
                    self._tool = "Automatic1111"
                    self._parser = Automatic1111(info=f.info)
                elif "prompt" in f.info:
                    self._tool = "ComfyUI"
                    try:
                        self._parser = ComfyUI(info=f.info)
                    except Exception:
                        self._parser = {}

            elif f.format == "JPEG" or f.format == "WEBP":
                try:
                    exif = piexif.load(f.info.get("exif")) or {}
                    self._raw = piexif.helper.UserComment.load(
                        exif.get("Exif").get(piexif.ExifIFD.UserComment)
                    )
                    self._original = self._raw
                    if is_json(self._raw) != True:
                        self._tool = "Automatic1111"
                        self._parser = Automatic1111(raw=self._raw)
                except Exception:
                    print('Exif data cannot read from selected image.')

    @property
    def parser(self):
        return self._parser

    @property
    def tool(self):
        return self._tool

    @property
    def original(self):
        return self._original

def compatibility_handler(data, meta_source):
    EXCHANGE_KEYS = {'model_name': 'model', 'cfg_scale': 'cfg', 'sampler_name': 'sampler', 'scheduler_name': 'scheduler', 'vae_name': 'vae', 'dynamic_positive': 'decoded_positive', 'dynamic_negative': 'decoded_negative'}
    MODEL_HASH = 'no_hash_data'

    for datakey, dataval in data.copy().items():
        if datakey in EXCHANGE_KEYS:
            newKey = EXCHANGE_KEYS[datakey]
            data[newKey] = dataval
            del data[datakey]

        match datakey:
            case "sampler" | "sampler_name":
                # if meta_source == 'Automatic1111':
                sampler_name_exif = dataval
                samplers = check_sampler_from_exif(sampler_name_exif.lower(), None, None)
                data['sampler'] = samplers['sampler']
                data['scheduler'] = samplers['scheduler']
            case "model" | "model_name":
                model_name_exif = dataval
                if 'model_hash' in data:
                    MODEL_HASH = data['model_hash']
                data['model'] = check_model_from_exif(MODEL_HASH, model_name_exif, dataval, False)

    return data