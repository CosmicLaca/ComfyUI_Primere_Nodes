import json

import piexif
import pyexiv2
import piexif.helper
from PIL import Image

from .exif.automatic1111 import Automatic1111
from .exif.primere import Primere
from .exif.comfyui import ComfyUI

# OopCompanion:suppressRename
class ImageExifReader:
    def __init__(self, file):
        self._raw = ""
        self._parser = {}
        self._parameter = {}
        self._tool = ""
        self.read_data(file)

    def read_data(self, file):
        def is_json(jsoninput):
            try:
                json.loads(jsoninput)
            except ValueError as e:
                return False
            return True

        with Image.open(file) as f:
            p2metadata = pyexiv2.Image(file)
            is_primere = p2metadata.read_exif()
            if 'Exif.Image.ImageDescription' in is_primere:
                primere_exif_string = is_primere.get('Exif.Image.ImageDescription').strip()
                if is_json(primere_exif_string) == True:
                    json_object = json.loads(primere_exif_string)
                    # keysList = {'positive', 'negative', 'positive_l', 'negative_l', 'positive_r', 'negative_r', 'seed', 'model_hash', 'model_name', 'sampler_name'}
                    # if not (keysList - json_object.keys()):
                    self._tool = "Primere"
                    self._parser = Primere(info=json_object)
            else:
                if f.format == "PNG":
                    if "parameters" in f.info:
                        print('A11')
                        self._tool = "Automatic1111"
                        self._parser = Automatic1111(info=f.info)
                    elif "prompt" in f.info:
                        print('Comfy')
                        self._tool = "ComfyUI"
                        self._parser = ComfyUI(info=f.info)

                elif f.format == "JPEG" or f.format == "WEBP":
                    exif = piexif.load(f.info.get("exif")) or {}
                    self._raw = piexif.helper.UserComment.load(
                        exif.get("Exif").get(piexif.ExifIFD.UserComment)
                    )
                    if is_json(self._raw) != True:
                        self._tool = "Automatic1111"
                        self._parser = Automatic1111(raw=self._raw)

    @property
    def parser(self):
        return self._parser

    @property
    def tool(self):
        return self._tool
