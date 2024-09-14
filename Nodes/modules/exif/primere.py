from ..exif.base_format import BaseFormat
import json

class Primere(BaseFormat):
    def __init__(self, info: dict = None, raw: str = ""):
        super().__init__(info, raw)
        self._pri_format()

    def _pri_format(self):
        gendata_json = {}
        if 'gendata' in self._info:
            gendata = self._info.get("gendata") or {}
            gendata_json = json.loads(gendata)

        if len(gendata_json) > 0:
            FINAL_DICT = gendata_json
            self._parameter = FINAL_DICT
        else:
            self._parameter = self._info