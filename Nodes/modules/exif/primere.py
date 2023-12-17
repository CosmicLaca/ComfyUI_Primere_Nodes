from ..exif.base_format import BaseFormat

class Primere(BaseFormat):
    def __init__(self, info: dict = None, raw: str = ""):
        super().__init__(info, raw)
        self._pri_format()

    def _pri_format(self):
        self._parameter = self._info