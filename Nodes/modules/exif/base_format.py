class BaseFormat:
    def __init__(self, info: dict = None, raw: str = ""):
        self._info = info
        self._raw = raw
        self._parameter = {}

    @property
    def info(self):
        return self._info

    @property
    def raw(self):
        return self._raw

    @property
    def parameter(self):
        return self._parameter