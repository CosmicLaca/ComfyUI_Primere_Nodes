import numpy as np
from PIL import Image


def img_posterize(
    image:         Image.Image,
    channels_data: dict,
) -> Image.Image:
    levels_r = int(channels_data.get("Red",   255))
    levels_g = int(channels_data.get("Green", 255))
    levels_b = int(channels_data.get("Blue",  255))

    levels_r = max(1, min(255, levels_r))
    levels_g = max(1, min(255, levels_g))
    levels_b = max(1, min(255, levels_b))

    if levels_r == 255 and levels_g == 255 and levels_b == 255:
        return image.convert("RGB")

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0

    def _posterize(ch_arr, levels):
        if levels >= 255:
            return ch_arr
        step = 1.0 / levels
        return np.floor(ch_arr / step) * step

    out = np.stack([
        _posterize(arr[..., 0], levels_r),
        _posterize(arr[..., 1], levels_g),
        _posterize(arr[..., 2], levels_b),
    ], axis=-1)

    return Image.fromarray(np.clip(out * 255, 0, 255).astype(np.uint8), mode="RGB")
