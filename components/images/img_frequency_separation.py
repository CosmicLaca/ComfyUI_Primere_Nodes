import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def img_frequency_separation(
    image: Image.Image,
    radius: float = 3.0,
    low_freq_strength: float = 1.0,
    high_freq_strength: float = 1.0,
    blend_mode: str = "add",
) -> Image.Image:

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    low = gaussian_filter(arr, sigma=(radius, radius, 0))
    high = arr - low
    low_mod = low * low_freq_strength
    high_mod = high * high_freq_strength
    if blend_mode == "add":
        out = low_mod + high_mod
    elif blend_mode == "multiply":
        out = low_mod * (1.0 + high_mod)
    elif blend_mode == "overlay":
        base = low_mod
        detail = high_mod
        out = np.where(
            base <= 0.5,
            2.0 * base * (1.0 + detail),
            1.0 - 2.0 * (1.0 - base) * (1.0 - detail)
        )
    else:
        out = low_mod + high_mod
    out = np.clip(out, 0.0, 1.0)
    out = (out * 255.0).astype(np.uint8)

    return Image.fromarray(out, mode="RGB")