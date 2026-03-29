import numpy as np
from PIL import Image
from scipy.ndimage import minimum_filter, gaussian_filter


def _estimate_atmospheric_light(arr: np.ndarray) -> np.ndarray:
    flat = arr.reshape(-1, 3)
    brightest = flat[np.argmax(np.sum(flat, axis=1))]
    return brightest


def img_dehaze(
    image: Image.Image,
    strength: float = 0.7,
    radius: int = 15,
    omega: float = 0.95,
    t0: float = 0.1,
    contrast: float = 1.05,
    precision: bool = False,
) -> Image.Image:

    img = image.convert("RGB")

    if precision:
        max_val = 65535.0
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr * max_val
        arr = arr / max_val
    else:
        arr = np.array(img, dtype=np.float32) / 255.0

    dark = np.min(arr, axis=2)
    dark = minimum_filter(dark, size=radius)
    A = _estimate_atmospheric_light(arr)
    transmission = 1.0 - omega * dark
    transmission = np.clip(transmission, t0, 1.0)
    transmission = gaussian_filter(transmission, sigma=radius * 0.25)
    J = (arr - A) / transmission[..., None] + A
    out = arr * (1.0 - strength) + J * strength
    out = (out - 0.5) * contrast + 0.5
    out = np.clip(out, 0.0, 1.0)
    out = (out * 255.0).astype(np.uint8)

    return Image.fromarray(out, mode="RGB")