import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def _to_luminance(arr: np.ndarray) -> np.ndarray:
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def img_local_laplacian(
    image: Image.Image,
    sigma: float = 1.0,
    contrast: float = 1.2,
    detail: float = 1.0,
    levels: int = 8,
) -> Image.Image:

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    luma = _to_luminance(arr)
    base = gaussian_filter(luma, sigma=sigma)
    detail_layer = luma - base
    remapped = base + contrast * (base - 0.5)
    step = 1.0 / levels
    quantized = np.floor(luma / step) * step + step * 0.5
    tone = (remapped * 0.7 + quantized * 0.3)
    enhanced = tone + detail * detail_layer
    enhanced = np.clip(enhanced, 0.0, 1.0)
    scale = enhanced / (luma + 1e-6)
    out = arr * scale[..., None]
    out = np.clip(out, 0.0, 1.0)
    out = (out * 255.0).astype(np.uint8)

    return Image.fromarray(out, mode="RGB")