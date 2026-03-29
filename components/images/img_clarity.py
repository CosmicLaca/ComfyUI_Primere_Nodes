import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def _to_luminance(arr: np.ndarray) -> np.ndarray:
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

def img_clarity(
    image: Image.Image,
    strength: float = 0.5,
    radius: float = 2.0,
    midtone_range: float = 0.5,
    edge_preservation: float = 0.8,
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

    luma = _to_luminance(arr)
    base = gaussian_filter(luma, sigma=radius)
    detail = luma - base
    mid = np.exp(-((luma - 0.5) ** 2) / (2.0 * (midtone_range ** 2)))
    edge = np.abs(detail)
    edge = edge / (edge.max() + 1e-6)
    edge_mask = edge_preservation * edge + (1.0 - edge_preservation)
    enhanced = luma + strength * detail * mid * edge_mask
    enhanced = np.clip(enhanced, 0.0, 1.0)
    scale = enhanced / (luma + 1e-6)
    out = arr * scale[..., None]
    out = np.clip(out, 0.0, 1.0)
    out = (out * 255.0).astype(np.uint8)

    return Image.fromarray(out, mode="RGB")