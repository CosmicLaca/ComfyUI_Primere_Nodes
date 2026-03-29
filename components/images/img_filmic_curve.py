import numpy as np
from PIL import Image


def _filmic_curve(x, contrast, highlight_rolloff, shadow_lift, pivot):
    y = (x - pivot) * contrast + pivot
    if shadow_lift > 0:
        y = y + shadow_lift * (1.0 - y) * (1.0 - x)
    if highlight_rolloff > 0:
        y = y / (y + highlight_rolloff)

    return y


def _log_curve(x, contrast):
    eps = 1e-6
    y = np.log1p(x * 9.0) / np.log1p(9.0)
    y = (y - 0.5) * contrast + 0.5
    return y


def img_filmic_curve(
    image: Image.Image,
    curve_type: str = "filmic",
    contrast: float = 1.0,
    highlight_rolloff: float = 0.5,
    shadow_lift: float = 0.0,
    pivot: float = 0.5,
) -> Image.Image:

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    if curve_type.lower() == "log":
        out = _log_curve(arr, contrast)
    else:
        out = _filmic_curve(arr, contrast, highlight_rolloff, shadow_lift, pivot)
    out = np.clip(out, 0.0, 1.0)
    out = (out * 255.0).astype(np.uint8)

    return Image.fromarray(out, mode="RGB")