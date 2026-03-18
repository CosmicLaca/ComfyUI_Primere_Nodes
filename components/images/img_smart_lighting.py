import numpy as np
from PIL import Image


def img_smart_lighting(image: Image.Image, intensity: float = 0) -> Image.Image:
    if intensity == 0:
        return image.convert("RGB")

    if not (0 <= intensity <= 100):
        raise ValueError(f"intensity must be 0 … 100, got {intensity}")

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0

    t        = np.linspace(0.0, 1.0, 256, dtype=np.float64)
    strength = intensity / 100.0

    shadow_gamma    = max(1.0 - strength * 0.55, 0.35)
    shadow_curve    = np.power(np.clip(t, 0, 1), shadow_gamma)

    highlight_gamma = 1.0 + strength * 0.40
    highlight_curve = np.power(np.clip(t, 0, 1), highlight_gamma)

    w        = 1.0 - t ** 1.5
    combined = w * shadow_curve + (1.0 - w) * highlight_curve

    lut     = np.clip((1.0 - strength) * t + strength * combined, 0.0, 1.0)
    indices = np.clip((arr * 255).astype(np.int32), 0, 255)
    result  = lut[indices].astype(np.float32)

    return Image.fromarray((result * 255).astype(np.uint8), mode="RGB")
