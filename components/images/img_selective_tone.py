import numpy as np
from PIL import Image


def img_selective_tone(image: Image.Image, highlights: float = 0, midtones: float = 0, shadows: float = 0, blacks: float = 0) -> Image.Image:
    if highlights == 0 and midtones == 0 and shadows == 0 and blacks == 0:
        return image.convert("RGB")

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0

    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

    def bell(lum, centre, width):
        return np.exp(-0.5 * ((lum - centre) / width) ** 2)

    mask_blacks     = bell(lum, 0.00, 0.12)
    mask_shadows    = bell(lum, 0.25, 0.18)
    mask_midtones   = bell(lum, 0.50, 0.18)
    mask_highlights = bell(lum, 0.80, 0.18)

    SCALE  = 0.35
    delta  = np.zeros_like(lum)
    delta += mask_highlights * (highlights / 100.0) * SCALE
    delta += mask_midtones   * (midtones   / 100.0) * SCALE
    delta += mask_shadows    * (shadows    / 100.0) * SCALE
    delta += mask_blacks     * (blacks     / 100.0) * SCALE * 0.6

    lum_new = np.clip(lum + delta, 1e-6, 1.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        scale = np.where(lum > 1e-6, lum_new / lum, 1.0)[..., np.newaxis]

    result = np.clip(arr * scale, 0.0, 1.0)
    return Image.fromarray((result * 255).astype(np.uint8), mode="RGB")
