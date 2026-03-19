import numpy as np
from PIL import Image


def img_selective_tone(
    image:         Image.Image,
    channels_data: dict,
    separation:    float = 50,
    strength:      float = 0.5,
) -> Image.Image:
    highlights = channels_data.get('highlights', 0)
    midtones   = channels_data.get('midtones',   0)
    shadows    = channels_data.get('shadows',    0)
    blacks     = channels_data.get('blacks',     0)

    if highlights == 0 and midtones == 0 and shadows == 0 and blacks == 0:
        return image.convert("RGB")

    if not (0 <= separation <= 100):
        raise ValueError(f"separation must be 0 … 100, got {separation}")

    if not (0.0 <= strength <= 1.0):
        raise ValueError(f"strength must be 0.0 … 1.0, got {strength}")

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0

    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

    t           = separation / 100.0
    width_mid   = 0.06 + t * 0.26
    width_black = 0.05 + t * 0.17

    def bell(x, centre, width):
        return np.exp(-0.5 * ((x - centre) / width) ** 2)

    mask_blacks     = bell(lum, 0.00, width_black)
    mask_shadows    = bell(lum, 0.25, width_mid)
    mask_midtones   = bell(lum, 0.50, width_mid)
    mask_highlights = bell(lum, 0.80, width_mid)

    SCALE  = 0.10 + strength * 0.50

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
