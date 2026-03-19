import numpy as np
from PIL import Image


def img_color_balance(
    image:               Image.Image,
    channels_data:       dict,
    preserve_luminosity: bool  = True,
    separation:          float = 50,
) -> Image.Image:
    VALID_TONES = {'shadows', 'midtones', 'highlights'}

    if not (0 <= separation <= 100):
        raise ValueError(f"separation must be 0 … 100, got {separation}")

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32)

    lum = (0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]) / 255.0

    t             = separation / 100.0
    exponent_edge = 3.0 - t * 2.5
    exponent_mid  = 1.8 - t * 1.5

    adjusted = arr.copy()
    any_applied = False

    for tone, vals in channels_data.items():
        tone = tone.strip().lower()
        if tone not in VALID_TONES or not vals:
            continue

        cr = vals.get('cyan_red',      0)
        mg = vals.get('magenta_green', 0)
        yb = vals.get('yellow_blue',   0)

        if cr == 0 and mg == 0 and yb == 0:
            continue

        if tone == 'shadows':
            mask = (1.0 - lum) ** exponent_edge
        elif tone == 'highlights':
            mask = lum ** exponent_edge
        else:
            mask = (1.0 - np.abs(2.0 * lum - 1.0)) ** exponent_mid

        shift     = np.array([cr, mg, yb], dtype=np.float32)
        adjusted += mask[:, :, np.newaxis] * shift
        any_applied = True

    if not any_applied:
        return image.convert("RGB")

    if preserve_luminosity:
        lum_before = 0.299*arr[:,:,0]      + 0.587*arr[:,:,1]      + 0.114*arr[:,:,2]
        lum_after  = 0.299*adjusted[:,:,0] + 0.587*adjusted[:,:,1] + 0.114*adjusted[:,:,2]
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(lum_after > 1e-6, lum_before / lum_after, 1.0)
        adjusted = adjusted * scale[:, :, np.newaxis]

    result = np.clip(adjusted, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="RGB")
