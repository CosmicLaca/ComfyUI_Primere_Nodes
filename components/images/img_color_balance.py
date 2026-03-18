import numpy as np
from PIL import Image

def img_color_balance(image: Image.Image, cyan_red: float = 0, magenta_green: float = 0, yellow_blue: float = 0, tone: str   = 'midtones', preserve_luminosity: bool  = True,) -> Image.Image:
    VALID_TONES = {'shadows', 'midtones', 'highlights'}
    tone = tone.strip().lower()
    if tone not in VALID_TONES:
        raise ValueError(f"tone must be one of {VALID_TONES}, got '{tone}'")

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32)

    lum = (0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]) / 255.0

    if tone == 'shadows':
        mask = 1.0 - lum
        mask = mask ** 1.5
    elif tone == 'highlights':
        mask = lum
        mask = mask ** 1.5
    else:
        mask = 1.0 - np.abs(2.0 * lum - 1.0)
        mask = mask ** 0.8

    mask = mask[:, :, np.newaxis]

    delta_r =  cyan_red
    delta_g =  magenta_green
    delta_b =  yellow_blue

    shift = np.array([delta_r, delta_g, delta_b], dtype=np.float32)

    adjusted = arr + mask * shift

    if preserve_luminosity:
        lum_before = (0.299 * arr[:,:,0]      + 0.587 * arr[:,:,1]      + 0.114 * arr[:,:,2])
        lum_after  = (0.299 * adjusted[:,:,0] + 0.587 * adjusted[:,:,1] + 0.114 * adjusted[:,:,2])

        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(lum_after > 1e-6, lum_before / lum_after, 1.0)
        scale = scale[:, :, np.newaxis]

        scale_blended = 1.0 + mask[:,:,0:1] * (scale - 1.0)
        adjusted = adjusted * scale_blended

    result = np.clip(adjusted, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="RGB")
