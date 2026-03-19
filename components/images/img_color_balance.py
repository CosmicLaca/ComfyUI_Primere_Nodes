import numpy as np
from PIL import Image


def img_color_balance(
    image:               Image.Image,
    cyan_red:            float = 0,
    magenta_green:       float = 0,
    yellow_blue:         float = 0,
    tone:                str   = 'midtones',
    preserve_luminosity: bool  = True,
    separation:          float = 50,
) -> Image.Image:
    """
    Photoshop-style Color Balance with controllable zone separation.

    Args:
        image               : PIL Image (RGB)
        cyan_red            : -100 (Cyan) … +100 (Red)
        magenta_green       : -100 (Magenta) … +100 (Green)
        yellow_blue         : -100 (Yellow) … +100 (Blue)
        tone                : 'shadows' | 'midtones' | 'highlights'
        preserve_luminosity : True = restore original luminosity after shift
        separation          : 0 … 100. Controls tonal zone width.
                              0   = tightest zone, minimal bleed into
                                    adjacent tones (shadows stay in shadows)
                              50  = default balanced (previous behaviour)
                              100 = broadest zone, heavily overlaps neighbours
    Returns:
        PIL Image (RGB)
    """
    if cyan_red == 0 and magenta_green == 0 and yellow_blue == 0:
        return image.convert("RGB")

    VALID_TONES = {'shadows', 'midtones', 'highlights'}
    tone = tone.strip().lower()
    if tone not in VALID_TONES:
        raise ValueError(f"tone must be one of {VALID_TONES}, got '{tone}'")

    if not (0 <= separation <= 100):
        raise ValueError(f"separation must be 0 … 100, got {separation}")

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32)

    lum = (0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2]) / 255.0

    # ── Tone mask with separation control ─────────────────────────────────────
    #
    # The exponent controls mask steepness:
    #   higher exponent → steeper rolloff → tighter zone → less bleed
    #   lower exponent  → flatter rolloff → broader zone → more bleed
    #
    # separation=0   → exponent_edge=3.0, exponent_mid=1.8  (tight)
    # separation=50  → exponent_edge=1.5, exponent_mid=0.8  (previous default)
    # separation=100 → exponent_edge=0.5, exponent_mid=0.3  (very broad)

    t              = separation / 100.0
    exponent_edge  = 3.0 - t * 2.5    # shadows / highlights:  3.0 → 0.5
    exponent_mid   = 1.8 - t * 1.5    # midtones:              1.8 → 0.3

    if tone == 'shadows':
        mask = (1.0 - lum) ** exponent_edge
    elif tone == 'highlights':
        mask = lum ** exponent_edge
    else:  # midtones
        mask = (1.0 - np.abs(2.0 * lum - 1.0)) ** exponent_mid

    mask = mask[:, :, np.newaxis]

    # ── Apply colour shift ────────────────────────────────────────────────────
    shift    = np.array([cyan_red, magenta_green, yellow_blue], dtype=np.float32)
    adjusted = arr + mask * shift

    # ── Preserve Luminosity ───────────────────────────────────────────────────
    if preserve_luminosity:
        lum_before = 0.299*arr[:,:,0]      + 0.587*arr[:,:,1]      + 0.114*arr[:,:,2]
        lum_after  = 0.299*adjusted[:,:,0] + 0.587*adjusted[:,:,1] + 0.114*adjusted[:,:,2]
        with np.errstate(invalid="ignore", divide="ignore"):
            scale = np.where(lum_after > 1e-6, lum_before / lum_after, 1.0)
        scale_blended = 1.0 + mask[:,:,0:1] * (scale[:,:,np.newaxis] - 1.0)
        adjusted = adjusted * scale_blended

    result = np.clip(adjusted, 0, 255).astype(np.uint8)
    return Image.fromarray(result, mode="RGB")