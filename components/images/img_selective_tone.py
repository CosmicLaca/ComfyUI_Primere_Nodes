import numpy as np
from PIL import Image


def img_selective_tone(
    image:      Image.Image,
    highlights: float = 0,
    midtones:   float = 0,
    shadows:    float = 0,
    blacks:     float = 0,
    separation: float = 50,
    strength:   float = 0.5,
) -> Image.Image:
    """
    DxO-style Selective Tone — independent luminance adjustment for four
    tonal zones with controllable zone separation and strength ceiling.

    Args:
        image      : PIL Image (RGB)
        highlights : -100 … +100  (lum zone peaks at ~0.80)
        midtones   : -100 … +100  (lum zone peaks at ~0.50)
        shadows    : -100 … +100  (lum zone peaks at ~0.25)
        blacks     : -100 … +100  (lum zone peaks at ~0.00)
        separation : 0 … 100. Controls zone isolation.
                     0   = sharpest zones, minimal cross-influence.
                     50  = default balanced overlap.
                     100 = maximum overlap, zones bleed into each other.
        strength   : 0.0 … 1.0. Ceiling of what slider ±100 can do.
                     0.0 = subtle   — max luminance shift ±0.10 (~26/255)
                     0.5 = default  — max luminance shift ±0.35 (~89/255)
                     1.0 = dramatic — max luminance shift ±0.60 (~153/255)
    Returns:
        PIL Image (RGB)
    """
    if highlights == 0 and midtones == 0 and shadows == 0 and blacks == 0:
        return image.convert("RGB")

    if not (0 <= separation <= 100):
        raise ValueError(f"separation must be 0 … 100, got {separation}")

    if not (0.0 <= strength <= 1.0):
        raise ValueError(f"strength must be 0.0 … 1.0, got {strength}")

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0

    lum = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

    # ── Zone bell widths (separation) ─────────────────────────────────────────
    t           = separation / 100.0
    width_mid   = 0.06 + t * 0.26
    width_black = 0.05 + t * 0.17

    def bell(x, centre, width):
        return np.exp(-0.5 * ((x - centre) / width) ** 2)

    mask_blacks     = bell(lum, 0.00, width_black)
    mask_shadows    = bell(lum, 0.25, width_mid)
    mask_midtones   = bell(lum, 0.50, width_mid)
    mask_highlights = bell(lum, 0.80, width_mid)

    # ── Strength ceiling ───────────────────────────────────────────────────────
    # strength 0.0 → SCALE=0.10,  0.5 → SCALE=0.35,  1.0 → SCALE=0.60
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