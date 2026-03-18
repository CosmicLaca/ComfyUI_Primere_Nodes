import numpy as np
from PIL import Image


def img_selective_tone(image: Image.Image, highlights: float = 0, midtones: float = 0, shadows: float = 0, blacks: float = 0,) -> Image.Image:
    """
    DxO-style Selective Tone — independent luminance adjustment for four
    tonal zones. Each zone has a soft bell-shaped mask so adjustments
    blend naturally without hard edges between zones.

    Only luminance is affected. Hue and saturation are preserved.

    Args:
        image      : PIL Image (RGB)
        highlights : -100 … +100. Bright pixel zone  (lum ~0.65–1.0)
        midtones   : -100 … +100. Mid pixel zone     (lum ~0.3–0.7)
        shadows    : -100 … +100. Dark pixel zone    (lum ~0.05–0.45)
        blacks     : -100 … +100. Deepest shadows    (lum ~0.0–0.2)
                     Negative = crush blacks further.
                     Positive = lift blacks (reduce blocking).
    Returns:
        PIL Image (RGB)
    """
    if highlights == 0 and midtones == 0 and shadows == 0 and blacks == 0:
        return image.convert("RGB")

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, 3)  0–1

    # ── Luminance map (Rec.601) ────────────────────────────────────────────────
    lum = (0.299 * arr[...,0] + 0.587 * arr[...,1] + 0.114 * arr[...,2])  # 0–1

    # ── Zone masks ────────────────────────────────────────────────────────────
    # Each mask is 0.0 (unaffected) … 1.0 (fully affected).
    # Zones overlap softly so transitions are smooth.
    #
    #  blacks     : peaks at lum=0,    half-width ~0.15
    #  shadows    : peaks at lum=0.25, half-width ~0.20
    #  midtones   : peaks at lum=0.50, half-width ~0.20
    #  highlights : peaks at lum=0.80, half-width ~0.20

    def bell(lum, centre, width):
        """Smooth bell curve, 1.0 at centre, falls off with given half-width."""
        return np.exp(-0.5 * ((lum - centre) / width) ** 2)

    mask_blacks     = bell(lum, 0.00, 0.12)
    mask_shadows    = bell(lum, 0.25, 0.18)
    mask_midtones   = bell(lum, 0.50, 0.18)
    mask_highlights = bell(lum, 0.80, 0.18)

    # ── Convert sliders to luminance deltas ───────────────────────────────────
    # +100 → lift by up to +0.35 luminance units (enough to visibly open shadows)
    # -100 → pull by up to -0.35
    SCALE = 0.35

    delta  = np.zeros_like(lum)
    delta += mask_highlights * (highlights / 100.0) * SCALE
    delta += mask_midtones   * (midtones   / 100.0) * SCALE
    delta += mask_shadows    * (shadows    / 100.0) * SCALE
    delta += mask_blacks     * (blacks     / 100.0) * SCALE * 0.6  # blacks: gentler range

    # ── Apply delta to each channel proportionally ────────────────────────────
    # To preserve hue and saturation we scale all three channels by the same
    # ratio: new_lum / old_lum.  This is equivalent to adjusting value (V)
    # in HSV space without touching H or S.
    lum_new = np.clip(lum + delta, 1e-6, 1.0)

    with np.errstate(divide='ignore', invalid='ignore'):
        scale = np.where(lum > 1e-6, lum_new / lum, 1.0)[..., np.newaxis]

    result = np.clip(arr * scale, 0.0, 1.0)
    return Image.fromarray((result * 255).astype(np.uint8), mode="RGB")
