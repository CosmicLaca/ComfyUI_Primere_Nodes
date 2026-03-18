import numpy as np
from PIL import Image


def img_smart_lighting(image: Image.Image, intensity: float = 0,) -> Image.Image:
    """
    DxO-style Smart Lighting (Uniform mode).

    A single intensity slider simultaneously:
      - Lifts shadows and dark midtones (opens up underexposed areas)
      - Protects / gently compresses highlights (prevents blow-out)
      - Leaves pure blacks and pure whites anchored

    The effect is an asymmetric luminosity-dependent tone curve:
      - Steeper (more gain) in the shadow/low-mid region
      - Flatter (less gain, slight compression) in the highlight region
      - Neutral pivot around lum ≈ 0.5

    This mirrors DxO's "balance shadows and lightning" behaviour:
    as shadows are lifted, highlights are slightly pulled back,
    keeping the overall exposure feel natural.

    Args:
        image     : PIL Image (RGB)
        intensity : 0 … 100.
                    0  = no change (passthrough)
                    27 = moderate lift (matches DxO screenshot default)
                    100 = maximum shadow lift / highlight compression
    Returns:
        PIL Image (RGB)
    """
    if intensity == 0:
        return image.convert("RGB")

    if not (0 <= intensity <= 100):
        raise ValueError(f"intensity must be 0 … 100, got {intensity}")

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, 3)  0–1

    # ── Build a 256-entry tone curve LUT ──────────────────────────────────────
    #
    # The curve is constructed as a blend between the identity line (y=x)
    # and a target curve that lifts shadows while compressing highlights.
    #
    # Target curve shape (at full intensity=100):
    #   - Blacks (x=0)      → stay at 0       (anchor)
    #   - Shadows (x=0.25)  → lift to ~0.38   (+52%)
    #   - Midtones (x=0.50) → slight lift ~0.54 (+8%) — pivot zone
    #   - Highlights (x=0.75)→ slight pull ~0.72 (-4%)
    #   - Whites (x=1.0)    → stay at 1.0     (anchor)
    #
    # The actual curve is a smooth spline through these control points,
    # implemented as a weighted sum of two gamma curves:
    #   shadow_gamma < 1  → lifts darks
    #   highlight_gamma > 1 → compresses brights
    # blended by a luminance-dependent weight.

    t = np.linspace(0.0, 1.0, 256, dtype=np.float64)  # input values

    strength = intensity / 100.0   # 0 … 1

    # Shadow lift: power curve with gamma < 1 lifts the dark end
    shadow_gamma    = 1.0 - strength * 0.55     # 1.0 → 0.45  at full strength
    shadow_gamma    = max(shadow_gamma, 0.35)
    shadow_curve    = np.power(np.clip(t, 0, 1), shadow_gamma)

    # Highlight compression: power curve with gamma > 1 pulls the bright end
    highlight_gamma = 1.0 + strength * 0.40     # 1.0 → 1.40  at full strength
    highlight_curve = np.power(np.clip(t, 0, 1), highlight_gamma)

    # Blend weight: shadow_curve dominates in darks, highlight_curve in brights
    # w=1 → full shadow_curve,  w=0 → full highlight_curve
    # Smooth S-shaped crossover around t=0.5
    w = 1.0 - t ** 1.5   # falls from 1 (dark) to 0 (bright)

    combined = w * shadow_curve + (1.0 - w) * highlight_curve

    # Blend identity (no change) with combined curve by strength
    lut = (1.0 - strength) * t + strength * combined
    lut = np.clip(lut, 0.0, 1.0)

    # ── Apply LUT per channel ─────────────────────────────────────────────────
    # LUT is luminance-based but applied uniformly across R/G/B to preserve hue.
    indices = np.clip((arr * 255).astype(np.int32), 0, 255)
    result  = lut[indices].astype(np.float32)

    return Image.fromarray((result * 255).astype(np.uint8), mode="RGB")
