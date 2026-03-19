import numpy as np
from PIL import Image


def img_white_balance(
    image:       Image.Image,
    temperature: float = 6500,
    tint:        float = 0,
) -> Image.Image:
    """
    RGB white balance adjustment using colour temperature in Kelvin.
    Approximates the DxO-style cool/warm slider with real Kelvin values.

    Works on JPEG/PNG by applying per-channel RGB gain corrections derived
    from the difference between the target temperature and neutral (6500K).
    This is an approximation — true white balance requires RAW data — but
    produces visually correct and useful results on rendered images.

    Args:
        image       : PIL Image (RGB)
        temperature : 2000 … 12000. Colour temperature in Kelvin.
                      2000K = very warm / candlelight (strong orange)
                      3200K = tungsten / indoor lamp (warm)
                      5500K = daylight / flash (neutral-warm)
                      6500K = sRGB standard white point (no change)
                      7500K = overcast / shade (cool-blue)
                      10000K = deep shade / very blue sky (very cool)
                      12000K = extreme cool (strong blue)
        tint        : -100 … +100. Green/magenta tint correction.
                      Negative = shift toward magenta.
                      Positive = shift toward green.
                      0 = no tint (default).

    Returns:
        PIL Image (RGB)
    """
    if temperature == 6500 and tint == 0:
        return image.convert("RGB")

    if not (2000 <= temperature <= 12000):
        raise ValueError(f"temperature must be 2000–12000K, got {temperature}")
    if not (-100 <= tint <= 100):
        raise ValueError(f"tint must be -100 … +100, got {tint}")

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0

    # ── Kelvin → RGB gain ─────────────────────────────────────────────────────
    # Derived from Tanner Helland's empirical Kelvin→RGB formula, then
    # normalised so 6500K = (1.0, 1.0, 1.0) — no change at neutral point.
    # We compute gain for the target temp and divide by the 6500K gain to
    # get the relative correction needed on a sRGB-origin image.

    def kelvin_to_rgb(K):
        K = K / 100.0
        # Red
        if K <= 66:
            R = 255.0
        else:
            R = 329.698727446 * ((K - 60) ** -0.1332047592)
        R = np.clip(R, 0, 255)

        # Green
        if K <= 66:
            G = 99.4708025861 * np.log(K) - 161.1195681661
        else:
            G = 288.1221695283 * ((K - 60) ** -0.0755148492)
        G = np.clip(G, 0, 255)

        # Blue
        if K >= 66:
            B = 255.0
        elif K <= 19:
            B = 0.0
        else:
            B = 138.5177312231 * np.log(K - 10) - 305.0447927307
        B = np.clip(B, 0, 255)

        return np.array([R, G, B]) / 255.0

    # RGB at target temperature and at neutral 6500K
    rgb_target  = kelvin_to_rgb(temperature)
    rgb_neutral = kelvin_to_rgb(6500)

    # Per-channel gain relative to neutral
    with np.errstate(divide='ignore', invalid='ignore'):
        gain = np.where(rgb_neutral > 0, rgb_target / rgb_neutral, 1.0)

    # ── Apply temperature gain ─────────────────────────────────────────────────
    arr = arr * gain

    # ── Apply tint (green ↔ magenta on green channel) ─────────────────────────
    if tint != 0:
        tint_gain = 1.0 + (tint / 100.0) * 0.3
        arr[..., 1] = arr[..., 1] * tint_gain

    arr = np.clip(arr, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")
