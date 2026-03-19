import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter


def img_shade_level(
    image:       Image.Image,
    shade_level: float = 0,
    radius:      float = 0,
    strength:    float = 0.5,
) -> Image.Image:
    """
    Micro-contrast / shade density adjustment on the L channel only.
    Hue and saturation are never modified.

    Args:
        image       : PIL Image (RGB)
        shade_level : -100 … +100.  0 = no change (passthrough).
                      Positive = amplify local shade differences (more texture,
                                 less plastic/flat look).
                      Negative = suppress local shade differences (smoother,
                                 more painterly).
        radius      : 0.0 … 50.0. Gaussian blur radius in pixels.
                      Defines the spatial scale of the effect:
                      0     = auto: ~1% of the shorter image dimension.
                      1–3   = pixel-level texture and fine grain only.
                      4–10  = mid-level detail, good for portraits.
                      15–50 = broad tonal area transitions.
                      Tip: smaller radius = sharper/grainier effect.
                           larger radius  = smoother/broader contrast lift.
        strength    : 0.0 … 1.0. Controls the ceiling of what shade_level
                      ±100 can do, and balances the positive/negative asymmetry.
                      0.0 = very subtle — positive and negative equally gentle.
                      0.5 = default     — positive ~3× stronger than negative
                                         (amplifying detail is more dramatic
                                          than suppressing it).
                      1.0 = maximum     — positive very aggressive,
                                          negative also stronger than default.
                      Positive multiplier: 1.0 + strength * 4.0  (1.0 … 5.0)
                      Negative multiplier: 0.5 + strength * 1.0  (0.5 … 1.5)
    Returns:
        PIL Image (RGB)
    """
    if shade_level == 0:
        return image.convert("RGB")

    if not (-100 <= shade_level <= 100):
        raise ValueError(f"shade_level must be -100 … +100, got {shade_level}")

    if not (0.0 <= strength <= 1.0):
        raise ValueError(f"strength must be 0.0 … 1.0, got {strength}")

    if not (0.0 <= radius <= 50.0):
        raise ValueError(f"radius must be 0.0 … 50.0, got {radius}")

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0

    # ── RGB → Lab ─────────────────────────────────────────────────────────────
    def rgb_to_lab(rgb):
        linear = np.where(
            rgb <= 0.04045,
            rgb / 12.92,
            ((rgb + 0.055) / 1.055) ** 2.4
        )
        M = np.array([
            [0.4124564, 0.3575761, 0.1804375],
            [0.2126729, 0.7151522, 0.0721750],
            [0.0193339, 0.1191920, 0.9503041],
        ], dtype=np.float32)
        xyz   = linear @ M.T
        xyz_n = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
        xyz_r = xyz / xyz_n
        def f(t):
            delta = 6.0 / 29.0
            return np.where(t > delta ** 3, np.cbrt(t),
                            t / (3 * delta ** 2) + 4.0 / 29.0)
        fx, fy, fz = f(xyz_r[...,0]), f(xyz_r[...,1]), f(xyz_r[...,2])
        return np.stack([116.0*fy - 16.0, 500.0*(fx-fy), 200.0*(fy-fz)], axis=-1)

    def lab_to_rgb(lab):
        L, a, b = lab[...,0], lab[...,1], lab[...,2]
        fy = (L + 16.0) / 116.0
        fx = a / 500.0 + fy
        fz = fy - b / 200.0
        def f_inv(t):
            delta = 6.0 / 29.0
            return np.where(t > delta, t ** 3,
                            3 * delta**2 * (t - 4.0/29.0))
        xyz_n = np.array([0.95047, 1.00000, 1.08883], dtype=np.float32)
        xyz   = np.stack([f_inv(fx), f_inv(fy), f_inv(fz)], axis=-1) * xyz_n
        M_inv = np.array([
            [ 3.2404542, -1.5371385, -0.4985314],
            [-0.9692660,  1.8760108,  0.0415560],
            [ 0.0556434, -0.2040259,  1.0572252],
        ], dtype=np.float32)
        linear = xyz @ M_inv.T
        srgb   = np.where(linear <= 0.0031308, linear * 12.92,
                          1.055 * np.power(np.clip(linear, 0, None), 1.0/2.4) - 0.055)
        return np.clip(srgb, 0.0, 1.0)

    # ── Resolve radius ─────────────────────────────────────────────────────────
    lab    = rgb_to_lab(arr)
    L      = lab[..., 0]
    H, W   = L.shape

    r = radius if radius > 0.0 else max(1.0, min(H, W) * 0.01)

    # ── Detail layer ──────────────────────────────────────────────────────────
    L_blurred = gaussian_filter(L, sigma=r)
    detail    = L - L_blurred

    # ── Strength-controlled asymmetric multiplier ─────────────────────────────
    # Positive ceiling: 1.0 … 5.0  (amplifying detail is inherently stronger)
    # Negative ceiling: 0.5 … 1.5  (suppression is gentler by nature)
    if shade_level > 0:
        multiplier = 1.0 + strength * 4.0
    else:
        multiplier = 0.5 + strength * 1.0

    raw_strength         = (shade_level / 100.0) * multiplier
    lab_new              = lab.copy()
    lab_new[..., 0]      = np.clip(L + raw_strength * detail, 0.0, 100.0)

    return Image.fromarray((lab_to_rgb(lab_new) * 255).astype(np.uint8), mode="RGB")