import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, zoom

def img_film_grain(
    image:              Image.Image,
    intensity:          float = 20.0,
    grain_size:         float = 1.0,
    grain_type:         str   = "gaussian",
    color_mode:         str   = "color",
    color_tint:         str   = "neutral",
    color_tint_rgb:     tuple = (0, 0, 0),
    shadow_strength:    float = 1.0,
    highlight_strength: float = 0.3,
    midtone_peak:       float = 0.4,
    vignette_boost:     float = 0.0,
    seed:               int   = None,
) -> Image.Image:
    if intensity == 0:
        return image.convert("RGB")

    # ── Validation ────────────────────────────────────────────────────────────
    if not (0.0 <= intensity <= 100.0):
        raise ValueError(f"intensity must be 0–100, got {intensity}")
    if not (0.5 <= grain_size <= 8.0):
        raise ValueError(f"grain_size must be 0.5–8.0, got {grain_size}")
    if grain_type not in {"gaussian", "organic", "salt_pepper", "fine"}:
        raise ValueError(f"grain_type must be gaussian|organic|salt_pepper|fine, got '{grain_type}'")
    if color_mode not in {"color", "monochrome"}:
        raise ValueError(f"color_mode must be color|monochrome, got '{color_mode}'")
    if color_tint not in {"neutral", "warm", "cool", "green", "custom"}:
        raise ValueError(f"color_tint must be neutral|warm|cool|green|custom, got '{color_tint}'")

    rng = np.random.default_rng(seed)

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32)       # (H, W, 3)  0–255
    H, W = arr.shape[:2]

    lum = (0.299 * arr[...,0] + 0.587 * arr[...,1] + 0.114 * arr[...,2]) / 255.0

    bell_width = 0.28
    bell       = np.exp(-0.5 * ((lum - midtone_peak) / bell_width) ** 2)

    shadow_mask    = np.clip(1.0 - lum / (midtone_peak + 1e-6), 0, 1)
    highlight_mask = np.clip((lum - midtone_peak) / (1.0 - midtone_peak + 1e-6), 0, 1)

    lum_mask = bell * (
        1.0
        + shadow_mask    * (shadow_strength    - 1.0)
        + highlight_mask * (highlight_strength - 1.0)
    )
    lum_mask = np.clip(lum_mask, 0, None)

    if vignette_boost > 0:
        cy, cx  = H / 2.0, W / 2.0
        y_idx, x_idx = np.mgrid[0:H, 0:W]
        dist    = np.sqrt(((y_idx - cy) / cy) ** 2 + ((x_idx - cx) / cx) ** 2)
        vig_map = np.clip(dist / np.sqrt(2.0), 0, 1) ** 1.5
        lum_mask = lum_mask * (1.0 + vignette_boost * vig_map)

    sigma = intensity / 255.0 * 40.0      # scale to pixel units (0–~16 at max)

    def make_noise(shape):
        raw = rng.standard_normal(shape).astype(np.float32)

        if grain_type == "gaussian":
            if grain_size > 0.6:
                raw = gaussian_filter(raw, sigma=grain_size * 0.5)

        elif grain_type == "organic":
            coarse = gaussian_filter(rng.standard_normal(shape).astype(np.float32),
                                     sigma=grain_size * 2.0)
            fine   = gaussian_filter(raw, sigma=grain_size * 0.3)
            raw    = coarse * 0.6 + fine * 0.4

        elif grain_type == "salt_pepper":
            raw    = np.zeros(shape, dtype=np.float32)
            thresh = 0.01 + (intensity / 100.0) * 0.04
            raw[rng.random(shape) < thresh]  =  5.0
            raw[rng.random(shape) < thresh]  = -5.0

        elif grain_type == "fine":
            raw = gaussian_filter(raw, sigma=max(0.3, grain_size * 0.2))

        return raw

    if color_mode == "monochrome":
        base_noise = make_noise((H, W))
        noise_r = base_noise.copy()
        noise_g = base_noise.copy()
        noise_b = base_noise.copy()
    else:
        noise_r = make_noise((H, W))
        noise_b = make_noise((H, W))
        if grain_size > 0.6 and grain_type == "gaussian":
            raw_g = rng.standard_normal((H, W)).astype(np.float32)
            noise_g = gaussian_filter(raw_g, sigma=grain_size * 0.35)
        else:
            noise_g = make_noise((H, W))

    TINTS = {
        "neutral": (1.00, 1.00, 1.00),
        "warm":    (1.20, 1.05, 0.75),   # Kodak Portra: warm orange grain
        "cool":    (0.80, 0.95, 1.25),   # Fuji/Ilford: cool blue grain
        "green":   (0.90, 1.20, 0.80),   # cross-process: green-yellow grain
        "custom":  None,
    }

    if color_tint == "custom":
        tr = 1.0 + color_tint_rgb[0] / 50.0
        tg = 1.0 + color_tint_rgb[1] / 50.0
        tb = 1.0 + color_tint_rgb[2] / 50.0
        tint = (tr, tg, tb)
    else:
        tint = TINTS[color_tint]

    grain_r = noise_r * sigma * lum_mask * tint[0]
    grain_g = noise_g * sigma * lum_mask * tint[1]
    grain_b = noise_b * sigma * lum_mask * tint[2]

    out = arr.copy()
    out[..., 0] = np.clip(arr[..., 0] + grain_r, 0, 255)
    out[..., 1] = np.clip(arr[..., 1] + grain_g, 0, 255)
    out[..., 2] = np.clip(arr[..., 2] + grain_b, 0, 255)

    return Image.fromarray(out.astype(np.uint8), mode="RGB")
