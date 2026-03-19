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
    """
    Professional film grain / noise simulation.

    Args:
        image              : PIL Image (RGB)

        intensity          : 0.0 … 100.0. Overall grain amount.
                             0   = no grain (passthrough)
                             10  = subtle, fine film grain
                             25  = visible ISO 800-style grain
                             50  = heavy ISO 3200-style grain
                             100 = extreme / damaged film

        grain_size         : 0.5 … 8.0. Physical grain clump size in pixels.
                             0.5 = ultra-fine digital noise
                             1.0 = fine grain (ISO 100–400 film)
                             2.5 = medium grain (ISO 800–1600)
                             5.0 = chunky grain (ISO 3200+ / push-processed)
                             8.0 = very coarse grain / heavy texture

        grain_type         : Type of noise pattern.
                             "gaussian"    — smooth Gaussian distribution,
                                            classic film look (default)
                             "organic"     — smooth flowing variation via
                                            multi-scale Gaussian blending,
                                            mimics natural film crystal clumping
                             "salt_pepper" — harsh bright/dark specks,
                                            damaged film / scanner artifacts
                             "fine"        — very tight high-frequency grain,
                                            emphasises texture details

        color_mode         : "color"       — independent per-channel grain,
                                            green channel 30% finer (matches
                                            real color film layer structure)
                             "monochrome"  — identical grain on R/G/B,
                                            classic B&W film look

        color_tint         : Built-in grain color character presets.
                             "neutral"     — no color bias in grain (default)
                             "warm"        — Kodak Portra-style: grain has
                                            slight orange-yellow warmth
                             "cool"        — Fuji/Ilford-style: grain has
                                            slight blue-cyan coolness
                             "green"       — pushed/cross-processed look,
                                            grain skews green-yellow
                             "custom"      — use color_tint_rgb values

        color_tint_rgb     : (R, G, B) tuple. Each value -50 … +50.
                             Only used when color_tint="custom".
                             Adds a per-channel bias to the grain itself.
                             Example: (10, -5, -8) = warm reddish grain tint.

        shadow_strength    : 0.0 … 3.0. Grain multiplier in shadow zones.
                             1.0 = normal (default)
                             2.0 = twice as much grain in darks
                             0.0 = no grain in shadows

        highlight_strength : 0.0 … 3.0. Grain multiplier in highlight zones.
                             0.3 = subtle grain in brights (default, film-like)
                             0.0 = clean highlights, grain only in shadows/mids
                             1.0 = equal grain in highlights

        midtone_peak       : 0.0 … 1.0. Luminance value where grain is
                             strongest (bell curve peak).
                             0.4 = peaks in shadows/low-mids (default, film-like)
                             0.5 = peaks exactly at midtone grey
                             0.2 = grain concentrated in the darkest areas

        vignette_boost     : 0.0 … 1.0. Extra grain intensity toward the
                             image edges/corners, simulating lens falloff and
                             uneven film development.
                             0.0 = uniform grain (default)
                             0.3 = subtle edge boost
                             1.0 = strong vignette grain effect

        seed               : int or None. Random seed for reproducibility.
                             None = different grain every call (default).
                             Any int = same grain pattern every time for
                             the same image size and parameters.

    Returns:
        PIL Image (RGB)
    """

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

    # ── Luminance map ─────────────────────────────────────────────────────────
    lum = (0.299 * arr[...,0] + 0.587 * arr[...,1] + 0.114 * arr[...,2]) / 255.0

    # ── Luminance zone mask ───────────────────────────────────────────────────
    # Bell curve peaks at midtone_peak, width tuned to give natural film rolloff.
    # Shadows and highlights are then independently scaled.
    bell_width = 0.28
    bell       = np.exp(-0.5 * ((lum - midtone_peak) / bell_width) ** 2)

    # Shadow mask (lum < midtone_peak) and highlight mask (lum > midtone_peak)
    shadow_mask    = np.clip(1.0 - lum / (midtone_peak + 1e-6), 0, 1)
    highlight_mask = np.clip((lum - midtone_peak) / (1.0 - midtone_peak + 1e-6), 0, 1)

    # Combine: bell base + zone-specific multipliers
    lum_mask = bell * (
        1.0
        + shadow_mask    * (shadow_strength    - 1.0)
        + highlight_mask * (highlight_strength - 1.0)
    )
    lum_mask = np.clip(lum_mask, 0, None)

    # ── Vignette boost map ────────────────────────────────────────────────────
    if vignette_boost > 0:
        cy, cx  = H / 2.0, W / 2.0
        y_idx, x_idx = np.mgrid[0:H, 0:W]
        dist    = np.sqrt(((y_idx - cy) / cy) ** 2 + ((x_idx - cx) / cx) ** 2)
        vig_map = np.clip(dist / np.sqrt(2.0), 0, 1) ** 1.5
        lum_mask = lum_mask * (1.0 + vignette_boost * vig_map)

    # ── Generate base noise ───────────────────────────────────────────────────
    sigma = intensity / 255.0 * 40.0      # scale to pixel units (0–~16 at max)

    def make_noise(shape):
        """Generate one channel of noise according to grain_type and grain_size."""
        raw = rng.standard_normal(shape).astype(np.float32)

        if grain_type == "gaussian":
            # Blur to grain_size then re-sharpen slightly for texture
            if grain_size > 0.6:
                raw = gaussian_filter(raw, sigma=grain_size * 0.5)

        elif grain_type == "organic":
            # Multi-scale blend: coarse structure + fine detail
            coarse = gaussian_filter(rng.standard_normal(shape).astype(np.float32),
                                     sigma=grain_size * 2.0)
            fine   = gaussian_filter(raw, sigma=grain_size * 0.3)
            raw    = coarse * 0.6 + fine * 0.4

        elif grain_type == "salt_pepper":
            # Replace ~2% of pixels with extreme values
            raw    = np.zeros(shape, dtype=np.float32)
            thresh = 0.01 + (intensity / 100.0) * 0.04
            raw[rng.random(shape) < thresh]  =  5.0
            raw[rng.random(shape) < thresh]  = -5.0

        elif grain_type == "fine":
            # Very high frequency — minimal blur, tight pixel-level texture
            raw = gaussian_filter(raw, sigma=max(0.3, grain_size * 0.2))

        return raw

    # ── Per-channel noise ─────────────────────────────────────────────────────
    if color_mode == "monochrome":
        base_noise = make_noise((H, W))
        noise_r = base_noise.copy()
        noise_g = base_noise.copy()
        noise_b = base_noise.copy()
    else:
        # Color film: independent layers, green is 30% finer (real film property)
        noise_r = make_noise((H, W))
        noise_b = make_noise((H, W))
        if grain_size > 0.6 and grain_type == "gaussian":
            # Green channel: reduce effective grain_size by 30%
            raw_g = rng.standard_normal((H, W)).astype(np.float32)
            noise_g = gaussian_filter(raw_g, sigma=grain_size * 0.35)
        else:
            noise_g = make_noise((H, W))

    # ── Color tint presets → per-channel scale ────────────────────────────────
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

    # ── Apply grain ───────────────────────────────────────────────────────────
    # Scale noise to sigma (pixel intensity units), apply lum_mask, apply tint
    grain_r = noise_r * sigma * lum_mask * tint[0]
    grain_g = noise_g * sigma * lum_mask * tint[1]
    grain_b = noise_b * sigma * lum_mask * tint[2]

    out = arr.copy()
    out[..., 0] = np.clip(arr[..., 0] + grain_r, 0, 255)
    out[..., 1] = np.clip(arr[..., 1] + grain_g, 0, 255)
    out[..., 2] = np.clip(arr[..., 2] + grain_b, 0, 255)

    return Image.fromarray(out.astype(np.uint8), mode="RGB")
