import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter

def _to_luminance(arr: np.ndarray) -> np.ndarray:
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

def _edge_magnitude(luma: np.ndarray, radius: float) -> np.ndarray:
    gx = gaussian_filter(luma, sigma=radius, order=[0, 1])
    gy = gaussian_filter(luma, sigma=radius, order=[1, 0])
    return np.sqrt(gx * gx + gy * gy)

def _grain(shape, scale, rng):
    h, w = shape
    noise = rng.normal(0.0, 1.0, (h, w)).astype(np.float32)
    if scale > 1.0:
        noise = gaussian_filter(noise, sigma=scale)
    noise = noise / (np.std(noise) + 1e-6)
    return noise

def img_solarization_bw(
    image: Image.Image,
    strength: float = 0.6,
    pivot: float = 0.5,
    sigma: float = 0.18,
    edge_boost: float = 0.8,
    edge_radius: float = 1.0,
    contrast: float = 1.1,
    precision: bool = False,
    hard_paper: bool = False,
    grain_modulation: bool = False,
    grain_strength: float = 0.15,
    grain_scale: float = 1.0,
    seed: int = 0,
) -> Image.Image:

    img = image.convert("RGB")

    if precision:
        max_val = 65535.0
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = arr * max_val
        arr = arr / max_val
    else:
        arr = np.array(img, dtype=np.float32) / 255.0

    luma = _to_luminance(arr)

    if hard_paper:
        sigma_eff = sigma * 0.65
        contrast_eff = contrast * 1.25
        edge_boost_eff = edge_boost * 1.2
    else:
        sigma_eff = sigma
        contrast_eff = contrast
        edge_boost_eff = edge_boost

    w = np.exp(-((luma - pivot) ** 2) / (2.0 * sigma_eff * sigma_eff))

    if grain_modulation:
        rng = np.random.default_rng(seed)
        g = _grain(luma.shape, grain_scale, rng)
        g = g * grain_strength
        w = np.clip(w + g * w, 0.0, 1.0)

    inverted = 1.0 - luma
    solar = luma * (1.0 - w) + inverted * w

    edges = _edge_magnitude(luma, edge_radius)
    edges = edges / (edges.max() + 1e-6)

    solar = luma * (1.0 - strength) + solar * strength * (1.0 + edge_boost_eff * edges)

    solar = (solar - 0.5) * contrast_eff + 0.5
    solar = np.clip(solar, 0.0, 1.0)

    out = (solar * 255.0).astype(np.uint8)
    out_rgb = np.stack([out, out, out], axis=-1)

    return Image.fromarray(out_rgb, mode="RGB")