import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates


def _to_luminance(arr):
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

def img_edge_jitter(
    image: Image.Image,
    strength: float = 0.5,
    radius: float = 1.5,
    edge_threshold: float = 0.1,
    randomness: float = 0.5,
    seed: int = 0,
    precision: bool = False,
) -> Image.Image:

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    if not precision:
        arr = np.round(arr * 255.0) / 255.0

    h, w, _ = arr.shape
    luma = _to_luminance(arr)
    gx = gaussian_filter(luma, sigma=radius, order=[0, 1])
    gy = gaussian_filter(luma, sigma=radius, order=[1, 0])
    edges = np.sqrt(gx * gx + gy * gy)
    edges = edges / (edges.max() + 1e-6)
    edge_mask = (edges > edge_threshold).astype(np.float32)
    rng = np.random.default_rng(seed)
    noise_x = gaussian_filter(rng.normal(0, 1, (h, w)), sigma=radius)
    noise_y = gaussian_filter(rng.normal(0, 1, (h, w)), sigma=radius)
    noise_x /= (np.std(noise_x) + 1e-6)
    noise_y /= (np.std(noise_y) + 1e-6)
    dx = noise_x * strength * edge_mask * randomness
    dy = noise_y * strength * edge_mask * randomness
    yy, xx = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    coords_y = np.clip(yy + dy, 0, h - 1)
    coords_x = np.clip(xx + dx, 0, w - 1)
    out = np.zeros_like(arr)
    for c in range(3):
        out[..., c] = map_coordinates(arr[..., c], [coords_y, coords_x], order=1, mode="reflect")
    out = np.clip(out, 0.0, 1.0)
    out = (out * 255.0).astype(np.uint8)

    return Image.fromarray(out, mode="RGB")