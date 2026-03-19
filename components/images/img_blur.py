import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter, convolve, sobel


def img_blur(
    image:                    Image.Image,
    blur_type:                str   = "gaussian",
    intensity:                float = 1.0,
    radius:                   float = 2.0,
    angle:                    float = 0.0,
    edge_only:                bool  = False,
    bilateral_edge_sensitivity: float = 0.5,
    edge_threshold:           float = 0.0,
) -> Image.Image:
    """
    Multi-type blur with consistent interface.

    Args:
        image                    : PIL Image (RGB)
        blur_type                : "gaussian" | "box" | "motion" |
                                   "bilateral" | "lens"
        intensity                : 0.0 … 5.0. Overall strength multiplier.
                                   0.0 = no effect (passthrough).
        radius                   : 0.5 … 50.0. Blur radius in pixels.
        angle                    : 0 … 360. Motion blur direction in degrees.
                                   Only used for blur_type="motion".
        edge_only                : If True, blur is applied only to flat/smooth
                                   areas. Edges are preserved using a Sobel map.
        bilateral_edge_sensitivity: 0.0 … 1.0. Only used for blur_type="bilateral".
                                   Controls how strongly edges are protected.
                                   0.0 = very strong edge protection — only
                                         near-flat areas are blurred, sharp
                                         edges are untouched.
                                   0.5 = balanced (default).
                                   1.0 = weak edge protection — edges also
                                         receive some blur.
        edge_threshold           : 0.0 … 1.0. Only used when edge_only=True.
                                   Pixels with edge strength above this value
                                   are kept fully sharp. Pixels below are
                                   blurred proportionally.
                                   0.0 = no threshold, all pixels get some
                                         blur (default, previous behaviour).
                                   0.3 = moderate — clear separation between
                                         sharp edges and blurred flat areas.
                                   0.7 = aggressive — only the strongest edges
                                         are protected, most areas are blurred.
    Returns:
        PIL Image (RGB)
    """
    if intensity == 0:
        return image.convert("RGB")

    VALID = {"gaussian", "box", "motion", "bilateral", "lens"}
    if blur_type not in VALID:
        raise ValueError(f"blur_type must be one of {VALID}, got '{blur_type}'")
    if not (0.0 <= intensity <= 5.0):
        raise ValueError(f"intensity must be 0.0 … 5.0, got {intensity}")
    if not (0.5 <= radius <= 50.0):
        raise ValueError(f"radius must be 0.5 … 50.0, got {radius}")
    if not (0.0 <= bilateral_edge_sensitivity <= 1.0):
        raise ValueError(f"bilateral_edge_sensitivity must be 0.0 … 1.0, got {bilateral_edge_sensitivity}")
    if not (0.0 <= edge_threshold <= 1.0):
        raise ValueError(f"edge_threshold must be 0.0 … 1.0, got {edge_threshold}")

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0

    effective_radius = radius * intensity

    # ── Blur types ────────────────────────────────────────────────────────────
    if blur_type == "gaussian":
        blurred = np.stack([
            gaussian_filter(arr[..., c], sigma=effective_radius)
            for c in range(3)
        ], axis=-1)

    elif blur_type == "box":
        size = max(1, int(effective_radius * 2 + 1))
        blurred = np.stack([
            uniform_filter(arr[..., c], size=size)
            for c in range(3)
        ], axis=-1)

    elif blur_type == "motion":
        length  = max(1, int(effective_radius * 2 + 1))
        kernel  = _motion_kernel(length, angle)
        blurred = np.stack([
            convolve(arr[..., c], kernel)
            for c in range(3)
        ], axis=-1)

    elif blur_type == "bilateral":
        # color_sigma: maps sensitivity 0.0 → 0.02 (very tight), 1.0 → 0.3 (loose)
        color_sigma = 0.02 + bilateral_edge_sensitivity * 0.28
        blurred = _bilateral_blur(arr, spatial_sigma=effective_radius,
                                  color_sigma=color_sigma)

    elif blur_type == "lens":
        kernel  = _disc_kernel(effective_radius)
        blurred = np.stack([
            convolve(arr[..., c], kernel)
            for c in range(3)
        ], axis=-1)

    # ── Edge-only mask ────────────────────────────────────────────────────────
    if edge_only:
        grey     = 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]
        sx       = sobel(grey, axis=0)
        sy       = sobel(grey, axis=1)
        edge_mag = np.sqrt(sx**2 + sy**2)
        edge_mag = np.clip(edge_mag / (edge_mag.max() + 1e-6), 0, 1)

        # Apply threshold: remap [threshold … 1] → [0 … 1] so pixels above
        # threshold are fully sharp, pixels below are progressively blurred.
        if edge_threshold > 0:
            edge_mag = np.clip((edge_mag - edge_threshold) /
                               (1.0 - edge_threshold + 1e-6), 0, 1)

        edge_mag = edge_mag[..., np.newaxis]
        blurred  = arr * edge_mag + blurred * (1.0 - edge_mag)

    result = np.clip(blurred, 0.0, 1.0)
    return Image.fromarray((result * 255).astype(np.uint8), mode="RGB")


# ── Kernel helpers ────────────────────────────────────────────────────────────

def _motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    angle_rad = np.deg2rad(angle_deg)
    cx, cy    = length // 2, length // 2
    kernel    = np.zeros((length, length), dtype=np.float32)
    for i in range(length):
        offset = i - cx
        x = int(round(cx + offset * np.cos(angle_rad)))
        y = int(round(cy + offset * np.sin(angle_rad)))
        if 0 <= x < length and 0 <= y < length:
            kernel[y, x] = 1.0
    s = kernel.sum()
    return kernel / s if s > 0 else kernel


def _disc_kernel(radius: float) -> np.ndarray:
    r    = int(np.ceil(radius))
    size = 2 * r + 1
    y, x = np.ogrid[-r:r+1, -r:r+1]
    disc = (x**2 + y**2 <= radius**2).astype(np.float32)
    s    = disc.sum()
    return disc / s if s > 0 else disc


def _bilateral_blur(arr: np.ndarray, spatial_sigma: float,
                    color_sigma: float) -> np.ndarray:
    blurred_spatial = np.stack([
        gaussian_filter(arr[..., c], sigma=spatial_sigma)
        for c in range(3)
    ], axis=-1)
    diff        = np.abs(arr - blurred_spatial).mean(axis=-1)
    edge_weight = np.exp(-0.5 * (diff / (color_sigma + 1e-6)) ** 2)
    edge_weight = edge_weight[..., np.newaxis]
    return arr * (1.0 - edge_weight) + blurred_spatial * edge_weight