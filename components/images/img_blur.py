import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, uniform_filter


def img_blur(image: Image.Image, blur_type: str = "gaussian", intensity: float = 1.0, radius: float = 2.0, angle: float = 0.0, edge_only: bool = False,) -> Image.Image:
    """
    Multi-type blur with consistent interface.

    Args:
        image     : PIL Image (RGB)
        blur_type : Type of blur. One of:
                    "gaussian"   — smooth natural blur (default)
                    "box"        — fast uniform average blur
                    "motion"     — directional linear blur (use angle param)
                    "bilateral"  — edge-preserving blur (keeps sharp edges,
                                   smooths flat areas — most useful for AI images)
                    "lens"       — simulates circular lens defocus (disc kernel)
        intensity : Overall strength multiplier. 0.0 = no effect, 1.0 = normal,
                    2.0 = double strength. Range: 0.0 … 5.0
        radius    : Blur radius / kernel size in pixels. Range: 0.5 … 50.0
                    Meaning varies by type:
                    gaussian  → sigma value
                    box       → half-width of averaging window
                    motion    → length of motion trail in pixels
                    bilateral → spatial sigma (colour sigma auto-scaled)
                    lens      → radius of defocus disc
        angle     : Motion blur direction in degrees. 0 = horizontal,
                    90 = vertical. Only used for blur_type="motion".
        edge_only : If True, apply blur only to low-frequency (flat) areas,
                    preserving edges. Works with all blur types by masking
                    with a Sobel edge map. Default: False.
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

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0   # (H, W, 3)  0–1

    effective_radius = radius * intensity

    # ── GAUSSIAN ──────────────────────────────────────────────────────────────
    if blur_type == "gaussian":
        blurred = np.stack([
            gaussian_filter(arr[..., c], sigma=effective_radius)
            for c in range(3)
        ], axis=-1)

    # ── BOX ───────────────────────────────────────────────────────────────────
    elif blur_type == "box":
        size = max(1, int(effective_radius * 2 + 1))
        blurred = np.stack([
            uniform_filter(arr[..., c], size=size)
            for c in range(3)
        ], axis=-1)

    # ── MOTION ────────────────────────────────────────────────────────────────
    elif blur_type == "motion":
        length = max(1, int(effective_radius * 2 + 1))
        # Build a 1D motion kernel at the given angle
        kernel = _motion_kernel(length, angle)
        from scipy.ndimage import convolve
        blurred = np.stack([
            convolve(arr[..., c], kernel)
            for c in range(3)
        ], axis=-1)

    # ── BILATERAL (edge-preserving) ───────────────────────────────────────────
    elif blur_type == "bilateral":
        blurred = _bilateral_blur(arr, spatial_sigma=effective_radius,
                                  color_sigma=0.1 * intensity)

    # ── LENS (disc defocus) ───────────────────────────────────────────────────
    elif blur_type == "lens":
        kernel = _disc_kernel(effective_radius)
        from scipy.ndimage import convolve
        blurred = np.stack([
            convolve(arr[..., c], kernel)
            for c in range(3)
        ], axis=-1)

    # ── Edge mask (optional) ──────────────────────────────────────────────────
    if edge_only:
        # Compute edge strength via Sobel, use as inverse mask
        # High edge → keep original. Low edge (flat area) → use blurred.
        grey = 0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]
        from scipy.ndimage import sobel
        sx = sobel(grey, axis=0)
        sy = sobel(grey, axis=1)
        edge_mag = np.sqrt(sx**2 + sy**2)
        # Normalise to 0–1, then use as "keep original" weight
        edge_mag = np.clip(edge_mag / (edge_mag.max() + 1e-6), 0, 1)
        edge_mag = edge_mag[..., np.newaxis]
        blurred = arr * edge_mag + blurred * (1.0 - edge_mag)

    result = np.clip(blurred, 0.0, 1.0)
    return Image.fromarray((result * 255).astype(np.uint8), mode="RGB")


# ── Kernel helpers ────────────────────────────────────────────────────────────

def _motion_kernel(length: int, angle_deg: float) -> np.ndarray:
    """1-D motion blur kernel rotated to angle_deg degrees."""
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
    """Circular disc kernel for lens defocus simulation."""
    r    = int(np.ceil(radius))
    size = 2 * r + 1
    y, x = np.ogrid[-r:r+1, -r:r+1]
    disc = (x**2 + y**2 <= radius**2).astype(np.float32)
    s    = disc.sum()
    return disc / s if s > 0 else disc


def _bilateral_blur(arr: np.ndarray, spatial_sigma: float,
                    color_sigma: float) -> np.ndarray:
    """
    Fast approximate bilateral filter via repeated small Gaussian passes.
    True bilateral is O(N²·k²) per pixel — prohibitively slow in pure numpy
    for large images. This approximation (range decomposition) gives a
    visually equivalent result at a fraction of the cost.
    """
    # Decompose into low-freq (Gaussian blurred) and high-freq (detail)
    # then selectively suppress detail in smooth regions
    blurred_spatial = np.stack([
        gaussian_filter(arr[..., c], sigma=spatial_sigma)
        for c in range(3)
    ], axis=-1)

    # Per-pixel colour distance from blurred version (how "edgy" the pixel is)
    diff       = np.abs(arr - blurred_spatial).mean(axis=-1)  # (H, W)
    # Pixels with large colour difference (edges) → keep original
    # Pixels with small difference (flat) → use blurred
    edge_weight = np.exp(-0.5 * (diff / (color_sigma + 1e-6)) ** 2)
    edge_weight = edge_weight[..., np.newaxis]

    return arr * (1.0 - edge_weight) + blurred_spatial * edge_weight
