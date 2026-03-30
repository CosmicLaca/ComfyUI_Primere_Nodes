import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_closing, binary_fill_holes, distance_transform_edt

def _to_luminance(arr):
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]

def _sharpness_map(luma, radius):
    gx = gaussian_filter(luma, sigma=radius, order=[0, 1])
    gy = gaussian_filter(luma, sigma=radius, order=[1, 0])
    mag = np.sqrt(gx * gx + gy * gy)
    mag = mag / (mag.max() + 1e-6)
    return mag

def img_depth_blur(
    image: Image.Image,
    focus_depth: float = 0.5,
    depth_range: float = 0.2,
    max_blur: float = 8.0,
    depth_gamma: float = 1.0,
    precision: bool = False,
    sharpness_threshold: float = 0.2,
) -> Image.Image:
    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    if not precision:
        arr = np.round(arr * 255.0) / 255.0

    h, w, _ = arr.shape
    luma = _to_luminance(arr)
    depth = np.clip(luma, 0.0, 1.0)
    depth = depth ** depth_gamma
    sharp = _sharpness_map(luma, radius=1.0)
    binary = sharp > sharpness_threshold
    binary = binary_closing(binary, structure=np.ones((5, 5)))
    binary = binary_fill_holes(binary)
    dist = distance_transform_edt(~binary)
    region_mask = (dist < 3.0).astype(np.float32)
    region_mask = gaussian_filter(region_mask, sigma=1.0)
    region_mask = np.clip(region_mask, 0.0, 1.0)
    sharp_mask = region_mask
    blur_mask = 1.0 - sharp_mask
    depth_blur = np.abs(depth - focus_depth) / (depth_range + 1e-6)
    depth_blur = np.clip(depth_blur, 0.0, 1.0)
    blur_map = depth_blur * blur_mask
    blur_map = blur_map * max_blur
    levels = 5
    sigmas = np.linspace(0.0, max_blur, levels)
    blurred_stack = []
    for s in sigmas:
        if s > 0:
            blurred = gaussian_filter(arr, sigma=(s, s, 0))
        else:
            blurred = arr
        blurred_stack.append(blurred)
    blurred_stack = np.stack(blurred_stack, axis=0)
    idx = blur_map / (max_blur + 1e-6) * (levels - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.clip(i0 + 1, 0, levels - 1)
    f = idx - i0
    out = np.zeros_like(arr)
    for c in range(3):
        b0 = blurred_stack[i0, np.arange(h)[:, None], np.arange(w), c]
        b1 = blurred_stack[i1, np.arange(h)[:, None], np.arange(w), c]
        out[..., c] = b0 * (1 - f) + b1 * f
    edge = sharp > sharpness_threshold
    edge = gaussian_filter(edge.astype(np.float32), sigma=1.0)
    edge = np.clip(edge, 0.0, 1.0)
    out = out * (1.0 - edge[..., None]) + arr * edge[..., None]
    out = np.clip(out, 0.0, 1.0)
    out = (out * 255.0).astype(np.uint8)

    return Image.fromarray(out, mode="RGB")