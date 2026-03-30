import os
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter
from safetensors.torch import load_file

from ...utils import comfy_dir
from ...components.depth_anything_v2.dpt import DepthAnythingV2

_depth_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _find_best_model():
    base = os.path.join(comfy_dir, "models", "depthanything")

    priority = [
        "depth_anything_v2_vitl_fp32.safetensors",
        "depth_anything_v2_vitl_fp16.safetensors",
        "depth_anything_v2_vitb_fp16.safetensors",
        "depth_anything_v2_vitb_fp32.safetensors",
        "depth_anything_v2_vits_fp16.safetensors",
        "depth_anything_v2_vits_fp32.safetensors",
    ]

    for name in priority:
        path = os.path.join(base, name)
        if os.path.exists(path):
            return path

    return None


def _load_depth_model():
    global _depth_model

    if _depth_model is not None:
        return _depth_model

    model_path = _find_best_model()
    if model_path is None:
        raise RuntimeError("No Depth Anything model found")

    model = DepthAnythingV2()
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict, strict=False)

    model.to(_device)
    model.eval()

    _depth_model = model
    return _depth_model

def _predict_depth(arr):
    model = _load_depth_model()

    h, w, _ = arr.shape

    img = (arr * 255.0).astype(np.uint8)
    img = Image.fromarray(img)

    # --- preprocessing ---
    img = img.resize((518, 518))
    x = np.array(img).astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x).unsqueeze(0).to(_device)

    with torch.no_grad():
        depth = model(x)

    depth = depth.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

    depth = Image.fromarray((depth * 255).astype(np.uint8)).resize((w, h))
    depth = np.array(depth).astype(np.float32) / 255.0

    return depth


def _to_luminance(arr):
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def _sharpness_map(luma, radius):
    gx = gaussian_filter(luma, sigma=radius, order=[0, 1])
    gy = gaussian_filter(luma, sigma=radius, order=[1, 0])
    mag = np.sqrt(gx * gx + gy * gy)
    return mag / (mag.max() + 1e-6)


def img_depth_blur(
    image: Image.Image,
    focus_depth: float = 0.5,
    depth_range: float = 0.2,
    max_blur: float = 8.0,
    depth_gamma: float = 1.0,
    precision: bool = False,
    sharpness_bias: float = 1.5,
    sharpness_threshold: float = 0.2,
) -> Image.Image:

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0

    if not precision:
        arr = np.round(arr * 255.0) / 255.0

    h, w, _ = arr.shape

    # -----------------------------
    # Depth from AI model
    # -----------------------------
    depth = _predict_depth(arr)

    # remap depth (far = more blur)
    depth = depth ** depth_gamma

    depth_blur = np.clip((depth - focus_depth) / (depth_range + 1e-6), 0.0, 1.0)

    # -----------------------------
    # Sharpness refinement
    # -----------------------------
    luma = _to_luminance(arr)
    sharp = _sharpness_map(luma, radius=1.0)

    sharp_mask = np.clip((sharp - sharpness_threshold) * sharpness_bias, 0.0, 1.0)

    # final mask (depth + sharpness protection)
    blur_mask = depth_blur * (1.0 - sharp_mask)
    blur_map = blur_mask * max_blur

    # -----------------------------
    # Multi-scale blur
    # -----------------------------
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

    out = np.clip(out, 0.0, 1.0)
    out = (out * 255.0).astype(np.uint8)

    return Image.fromarray(out, mode="RGB")