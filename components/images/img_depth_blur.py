import os
import sys
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_erosion, distance_transform_edt
import tempfile

from safetensors.torch import load_file

from ...utils import comfy_dir
from ...components.depth_anything_v2.dpt import DepthAnythingV2
from ...components.depth_anything_v3 import load_model as load_model
from ...components.depth_anything_v3 import nodes_inference as nodes_inference
from ...components import utility

_depth_model = None
_depth_model_v3 = None
_depth_model_v3_name = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Tunable constants ────────────────────────────────────────────────────────

DEPTH_MAP_BLUR_SIGMA  = 2.5

EDGE_ERODE_PX         = 3
EDGE_ERODE_FEATHER    = 2.5

PROTECT_SIGMA_FACTOR  = 1.3
PROTECT_SHARPNESS_THR = 0.20
PROTECT_SHARPNESS_BIAS= 1.5

AUTO_RANGE_MIN        = 0.08
AUTO_RANGE_MAX        = 0.35
AUTO_RANGE_SCALE      = 1.20
AUTO_RANGE_STD_CLAMP  = 0.30

AUTO_GAMMA_BASE       = 1.05
AUTO_GAMMA_MIN        = 0.80
AUTO_GAMMA_MAX        = 1.70
AUTO_GAMMA_ADJUST     = 2.8

AUTO_MAXBLUR_BASE     = 4.0
AUTO_MAXBLUR_SCALE    = 28.0
AUTO_MAXBLUR_MIN      = 2.5
AUTO_MAXBLUR_MAX      = 16.0

V3_MODEL_DIR = os.path.join(comfy_dir, "models", "depthanything3")


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

def _find_best_model_v3():
    priority = [
        "da3_large_1.1.safetensors",
        "da3metric_large.safetensors",
        "da3mono_large.safetensors",
        "da3_large.safetensors",
        "da3_base.safetensors",
        "da3_small.safetensors",
        "da3_giant_1.1.safetensors",
        "da3_giant.safetensors",
        "da3_nested_giant_large_1.1.safetensors",
        "da3nested_giant_large.safetensors"
    ]
    for name in priority:
        path = os.path.join(V3_MODEL_DIR, name)
        if os.path.exists(path):
            return name
    return None

def _find_best_model_V3():
    base = os.path.join(comfy_dir, "models", "depthanything3")
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
        base = os.path.join(comfy_dir, "models", "depthanything")
        raise RuntimeError(f"No Depth Anything V2 model found in {base}")

    model = DepthAnythingV2()
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict, strict=False)
    model.to(_device)
    model.eval()
    _depth_model = model
    return _depth_model

def _load_depth_model_v3():
    global _depth_model_v3
    if _depth_model_v3 is not None:
        return _depth_model_v3

    if not os.path.exists(V3_MODEL_DIR):
        raise RuntimeError(
            f"Depth Anything V3 model folder not found!\n"
            f"Create folder: {V3_MODEL_DIR}\n"
            f"Download ALL files from https://huggingface.co/depth-anything/DA3MONO-LARGE "
            f"and put them inside that folder."
        )

    model = DepthAnything3.from_pretrained(V3_MODEL_DIR)   # local folder only
    model = model.to(_device)
    model.eval()
    _depth_model_v3 = model
    return _depth_model_v3

def _load_local_depth_model_v3(model_name=None):
    global _depth_model_v3, _depth_model_v3_name
    if model_name is None:
        model_name = _find_best_model_v3()

    if model_name is None:
        raise RuntimeError(f"No Depth Anything V3 model found in {V3_MODEL_DIR}. ")

    if _depth_model_v3 is not None and _depth_model_v3_name == model_name:
        return _depth_model_v3

    model = load_model.DownloadAndLoadDepthAnythingV3Model.execute(model_name)
    _depth_model_v3 = model
    _depth_model_v3_name = model_name
    return _depth_model_v3

def _predict_depth(arr, imagei, use_v3: bool = False):
    if not use_v3:
        # === V2 (unchanged) ===
        model = _load_depth_model()
        h, w, _ = arr.shape
        img = Image.fromarray((arr * 255.0).astype(np.uint8))
        img = img.resize((518, 518))
        x = np.array(img).astype(np.float32) / 255.0
        x = (x - 0.5) / 0.5
        x = torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0).to(_device)

        with torch.no_grad():
            depth = model(x)

        depth = depth.squeeze().cpu().numpy()
        depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
        depth = 1.0 - depth
        depth = np.array(Image.fromarray((depth * 255).astype(np.uint8)).resize((w, h))).astype(np.float32) / 255.0
        depth = gaussian_filter(depth, sigma=DEPTH_MAP_BLUR_SIGMA)
        return depth

    model = _load_local_depth_model_v3()
    depth = nodes_inference.DepthAnything_V3.execute(model, imagei, normalization_mode="Raw", invert_depth=True,)
    h, w, _ = arr.shape

    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().float()
        if depth.dim() == 4:
            depth = depth[0]  # [H, W, C]
            if depth.shape[-1] > 1:
                depth = depth[..., 0]
        elif depth.dim() == 3:
            if depth.shape[0] in (1, 3):
                depth = depth[0]
            elif depth.shape[-1] in (1, 3):
                depth = depth[..., 0]
        depth = depth.numpy()
    elif isinstance(depth, np.ndarray):
        if depth.ndim == 4:
            depth = depth[0]
        if depth.ndim == 3:
            if depth.shape[0] in (1, 3):
                depth = depth[0]
            elif depth.shape[-1] in (1, 3):
                depth = depth[..., 0]
    else:
        depth = np.asarray(depth)
        if depth.ndim > 2:
            depth = np.squeeze(depth)
    if depth.ndim != 2:
        depth = np.squeeze(depth)

    # depth = depth.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    depth = 1.0 - depth
    depth = np.array(Image.fromarray((depth * 255).astype(np.uint8)).resize((w, h))).astype(np.float32) / 255.0
    depth = gaussian_filter(depth, sigma=DEPTH_MAP_BLUR_SIGMA)
    return depth


def _to_luminance(arr):
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def _sharpness_map(luma, radius=1.0):
    gx = gaussian_filter(luma, sigma=radius, order=[0, 1])
    gy = gaussian_filter(luma, sigma=radius, order=[1, 0])
    mag = np.sqrt(gx * gx + gy * gy)
    return mag / (mag.max() + 1e-6)


def _erode_focus_mask(depth_blur, erode_px, feather_px):
    if erode_px <= 0:
        return depth_blur
    focus_mask = depth_blur < 1e-4
    eroded_mask = binary_erosion(focus_mask, iterations=erode_px)
    boundary = focus_mask & (~eroded_mask)
    if not boundary.any():
        return depth_blur
    dist = distance_transform_edt(~eroded_mask)
    ramp = np.clip(dist / max(feather_px, 1e-3), 0.0, 1.0) * boundary
    boundary_blur = ramp * 0.15
    result = depth_blur.copy()
    result[focus_mask] = 0.0
    result[focus_mask] = np.maximum(result[focus_mask], boundary_blur[focus_mask])
    return result


def _build_protection_mask(raw_depth, focus_depth, protect_sigma, arr):
    luma = _to_luminance(arr)
    sharp = _sharpness_map(luma, radius=1.0)
    sharp_mask = np.clip((sharp - PROTECT_SHARPNESS_THR) * PROTECT_SHARPNESS_BIAS, 0.0, 1.0)
    focus_band = np.exp(-((raw_depth - focus_depth) ** 2) / (protect_sigma ** 2 + 1e-8))
    return np.maximum(sharp_mask, focus_band)


def img_depth_blur(
    image:         Image.Image,
    focus_depth:   float = 0.5,
    depth_range:   float = 0.2,
    max_blur:      float = 8.0,
    depth_gamma:   float = 1.0,
    auto_optimize: bool  = False,
    use_v3:        bool  = False,
) -> Image.Image:

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    arrV3 = utility.image_to_tensor(image)
    h, w, _ = arr.shape

    raw_depth = _predict_depth(arr, arrV3, use_v3=use_v3)

    if auto_optimize:
        depth_std = float(np.std(raw_depth))
        clamped_std = min(depth_std, AUTO_RANGE_STD_CLAMP)
        depth_range = float(np.clip(clamped_std * AUTO_RANGE_SCALE, AUTO_RANGE_MIN, AUTO_RANGE_MAX))
        depth_gamma = float(np.clip(AUTO_GAMMA_BASE + (0.22 - depth_std) * AUTO_GAMMA_ADJUST, AUTO_GAMMA_MIN, AUTO_GAMMA_MAX))
        far_deviation = float(np.percentile(np.abs(raw_depth - focus_depth), 92))
        max_blur = float(np.clip(AUTO_MAXBLUR_BASE + far_deviation * AUTO_MAXBLUR_SCALE, AUTO_MAXBLUR_MIN, AUTO_MAXBLUR_MAX))

    depth = raw_depth ** depth_gamma
    depth_blur = np.clip((depth - focus_depth) / (depth_range + 1e-6), 0.0, 1.0)

    protect_sigma = depth_range * PROTECT_SIGMA_FACTOR
    protect_mask = _build_protection_mask(raw_depth, focus_depth, protect_sigma, arr)
    depth_blur = depth_blur * (1.0 - protect_mask)

    depth_blur = _erode_focus_mask(depth_blur, EDGE_ERODE_PX, EDGE_ERODE_FEATHER)

    levels = 5
    sigmas = np.linspace(0.0, max_blur, levels)
    blurred_stack = [gaussian_filter(arr, sigma=(s, s, 0)) if s > 0 else arr for s in sigmas]
    blurred_stack = np.stack(blurred_stack, axis=0)

    idx = depth_blur * (levels - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.clip(i0 + 1, 0, levels - 1)
    f = (idx - i0)[..., np.newaxis]

    rows = np.arange(h)[:, np.newaxis]
    cols = np.arange(w)

    b0 = blurred_stack[i0, rows, cols]
    b1 = blurred_stack[i1, rows, cols]
    out = b0 * (1.0 - f) + b1 * f

    out = np.clip(out, 0.0, 1.0)
    return Image.fromarray((out * 255.0).astype(np.uint8), mode="RGB")