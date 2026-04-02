import os
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_erosion, distance_transform_edt
from safetensors.torch import load_file

from ...utils import comfy_dir
from ...components.depth_anything_v2.dpt import DepthAnythingV2
from ...components.depth_anything_v3 import load_model as load_model
from ...components.depth_anything_v3 import nodes_inference as nodes_inference
from ...components import utility

_depth_model         = None
_depth_model_v3      = None
_depth_model_v3_name = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

REFERENCE_AREA           = 1024 * 1024

DEPTH_MAP_BLUR_BASE      = 2.5
DEPTH_MAP_BLUR_MIN       = 1.0
DEPTH_MAP_BLUR_MAX       = 5.0

EDGE_ERODE_BASE          = 2
EDGE_ERODE_MIN           = 1
EDGE_ERODE_MAX           = 5
EDGE_ERODE_FEATHER_BASE  = 2.5

PROTECT_SIGMA_FACTOR     = 1.5
PROTECT_SHARPNESS_THR    = 0.20
PROTECT_SHARPNESS_BIAS   = 1.5
PROTECT_BLUR_SIGMA       = 3.0

AUTO_RANGE_MIN           = 0.08
AUTO_RANGE_MAX           = 0.35
AUTO_RANGE_SCALE         = 1.20
AUTO_RANGE_STD_CLAMP     = 0.30
AUTO_GAMMA_BASE          = 1.05
AUTO_GAMMA_MIN           = 0.80
AUTO_GAMMA_MAX           = 1.70
AUTO_GAMMA_ADJUST        = 2.8
AUTO_MAXBLUR_BASE        = 4.0
AUTO_MAXBLUR_SCALE       = 32.0
AUTO_MAXBLUR_MIN         = 2.5
AUTO_MAXBLUR_MAX         = 16.0

V3_MODEL_DIR = os.path.join(comfy_dir, "models", "depthanything3")

V2_ValidModels = [
    "depth_anything_v2_vitl_fp32.safetensors",
    "depth_anything_v2_vitl_fp16.safetensors",
    "depth_anything_v2_vitb_fp32.safetensors",
    "depth_anything_v2_vitb_fp16.safetensors",
    "depth_anything_v2_vits_fp32.safetensors",
    "depth_anything_v2_vits_fp16.safetensors",
]

def _find_best_model_v2(DA_model):
    base = os.path.join(comfy_dir, "models", "depthanything")
    model_sign = f"_vit{DA_model[0]}"

    for name in V2_ValidModels:
        path = os.path.join(base, name)
        if os.path.exists(path):
            if model_sign.lower() in name.lower():
                return path

    for name in V2_ValidModels:
        path = os.path.join(base, name)
        if os.path.exists(path):
            return path
    return None

V3_ValidModels = [
    "da3_large_1.1.safetensors",
    "da3metric_large.safetensors",
    "da3mono_large.safetensors",
    "da3_large.safetensors",
    "da3_base.safetensors",
    "da3_small.safetensors",
    "da3_giant_1.1.safetensors",
    "da3_giant.safetensors",
    "da3_nested_giant_large_1.1.safetensors",
    "da3nested_giant_large.safetensors",
]

def _find_best_model_v3(DA_model):
    for name in V3_ValidModels:
        if os.path.exists(os.path.join(V3_MODEL_DIR, name)):
            if DA_model in name:
                return name

    for name in V3_ValidModels:
        if os.path.exists(os.path.join(V3_MODEL_DIR, name)):
            return name

    return None


def _load_depth_model_v2(DA_model):
    global _depth_model
    if _depth_model is not None:
       return _depth_model
    path = _find_best_model_v2(DA_model)
    if path is None:
        raise RuntimeError(f"No Depth Anything V2 model found in "f"{os.path.join(comfy_dir, 'models', 'depthanything')}")
    model = DepthAnythingV2()
    model.load_state_dict(load_file(path), strict=False)
    model.to(_device).eval()
    _depth_model = model
    return _depth_model


def _load_depth_model_v3(DA_model):
    global _depth_model_v3, _depth_model_v3_name
    name = _find_best_model_v3(DA_model)
    if name is None:
        raise RuntimeError(f"No Depth Anything V3 model found in {V3_MODEL_DIR}.")
    if _depth_model_v3 is not None and _depth_model_v3_name == name:
        return _depth_model_v3
    model = load_model.DownloadAndLoadDepthAnythingV3Model.execute(name)
    _depth_model_v3      = model
    _depth_model_v3_name = name
    return _depth_model_v3


def _depth_to_2d(depth):
    if isinstance(depth, torch.Tensor):
        depth = depth.detach().cpu().float()
    depth = np.asarray(depth, dtype=np.float32)
    while depth.ndim > 2:
        if depth.ndim == 4:
            depth = depth[0]
        elif depth.ndim == 3 and depth.shape[0] in (1, 3):
            depth = depth[0]
        elif depth.ndim == 3 and depth.shape[-1] in (1, 3):
            depth = depth[..., 0]
        else:
            depth = np.squeeze(depth)
            break
    return depth


def _res_scale(H, W):
    return float(np.sqrt((H * W) / REFERENCE_AREA))


def _postprocess_depth(raw, H, W):
    scale = _res_scale(H, W)
    sigma = float(np.clip(DEPTH_MAP_BLUR_BASE * scale, DEPTH_MAP_BLUR_MIN, DEPTH_MAP_BLUR_MAX))
    raw   = (raw - raw.min()) / (raw.max() - raw.min() + 1e-6)
    raw   = 1.0 - raw
    raw   = np.array(
        Image.fromarray((raw * 255).astype(np.uint8)).resize((W, H))
    ).astype(np.float32) / 255.0
    raw   = gaussian_filter(raw, sigma=sigma)
    return raw


def _predict_depth(arr, image_tensor, use_v3, DA_model, H, W):
    if not use_v3:
        model = _load_depth_model_v2(DA_model)
        img   = Image.fromarray((arr * 255.0).astype(np.uint8)).resize((518, 518))
        x     = np.array(img).astype(np.float32) / 255.0
        x     = (x - 0.5) / 0.5
        x     = torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0).to(_device)
        with torch.no_grad():
            raw = model(x)
        return _postprocess_depth(_depth_to_2d(raw), H, W)

    model = _load_depth_model_v3(DA_model)
    raw   = nodes_inference.DepthAnything_V3.execute(model, image_tensor, normalization_mode="Raw", invert_depth=True)
    return _postprocess_depth(_depth_to_2d(raw), H, W)


def _to_luminance(arr):
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def _sharpness_map(luma, radius=1.0):
    gx  = gaussian_filter(luma, sigma=radius, order=[0, 1])
    gy  = gaussian_filter(luma, sigma=radius, order=[1, 0])
    mag = np.sqrt(gx * gx + gy * gy)
    return mag / (mag.max() + 1e-6)


def _build_protection_mask(raw_depth, focus_depth, depth_range, arr, H, W):
    sigma = depth_range * PROTECT_SIGMA_FACTOR
    gaussian_protect = np.exp(-((raw_depth - focus_depth) ** 2) / (sigma ** 2))
    hard_floor       = (raw_depth <= focus_depth).astype(np.float32)
    depth_protect    = np.maximum(gaussian_protect, hard_floor)

    luma       = _to_luminance(arr)
    sharp      = _sharpness_map(luma, radius=1.0)
    sharp_mask = np.clip(
        (sharp - PROTECT_SHARPNESS_THR) * PROTECT_SHARPNESS_BIAS, 0.0, 1.0)

    combined = np.maximum(depth_protect, sharp_mask)

    blur_sigma = float(np.clip(
        PROTECT_BLUR_SIGMA * _res_scale(H, W), 1.0, 8.0))
    combined = gaussian_filter(combined, sigma=blur_sigma)
    return np.clip(combined, 0.0, 1.0)


def _erode_focus_mask(depth_blur, H, W):
    scale     = _res_scale(H, W)
    erode_px  = int(np.clip(round(EDGE_ERODE_BASE * scale), EDGE_ERODE_MIN, EDGE_ERODE_MAX))
    feather   = float(EDGE_ERODE_FEATHER_BASE * scale)

    if erode_px <= 0:
        return depth_blur

    focus_mask  = depth_blur < 1e-4
    eroded_mask = binary_erosion(focus_mask, iterations=erode_px)
    boundary    = focus_mask & (~eroded_mask)

    if not boundary.any():
        return depth_blur

    dist          = distance_transform_edt(~eroded_mask)
    ramp          = np.clip(dist / max(feather, 1e-3), 0.0, 1.0) * boundary
    result        = depth_blur.copy()
    result[boundary] = ramp[boundary] * 0.15
    return result


def img_depth_blur(
    image:         Image.Image,
    focus_depth:   float = 0.5,
    depth_range:   float = 0.2,
    max_blur:      float = 8.0,
    depth_gamma:   float = 1.0,
    auto_optimize: bool  = False,
    use_v3:        bool  = False,
    DA_model:      str = "large",
) -> Image.Image:

    img          = image.convert("RGB")
    arr          = np.array(img, dtype=np.float32) / 255.0
    image_tensor = utility.image_to_tensor(image)
    h, w, _      = arr.shape

    raw_depth = _predict_depth(arr, image_tensor, use_v3, DA_model, h, w)

    if auto_optimize:
        depth_std   = float(np.std(raw_depth))
        clamped_std = min(depth_std, AUTO_RANGE_STD_CLAMP)
        depth_range = float(np.clip(
            clamped_std * AUTO_RANGE_SCALE, AUTO_RANGE_MIN, AUTO_RANGE_MAX))
        depth_gamma = float(np.clip(
            AUTO_GAMMA_BASE + (0.22 - depth_std) * AUTO_GAMMA_ADJUST,
            AUTO_GAMMA_MIN, AUTO_GAMMA_MAX))
        far_mask = raw_depth > (focus_depth + depth_range)
        far_dev  = float(np.percentile(
            raw_depth[far_mask] - (focus_depth + depth_range), 90
        )) if far_mask.any() else 0.0
        max_blur = float(np.clip(
            AUTO_MAXBLUR_BASE + far_dev * AUTO_MAXBLUR_SCALE,
            AUTO_MAXBLUR_MIN, AUTO_MAXBLUR_MAX))

    depth      = raw_depth ** depth_gamma
    depth_blur = np.clip((depth - focus_depth) / (depth_range + 1e-6), 0.0, 1.0)

    protect_mask = _build_protection_mask(raw_depth, focus_depth, depth_range, arr, h, w)
    depth_blur   = depth_blur * (1.0 - protect_mask)
    depth_blur   = _erode_focus_mask(depth_blur, h, w)

    levels        = 5
    sigmas        = np.linspace(0.0, max_blur, levels)
    blurred_stack = np.stack([
        gaussian_filter(arr, sigma=(s, s, 0)) if s > 0 else arr
        for s in sigmas
    ], axis=0)

    idx  = depth_blur * (levels - 1)
    i0   = np.floor(idx).astype(int)
    i1   = np.clip(i0 + 1, 0, levels - 1)
    f    = (idx - i0)[..., np.newaxis]
    rows = np.arange(h)[:, np.newaxis]
    cols = np.arange(w)

    out = np.clip(
        blurred_stack[i0, rows, cols] * (1.0 - f) +
        blurred_stack[i1, rows, cols] * f,
        0.0, 1.0
    )
    return Image.fromarray((out * 255.0).astype(np.uint8), mode="RGB")