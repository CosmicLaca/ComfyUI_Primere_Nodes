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

# ===================================================================
# AUTO-OPTIMIZE TUNABLE CONSTANTS
# You can edit these values directly in the code later if you want
# different behavior. No new user inputs are required.
# ===================================================================
AUTO_RANGE_MIN   = 0.08      # shallowest possible auto depth_range
AUTO_RANGE_MAX   = 0.35      # widest possible auto depth_range
AUTO_RANGE_SCALE = 1.15      # how much of the depth std we use for range
AUTO_BLUR_FACTOR = 0.009     # stronger max_blur → slightly tighter DOF

AUTO_GAMMA_BASE   = 1.05     # starting gamma
AUTO_GAMMA_MIN    = 0.80
AUTO_GAMMA_MAX    = 1.70
AUTO_GAMMA_ADJUST = 2.8      # how strongly low depth variation increases gamma

AUTO_MAXBLUR_BASE  = 4.0
AUTO_MAXBLUR_SCALE = 28.0    # how strongly "far" depths increase the blur
AUTO_MAXBLUR_MIN   = 2.5
AUTO_MAXBLUR_MAX   = 16.0

# Protection when focus_depth is low (foreground focus)
AUTO_PROTECT_FOCUS_THRESHOLD = 0.4      # activate protection below this value
AUTO_SHARPNESS_THRESHOLD     = 0.2
AUTO_SHARPNESS_BIAS          = 1.5
AUTO_PROTECT_BAND_SIGMA      = 0.12     # depth-band width around focus_depth
                                         # (higher = wider protection on character)


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

    img = img.resize((518, 518))
    x = np.array(img).astype(np.float32) / 255.0
    x = (x - 0.5) / 0.5
    x = np.transpose(x, (2, 0, 1))
    x = torch.from_numpy(x).unsqueeze(0).to(_device)

    with torch.no_grad():
        depth = model(x)

    depth = depth.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)
    depth = 1.0 - depth
    depth = Image.fromarray((depth * 255).astype(np.uint8)).resize((w, h))
    depth = np.array(depth).astype(np.float32) / 255.0

    # Small blur ONLY on the depth map (eliminates the thin white edge halo)
    # Does NOT touch the final image sharpness at all
    depth = gaussian_filter(depth, sigma=2.0)

    return depth


def _to_luminance(arr):
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def _sharpness_map(luma, radius):
    gx = gaussian_filter(luma, sigma=radius, order=[0, 1])
    gy = gaussian_filter(luma, sigma=radius, order=[1, 0])
    mag = np.sqrt(gx * gx + gy * gy)
    return mag / (mag.max() + 1e-6)


def img_depth_blur(
    image:         Image.Image,
    focus_depth:   float = 0.5,
    depth_range:   float = 0.2,
    max_blur:      float = 8.0,
    depth_gamma:   float = 1.0,
    auto_optimize: bool  = False,
) -> Image.Image:

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0
    h, w, _ = arr.shape

    # Raw depth (before any gamma) – used for protection logic
    raw_depth = _predict_depth(arr)

    # ===================================================================
    # AUTO-OPTIMIZE LOGIC
    # ===================================================================
    if auto_optimize:
        depth_std = float(np.std(raw_depth))

        # 1. depth_range adapts to image content
        depth_range = max(AUTO_RANGE_MIN, min(AUTO_RANGE_MAX,
                         depth_std * AUTO_RANGE_SCALE - max_blur * AUTO_BLUR_FACTOR))

        # 2. depth_gamma boosts contrast when depth variation is low
        depth_gamma = max(AUTO_GAMMA_MIN, min(AUTO_GAMMA_MAX,
                         AUTO_GAMMA_BASE + (0.22 - depth_std) * AUTO_GAMMA_ADJUST))

        # 3. max_blur calculated automatically from focus_depth
        far_deviation = np.percentile(np.abs(raw_depth - focus_depth), 92)
        max_blur = AUTO_MAXBLUR_BASE + far_deviation * AUTO_MAXBLUR_SCALE
        max_blur = max(AUTO_MAXBLUR_MIN, min(AUTO_MAXBLUR_MAX, max_blur))

    # Apply gamma to the depth map used for blur calculation
    depth = raw_depth ** depth_gamma

    # Depth-based blur amount: 0 = perfectly in focus, 1 = maximum blur
    depth_blur = np.clip((depth - focus_depth) / (depth_range + 1e-6), 0.0, 1.0)

    # ===================================================================
    # IMPROVED PROTECTION (now works in BOTH manual and fully auto mode)
    # - Uses raw_depth (pre-gamma) so the band is always accurate
    # - Protects the ENTIRE character/object near the focus plane
    #   (smooth skin, neck, head, thin/narrow parts)
    # - Works perfectly when auto_optimize=True and focus_depth < 0.4
    # ===================================================================
    if focus_depth < AUTO_PROTECT_FOCUS_THRESHOLD:
        luma = _to_luminance(arr)
        sharp = _sharpness_map(luma, radius=1.0)
        sharp_mask = np.clip((sharp - AUTO_SHARPNESS_THRESHOLD) * AUTO_SHARPNESS_BIAS, 0.0, 1.0)

        # Soft band around the exact focus plane (protects whole subject)
        focus_proximity = np.exp(-((raw_depth - focus_depth) ** 2) / (AUTO_PROTECT_BAND_SIGMA ** 2))

        # Combine both masks (maximum = strongest protection)
        protect_mask = np.maximum(sharp_mask, focus_proximity)

        # Apply protection (no blur on the focused character)
        depth_blur = depth_blur * (1.0 - protect_mask)

    # Variable blur via stacked Gaussian levels + linear interpolation
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

    idx = depth_blur * (levels - 1)
    i0 = np.floor(idx).astype(int)
    i1 = np.clip(i0 + 1, 0, levels - 1)
    f = idx - i0

    out = np.zeros_like(arr)
    for c in range(3):
        b0 = blurred_stack[i0, np.arange(h)[:, None], np.arange(w), c]
        b1 = blurred_stack[i1, np.arange(h)[:, None], np.arange(w), c]
        out[..., c] = b0 * (1 - f) + b1 * f

    out = np.clip(out, 0.0, 1.0)
    return Image.fromarray((out * 255.0).astype(np.uint8), mode="RGB")