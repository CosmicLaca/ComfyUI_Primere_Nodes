import os
import numpy as np
import torch
from PIL import Image
from scipy.ndimage import gaussian_filter, binary_erosion, distance_transform_edt
from safetensors.torch import load_file

from ...utils import comfy_dir
from ...components.depth_anything_v2.dpt import DepthAnythingV2

_depth_model = None
_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── Tunable constants ────────────────────────────────────────────────────────

DEPTH_MAP_BLUR_SIGMA  = 2.5    # smoothing applied to raw depth map (halo reduction)

# Edge erosion — contracts the sharp/focus zone inward so the blur transition
# starts inside the object boundary rather than at the exact depth edge.
# This eliminates the thin bright halo visible outside sharp objects.
EDGE_ERODE_PX         = 3      # pixels to erode the focus mask inward (1–3 typical)
EDGE_ERODE_FEATHER    = 2.5    # distance in px over which eroded edge softly blends

# Protection — keeps focused subject fully sharp
PROTECT_SIGMA_FACTOR  = 1.3    # protection gaussian width = depth_range * this factor
PROTECT_SHARPNESS_THR = 0.20
PROTECT_SHARPNESS_BIAS= 1.5

# Auto-optimize constants — used only when auto_optimize=True
AUTO_RANGE_MIN        = 0.08
AUTO_RANGE_MAX        = 0.35
AUTO_RANGE_SCALE      = 1.20   # depth_range = depth_std * this
AUTO_RANGE_STD_CLAMP  = 0.30   # clamp depth_std before scaling (avoids outliers)

AUTO_GAMMA_BASE       = 1.05
AUTO_GAMMA_MIN        = 0.80
AUTO_GAMMA_MAX        = 1.70
AUTO_GAMMA_ADJUST     = 2.8

AUTO_MAXBLUR_BASE     = 4.0
AUTO_MAXBLUR_SCALE    = 28.0
AUTO_MAXBLUR_MIN      = 2.5
AUTO_MAXBLUR_MAX      = 16.0


# ── Model loading ─────────────────────────────────────────────────────────────

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
        base = os.path.join(comfy_dir, "models", "depthanything")
        raise RuntimeError(f"No Depth Anything model found. Download one to {base}.")
    model = DepthAnythingV2()
    state_dict = load_file(model_path)
    model.load_state_dict(state_dict, strict=False)
    model.to(_device)
    model.eval()
    _depth_model = model
    return _depth_model


# ── Depth prediction ──────────────────────────────────────────────────────────

def _predict_depth(arr):
    model = _load_depth_model()
    h, w, _ = arr.shape

    img = Image.fromarray((arr * 255.0).astype(np.uint8))
    img = img.resize((518, 518))
    x   = np.array(img).astype(np.float32) / 255.0
    x   = (x - 0.5) / 0.5
    x   = torch.from_numpy(np.transpose(x, (2, 0, 1))).unsqueeze(0).to(_device)

    with torch.no_grad():
        depth = model(x)

    depth = depth.squeeze().cpu().numpy()
    depth = (depth - depth.min()) / (depth.max() - depth.min() + 1e-6)

    # DepthAnythingV2 outputs disparity (close = HIGH). Invert so close = LOW.
    depth = 1.0 - depth

    # Resize back to original resolution
    depth = np.array(
        Image.fromarray((depth * 255).astype(np.uint8)).resize((w, h))
    ).astype(np.float32) / 255.0

    # Smooth depth map to reduce harsh transitions at object boundaries
    depth = gaussian_filter(depth, sigma=DEPTH_MAP_BLUR_SIGMA)

    return depth


# ── Helpers ───────────────────────────────────────────────────────────────────

def _to_luminance(arr):
    return 0.299 * arr[..., 0] + 0.587 * arr[..., 1] + 0.114 * arr[..., 2]


def _sharpness_map(luma, radius=1.0):
    gx  = gaussian_filter(luma, sigma=radius, order=[0, 1])
    gy  = gaussian_filter(luma, sigma=radius, order=[1, 0])
    mag = np.sqrt(gx * gx + gy * gy)
    return mag / (mag.max() + 1e-6)


def _erode_focus_mask(depth_blur, erode_px, feather_px):
    """
    Contract the in-focus zone inward by erode_px pixels so the blur
    transition starts inside the object boundary, eliminating the thin
    halo that appears just outside the sharp subject.

    Returns a modified depth_blur where the eroded boundary pixels
    receive a smoothly increasing blur amount over feather_px pixels.
    """
    if erode_px <= 0:
        return depth_blur

    focus_mask  = depth_blur < 1e-4        # boolean: True = in-focus pixel
    eroded_mask = binary_erosion(focus_mask, iterations=erode_px)

    # Pixels in original focus zone but outside the eroded zone = boundary band
    boundary    = focus_mask & (~eroded_mask)

    if not boundary.any():
        return depth_blur

    # Distance from eroded edge, only within the boundary band
    dist = distance_transform_edt(~eroded_mask)   # distance from eroded edge outward
    # Within boundary, dist ranges from 0 (eroded edge) to erode_px (original focus edge)
    # We want ramp: 0 at eroded edge → small blur at original focus edge
    ramp = np.clip(dist / max(feather_px, 1e-3), 0.0, 1.0) * boundary

    # Small blur amount at boundary pixels (ramp up to ~15% of max to stay subtle)
    boundary_blur = ramp * 0.15

    # Combine: eroded zone stays 0 (sharp), boundary zone gets gentle ramp,
    # existing blur zone keeps its values
    result = depth_blur.copy()
    result[focus_mask] = 0.0          # reset entire original focus zone
    result[focus_mask] = np.maximum(result[focus_mask], boundary_blur[focus_mask])

    return result


def _build_protection_mask(raw_depth, focus_depth, protect_sigma, arr):
    """
    Builds a mask in [0,1] where 1 = fully protected (no blur applied).
    Combines:
    - Gaussian band around focus_depth in depth space (subject thickness)
    - Sharpness-based protection (edges and texture near focus plane)
    """
    luma       = _to_luminance(arr)
    sharp      = _sharpness_map(luma, radius=1.0)
    sharp_mask = np.clip(
        (sharp - PROTECT_SHARPNESS_THR) * PROTECT_SHARPNESS_BIAS, 0.0, 1.0)

    focus_band = np.exp(
        -((raw_depth - focus_depth) ** 2) / (protect_sigma ** 2 + 1e-8))

    return np.maximum(sharp_mask, focus_band)


# ── Main function ─────────────────────────────────────────────────────────────

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

    raw_depth = _predict_depth(arr)

    # ── Auto-optimize ─────────────────────────────────────────────────────────
    if auto_optimize:
        depth_std = float(np.std(raw_depth))

        # depth_range: derived purely from depth statistics, no user input dependency
        clamped_std = min(depth_std, AUTO_RANGE_STD_CLAMP)
        depth_range = float(np.clip(
            clamped_std * AUTO_RANGE_SCALE,
            AUTO_RANGE_MIN, AUTO_RANGE_MAX))

        # depth_gamma: boosts contrast when depth variation is low
        depth_gamma = float(np.clip(
            AUTO_GAMMA_BASE + (0.22 - depth_std) * AUTO_GAMMA_ADJUST,
            AUTO_GAMMA_MIN, AUTO_GAMMA_MAX))

        # max_blur: driven by how far the background extends beyond focus
        far_deviation = float(np.percentile(np.abs(raw_depth - focus_depth), 92))
        max_blur = float(np.clip(
            AUTO_MAXBLUR_BASE + far_deviation * AUTO_MAXBLUR_SCALE,
            AUTO_MAXBLUR_MIN, AUTO_MAXBLUR_MAX))

    # ── Blur mask from depth ──────────────────────────────────────────────────
    depth       = raw_depth ** depth_gamma
    depth_blur  = np.clip((depth - focus_depth) / (depth_range + 1e-6), 0.0, 1.0)

    # ── Protection: keep focused subject fully sharp ───────────────────────────
    # Always applied — not gated by focus_depth threshold.
    # Sigma scales with depth_range so the protected band covers the full subject
    # regardless of whether we are in manual or auto mode.
    protect_sigma = depth_range * PROTECT_SIGMA_FACTOR
    protect_mask  = _build_protection_mask(raw_depth, focus_depth, protect_sigma, arr)
    depth_blur    = depth_blur * (1.0 - protect_mask)

    # ── Inward erosion of focus boundary (halo elimination) ──────────────────
    depth_blur = _erode_focus_mask(depth_blur, EDGE_ERODE_PX, EDGE_ERODE_FEATHER)

    # ── Multi-scale blur stack ────────────────────────────────────────────────
    levels  = 5
    sigmas  = np.linspace(0.0, max_blur, levels)

    blurred_stack = []
    for s in sigmas:
        blurred_stack.append(
            gaussian_filter(arr, sigma=(s, s, 0)) if s > 0 else arr)
    blurred_stack = np.stack(blurred_stack, axis=0)

    idx = depth_blur * (levels - 1)
    i0  = np.floor(idx).astype(int)
    i1  = np.clip(i0 + 1, 0, levels - 1)
    f   = (idx - i0)[..., np.newaxis]

    rows = np.arange(h)[:, np.newaxis]
    cols = np.arange(w)

    b0  = blurred_stack[i0, rows, cols]
    b1  = blurred_stack[i1, rows, cols]
    out = b0 * (1.0 - f) + b1 * f

    out = np.clip(out, 0.0, 1.0)
    return Image.fromarray((out * 255.0).astype(np.uint8), mode="RGB")
