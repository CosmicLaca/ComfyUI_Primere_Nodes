import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _to_tensor(img):
    if isinstance(img, Image.Image):
        arr = np.array(img.convert("RGB")).astype(np.float32) / 255.0
        t = torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)
        return t
    return img


def _to_image(t):
    t = t.clamp(0, 1).squeeze(0).permute(1, 2, 0).cpu().numpy()
    return Image.fromarray((t * 255).astype(np.uint8))


def _meshgrid(H, W, device):
    yy, xx = torch.meshgrid(
        torch.linspace(0, 1, H, device=device),
        torch.linspace(0, 1, W, device=device),
        indexing="ij"
    )
    return yy, xx


def _radial(H, W, device):
    yy, xx = _meshgrid(H, W, device)
    r = torch.sqrt((yy - 0.5) ** 2 + (xx - 0.5) ** 2)
    return r / r.max()


def _sample(img, grid):
    return F.grid_sample(img, grid, mode="bilinear", padding_mode="border", align_corners=True)


def _gaussian_blur(img, sigma):
    if sigma <= 0:
        return img
    k = int(max(3, sigma * 6))
    if k % 2 == 0:
        k += 1
    kernel = torch.arange(k, device=img.device) - k // 2
    kernel = torch.exp(-0.5 * (kernel / sigma) ** 2)
    kernel /= kernel.sum()

    kernel_x = kernel.view(1, 1, 1, -1)
    kernel_y = kernel.view(1, 1, -1, 1)

    img = F.conv2d(img, kernel_x.expand(img.size(1), 1, 1, k), padding=(0, k // 2), groups=img.size(1))
    img = F.conv2d(img, kernel_y.expand(img.size(1), 1, k, 1), padding=(k // 2, 0), groups=img.size(1))
    return img


# ──────────────────────────────────────────────────────────────────────────────
# Effects
# ──────────────────────────────────────────────────────────────────────────────

def _distortion(img, barrel, pincushion, zoom):
    B, C, H, W = img.shape
    device = img.device
    yy, xx = _meshgrid(H, W, device)

    xn = (xx - 0.5) * 2
    yn = (yy - 0.5) * 2
    r2 = xn**2 + yn**2

    k1 = barrel - pincushion
    k2 = pincushion

    zoom = max(zoom, 0.01)
    d = (1.0 + k1 * r2 + k2 * r2**2) / zoom

    grid = torch.stack([xn * d, yn * d], dim=-1).unsqueeze(0)
    return _sample(img, grid)


def _vignette(img, strength, radius, feather, shape):
    B, C, H, W = img.shape
    device = img.device
    yy, xx = _meshgrid(H, W, device)

    if shape == "oval":
        r = torch.sqrt(((yy - 0.5) / 0.6) ** 2 + ((xx - 0.5) / 0.45) ** 2)
    elif shape == "corner":
        r = torch.max(torch.abs(yy - 0.5), torch.abs(xx - 0.5)) * 2
    else:
        r = _radial(H, W, device)

    fw = max(feather * 0.5, 1e-3)
    t = ((r - radius) / fw).clamp(0, 1)
    t = t * t * (3 - 2 * t)

    return img * (1 - strength * t).unsqueeze(0).unsqueeze(0)


def _chroma(img, intensity, falloff, mode):
    B, C, H, W = img.shape
    device = img.device
    yy, xx = _meshgrid(H, W, device)

    xn = (xx - 0.5)
    yn = (yy - 0.5)
    r = torch.sqrt(xn**2 + yn**2) * (1 + falloff)

    def shift(scale):
        gx = (xn + xn * r * intensity * scale) * 2
        gy = (yn + yn * r * intensity * scale) * 2
        grid = torch.stack([gx, gy], dim=-1).unsqueeze(0)
        return _sample(img, grid)

    if mode == "green_magenta":
        return torch.cat([shift(-0.5)[:,0:1], shift(1.0)[:,1:2], shift(-0.5)[:,2:3]], dim=1)
    elif mode == "yellow_purple":
        return torch.cat([shift(0.8)[:,0:1], shift(0.3)[:,1:2], shift(-0.8)[:,2:3]], dim=1)
    else:
        return torch.cat([shift(1.0)[:,0:1], img[:,1:2], shift(-0.7)[:,2:3]], dim=1)


def _bokeh(img, radius, blades, boost, cat_eye):
    base = _gaussian_blur(img, radius)

    if boost > 0:
        lum = 0.299 * img[:,0] + 0.587 * img[:,1] + 0.114 * img[:,2]
        hi = ((lum - 0.75) / 0.25).clamp(0, 1).unsqueeze(1)
        base = base + hi * boost * base

    if cat_eye > 0:
        B, C, H, W = img.shape
        r = _radial(H, W, img.device).unsqueeze(0).unsqueeze(0)
        squeezed = _gaussian_blur(img, radius) * torch.tensor([1,1,1], device=img.device).view(1,3,1,1)
        base = base * (1 - r * cat_eye) + squeezed * (r * cat_eye)

    return base.clamp(0, 1)


def _flare(img, intensity, px, py, streaks, length, ghosts, color):
    B, C, H, W = img.shape
    device = img.device

    TINTS = {
        "warm": torch.tensor([1.0,0.85,0.6], device=device),
        "cool": torch.tensor([0.6,0.8,1.0], device=device),
        "neutral": torch.tensor([1.0,1.0,1.0], device=device),
        "rainbow": torch.tensor([1.0,0.9,0.7], device=device),
    }

    tint = TINTS.get(color, TINTS["warm"]).view(1,3,1,1)

    yy, xx = _meshgrid(H, W, device)
    x = xx * W
    y = yy * H

    px *= W
    py *= H

    flare = torch.zeros_like(img)

    for i in range(streaks):
        a = np.pi * i / streaks
        ca, sa = np.cos(a), np.sin(a)
        dx = x - px
        dy = y - py

        proj = dx * ca + dy * sa
        perp = torch.abs(-dx * sa + dy * ca)

        ml = length * max(H, W)
        streak = torch.exp(-0.5*(perp/1.2)**2) * torch.exp(-0.5*(proj/(ml*0.3+1e-3))**2)
        flare += streak.unsqueeze(0).unsqueeze(0) * tint

    m = flare.max().clamp(min=1e-3)
    flare = flare / m * intensity

    return (img + flare).clamp(0, 1)


def _halation(img, intensity, radius, threshold, warmth):
    lum = 0.299*img[:,0] + 0.587*img[:,1] + 0.114*img[:,2]
    den = max(1.0 - threshold, 1e-3)
    hi = ((lum - threshold) / den).clamp(0,1)

    glow = torch.cat([
        _gaussian_blur(img[:,0:1]*hi.unsqueeze(1)*(1+warmth*0.6), radius),
        _gaussian_blur(img[:,1:2]*hi.unsqueeze(1)*(1+warmth*0.1), radius),
        _gaussian_blur(img[:,2:3]*hi.unsqueeze(1)*(1-warmth*0.5), radius),
    ], dim=1)

    return (img + glow * intensity).clamp(0,1)


def _focus(img, mode, radius, pos, width, feather):
    B,C,H,W = img.shape
    yy, xx = _meshgrid(H, W, img.device)

    if mode == "horizontal":
        dist = torch.abs(yy - pos)
    elif mode == "vertical":
        dist = torch.abs(xx - pos)
    elif mode == "radial":
        dist = torch.sqrt((yy-pos)**2 + (xx-pos)**2)
    else:
        dist = torch.sqrt(((yy-pos)/0.4)**2 + ((xx-0.5)/0.6)**2)

    hw = width / 2
    f = max(feather * width, 1e-3)

    t = ((dist-hw)/f).clamp(0,1)
    t = t*t*(3-2*t)

    blurred = _gaussian_blur(img, radius)
    return img*(1-t).unsqueeze(0).unsqueeze(0) + blurred*t.unsqueeze(0).unsqueeze(0)


def _spherical(img, intensity, radius, zone):
    B,C,H,W = img.shape
    blurred = _gaussian_blur(img, radius)

    if zone == "global":
        bmap = torch.ones((H,W), device=img.device)
    elif zone == "edge":
        bmap = _radial(H,W,img.device)
    else:
        bmap = 1 - _radial(H,W,img.device)

    blend = (bmap * intensity).unsqueeze(0).unsqueeze(0)
    return (img*(1-blend) + blurred*blend).clamp(0,1)


def _anamorphic(img, intensity, color, length, oval, blue_bias):
    B,C,H,W = img.shape
    device = img.device

    TINTS = {
        "blue": torch.tensor([0.5,0.7,1.0], device=device),
        "warm": torch.tensor([1.0,0.85,0.5], device=device),
        "white": torch.tensor([1.0,1.0,1.0], device=device),
    }
    tint = TINTS.get(color, TINTS["blue"]).view(1,3,1,1)

    lum = 0.299*img[:,0] + 0.587*img[:,1] + 0.114*img[:,2]
    hi = ((lum-0.7)/0.3).clamp(0,1)

    slen = max(3, int(W * length * 0.5))
    kx = torch.arange(-slen, slen+1, device=device)
    decay = max(slen*0.25, 1.0)
    kern = torch.exp(-torch.abs(kx)/decay)
    kern /= kern.sum()
    kern = kern.view(1,1,1,-1)

    flare = F.conv2d(img*hi.unsqueeze(1), kern.expand(C,1,1,-1), padding=(0,slen), groups=C)
    result = (img + flare * tint * intensity).clamp(0,1)

    if oval > 0:
        vblur = _gaussian_blur(img, [6.0*oval, 0.5] if isinstance(oval,float) else oval)
        result = result*(1-oval*0.5) + vblur*(oval*0.5)

    if blue_bias > 0:
        smask = (1.0 - lum/0.4).clamp(0,1)
        result[:,2] = (result[:,2] + smask*blue_bias*0.15).clamp(0,1)

    return result


# ──────────────────────────────────────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────────────────────────────────────

def img_lens_effect(
    image: Image.Image,
    # ── Vignette ──────────────────────────────────────────────────────────────
    vignette_strength:          float = 0.0,
    vignette_radius:            float = 0.65,
    vignette_feather:           float = 0.4,
    vignette_shape:             str   = "circular",
    # ── Chromatic Aberration ──────────────────────────────────────────────────
    chroma_intensity:           float = 0.0,
    chroma_falloff:             float = 0.5,
    chroma_fringe_color:        str   = "red_blue",
    # ── Bokeh ─────────────────────────────────────────────────────────────────
    bokeh_radius:               float = 0.0,
    bokeh_blades:               int   = 0,
    bokeh_highlight_boost:      float = 0.3,
    bokeh_cat_eye:              float = 0.0,
    # ── Lens Distortion ───────────────────────────────────────────────────────
    distortion_barrel:          float = 0.0,
    distortion_pincushion:      float = 0.0,
    distortion_zoom:            float = 1.0,
    # ── Lens Flare ────────────────────────────────────────────────────────────
    flare_intensity:            float = 0.0,
    flare_pos_x:                float = 0.2,
    flare_pos_y:                float = 0.2,
    flare_streak_count:         int   = 6,
    flare_streak_length:        float = 0.4,
    flare_ghost_count:          int   = 4,
    flare_color:                str   = "warm",
    # ── Halation ─────────────────────────────────────────────────────────────
    halation_intensity:         float = 0.0,
    halation_radius:            float = 15.0,
    halation_threshold:         float = 0.75,
    halation_warmth:            float = 0.7,
    # ── Focus Falloff ─────────────────────────────────────────────────────────
    focus_blur_radius:          float = 0.0,
    focus_mode:                 str   = "horizontal",
    focus_pos:                  float = 0.5,
    focus_width:                float = 0.2,
    focus_feather:              float = 0.3,
    # ── Spherical Aberration ──────────────────────────────────────────────────
    spherical_intensity:        float = 0.0,
    spherical_radius:           float = 3.0,
    spherical_zone:             str   = "centre",
    # ── Anamorphic ────────────────────────────────────────────────────────────
    anamorphic_intensity:       float = 0.0,
    anamorphic_streak_color:    str   = "blue",
    anamorphic_streak_length:   float = 0.8,
    anamorphic_oval_bokeh:      float = 0.4,
    anamorphic_blue_bias:       float = 0.3,
) -> Image.Image:
    img = image.convert("RGB")
    if distortion_barrel != 0 or distortion_pincushion != 0:
        img = _distortion(img, distortion_barrel, distortion_pincushion, distortion_zoom)

    if focus_blur_radius > 0:
        img = _focus(img, focus_mode, focus_blur_radius, focus_pos, focus_width, focus_feather)

    if bokeh_radius > 0:
        img = _bokeh(img, bokeh_radius, bokeh_blades, bokeh_highlight_boost, bokeh_cat_eye)

    if spherical_intensity > 0:
        img = _spherical(img, spherical_intensity, spherical_radius, spherical_zone)

    if chroma_intensity > 0:
        img = _chroma(img, chroma_intensity, chroma_falloff, chroma_fringe_color)

    if anamorphic_intensity > 0:
        img = _anamorphic(img, anamorphic_intensity, anamorphic_streak_color,
                          anamorphic_streak_length, anamorphic_oval_bokeh,
                          anamorphic_blue_bias)

    if halation_intensity > 0:
        img = _halation(img, halation_intensity, halation_radius,
                        halation_threshold, halation_warmth)

    if flare_intensity > 0:
        img = _flare(img, flare_intensity, flare_pos_x, flare_pos_y,
                     flare_streak_count, flare_streak_length,
                     flare_ghost_count, flare_color)

    if vignette_strength > 0:
        img = _vignette(img, vignette_strength, vignette_radius,
                        vignette_feather, vignette_shape)

    return _to_image(img)