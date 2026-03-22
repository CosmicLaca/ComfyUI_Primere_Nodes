import numpy as np
from PIL import Image
from scipy.ndimage import gaussian_filter, map_coordinates, convolve

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
        img = _focus_falloff(img, focus_mode, focus_blur_radius,
                             focus_pos, focus_width, focus_feather)

    if bokeh_radius > 0:
        img = _bokeh(img, bokeh_radius, bokeh_blades,
                     bokeh_highlight_boost, bokeh_cat_eye)

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

    return img


def _f(img):
    return np.array(img.convert("RGB"), dtype=np.float32) / 255.0

def _p(arr):
    return Image.fromarray(np.clip(arr * 255, 0, 255).astype(np.uint8), mode="RGB")

def _radial(H, W):
    cy, cx = H / 2.0, W / 2.0
    y, x   = np.mgrid[0:H, 0:W]
    return np.sqrt(((y-cy)/cy)**2 + ((x-cx)/cx)**2) / np.sqrt(2.0)


def _vignette(img, strength, radius, feather, shape):
    arr  = _f(img)
    H, W = arr.shape[:2]
    cy, cx = H / 2.0, W / 2.0
    y, x   = np.mgrid[0:H, 0:W]

    if shape == "oval":
        dist = np.sqrt(((y-cy)/(cy*1.2))**2 + ((x-cx)/(cx*0.9))**2)
        dist /= np.sqrt(2.0) / 1.05
    elif shape == "corner":
        dist = np.maximum(np.abs((y-cy)/cy), np.abs((x-cx)/cx))
    else:
        dist = np.sqrt(((y-cy)/cy)**2 + ((x-cx)/cx)**2) / np.sqrt(2.0)

    fw = max(feather * 0.5, 0.01)
    t  = np.clip((dist - radius) / fw, 0, 1)
    t  = t * t * (3 - 2 * t)
    return _p(arr * (1.0 - strength * t)[..., np.newaxis])


def _chroma(img, intensity, falloff, fringe_color):
    arr  = _f(img)
    H, W = arr.shape[:2]
    cy, cx = H / 2.0, W / 2.0
    y, x   = np.mgrid[0:H, 0:W]
    dy = (y - cy) / cy
    dx = (x - cx) / cx
    r  = np.sqrt(dy**2 + dx**2) ** (1.0 + falloff * 2.0)

    def shift(ch, scale):
        sy = np.clip(y + dy * r * intensity * scale, 0, H-1)
        sx = np.clip(x + dx * r * intensity * scale, 0, W-1)
        return map_coordinates(arr[..., ch], [sy, sx], order=1,
                               mode='nearest').astype(np.float32)

    if fringe_color == "red_blue":
        R, G, B = shift(0, +1.0), arr[...,1].copy(), shift(2, -0.7)
    elif fringe_color == "green_magenta":
        R, G, B = shift(0, -0.5), shift(1, +1.0), shift(2, -0.5)
    else:
        R, G, B = shift(0, +0.8), shift(1, +0.3), shift(2, -0.8)

    return _p(np.stack([R, G, B], axis=-1))


def _bokeh_kernel(radius, blades, squeeze=1.0):
    ri   = max(1, int(np.ceil(radius)))
    sz   = 2 * ri + 1
    yy, xx = np.mgrid[-ri:ri+1, -ri:ri+1]
    if blades < 3:
        k = (xx**2 / squeeze + yy**2 <= radius**2).astype(np.float32)
    else:
        angles = np.linspace(0, 2*np.pi, blades, endpoint=False) + np.pi/blades
        k = np.ones((sz, sz), dtype=np.float32)
        for a in angles:
            k *= (xx*np.cos(a) + yy*np.sin(a) <= radius*0.92).astype(np.float32)
    s = k.sum()
    return k / s if s > 0 else k


def _bokeh(img, radius, blades, highlight_boost, cat_eye):
    arr  = _f(img)
    H, W = arr.shape[:2]
    kernel  = _bokeh_kernel(radius, blades)
    blurred = np.stack([convolve(arr[...,c], kernel) for c in range(3)], axis=-1)

    if highlight_boost > 0:
        lum     = 0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]
        hi      = np.clip((lum - 0.75) / 0.25, 0, 1)[..., np.newaxis]
        blurred = blurred + hi * highlight_boost * blurred

    if cat_eye > 0:
        k2       = _bokeh_kernel(radius, blades, squeeze=0.6)
        squeezed = np.stack([convolve(arr[...,c], k2) for c in range(3)], axis=-1)
        blend    = np.clip(_radial(H, W) * cat_eye * 2, 0, 1)[..., np.newaxis]
        blurred  = blurred*(1-blend) + squeezed*blend

    return _p(np.clip(blurred, 0, 1))


def _distortion(img, barrel, pincushion, zoom):
    arr  = _f(img)
    H, W = arr.shape[:2]
    cy, cx = H / 2.0, W / 2.0
    y, x   = np.mgrid[0:H, 0:W]
    yn = (y - cy) / cy
    xn = (x - cx) / cx
    r2 = xn**2 + yn**2
    zoom = max(zoom, 0.01)   # guard against division by zero
    d  = (1.0 + barrel*0.4*r2 - pincushion*0.4*r2**2) / zoom
    sy = np.clip(yn*d*cy + cy, 0, H-1)
    sx = np.clip(xn*d*cx + cx, 0, W-1)
    result = np.stack([
        map_coordinates(arr[...,c], [sy, sx], order=1, mode='nearest')
        for c in range(3)
    ], axis=-1)
    return _p(result.astype(np.float32))


def _flare(img, intensity, pos_x, pos_y, streak_count, streak_length,
           ghost_count, flare_color):
    TINTS = {
        "warm":    (1.0, 0.85, 0.6),
        "cool":    (0.6, 0.8,  1.0),
        "neutral": (1.0, 1.0,  1.0),
        "rainbow": (1.0, 0.9,  0.7),
    }
    tint = TINTS.get(flare_color, TINTS["warm"])

    arr  = _f(img)
    H, W = arr.shape[:2]
    flare_layer = np.zeros((H, W, 3), dtype=np.float32)
    px, py = int(pos_x * W), int(pos_y * H)
    y_idx, x_idx = np.mgrid[0:H, 0:W]

    for i in range(streak_count):
        a  = np.pi * i / streak_count
        ca, sa = np.cos(a), np.sin(a)
        dx, dy = x_idx - px, y_idx - py
        proj   = dx*ca + dy*sa
        perp   = np.abs(-dx*sa + dy*ca)
        ml     = streak_length * max(H, W)
        streak = (np.exp(-0.5*(perp/1.2)**2) *
                  np.exp(-0.5*(proj/(ml*0.3))**2))
        for c in range(3):
            flare_layer[...,c] += streak * tint[c]

    d_src = np.sqrt((x_idx-px)**2 + (y_idx-py)**2)
    glow  = np.exp(-0.5*(d_src/(max(H,W)*0.04))**2)
    flare_layer += glow[...,np.newaxis] * np.array(tint)

    ax, ay = W/2 - px, H/2 - py
    for i in range(1, ghost_count+1):
        t  = i / (ghost_count+1)
        gx = int(px + ax*t*2)
        gy = int(py + ay*t*2)
        gr = max(H,W) * 0.02 * (1 + i*0.5)
        gd = np.sqrt((x_idx-gx)**2 + (y_idx-gy)**2)
        ring = np.exp(-0.5*((gd-gr)/(gr*0.3))**2) * 0.3 / (i**0.7)
        if flare_color == "rainbow":
            hues = [(1.0,0.3,0.8),(0.3,1.0,0.4),(0.3,0.5,1.0),
                    (1.0,0.8,0.2),(0.8,0.2,1.0)]
            gt = hues[i % len(hues)]
        else:
            gt = (0.7,0.5,1.0) if i%2==0 else (0.5,0.9,1.0)
        for c in range(3):
            flare_layer[...,c] += ring * gt[c]

    flare_layer = flare_layer / (flare_layer.max() + 1e-6) * intensity
    return _p(np.clip(arr + flare_layer, 0, 1))


def _halation(img, intensity, radius, threshold, warmth):
    arr = _f(img)
    lum = 0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]
    hi  = np.clip((lum - threshold) / (1.0 - threshold + 1e-6), 0, 1)
    glow = np.stack([
        gaussian_filter(arr[...,0]*hi*(1.0+warmth*0.6), sigma=radius),
        gaussian_filter(arr[...,1]*hi*(1.0+warmth*0.1), sigma=radius),
        gaussian_filter(arr[...,2]*hi*(1.0-warmth*0.5), sigma=radius),
    ], axis=-1)
    return _p(np.clip(arr + glow*intensity, 0, 1))


def _focus_falloff(img, mode, blur_radius, focus_pos, focus_width, feather):
    arr  = _f(img)
    H, W = arr.shape[:2]
    yy, xx = np.meshgrid(np.linspace(0,1,H), np.linspace(0,1,W), indexing='ij')
    hw = focus_width / 2.0
    f  = max(feather * focus_width, 0.01)

    if mode == "horizontal":
        dist = np.abs(yy - focus_pos)
    elif mode == "vertical":
        dist = np.abs(xx - focus_pos)
    elif mode == "radial":
        dist = np.sqrt((yy-focus_pos)**2 + (xx-0.5)**2) * 0.7
    else:
        dist = np.sqrt(((yy-focus_pos)/0.4)**2 + ((xx-0.5)/0.6)**2) * 0.35

    t = np.clip((dist-hw)/f, 0, 1)
    blur_map = (t*t*(3-2*t))[..., np.newaxis]
    blurred  = np.stack([gaussian_filter(arr[...,c], sigma=blur_radius)
                         for c in range(3)], axis=-1)
    return _p(arr*(1-blur_map) + blurred*blur_map)


def _spherical(img, intensity, radius, zone):
    arr     = _f(img)
    H, W    = arr.shape[:2]
    blurred = np.stack([gaussian_filter(arr[...,c], sigma=radius)
                        for c in range(3)], axis=-1)
    if zone == "global":
        bmap = np.ones((H,W), dtype=np.float32)
    elif zone == "edge":
        bmap = _radial(H, W)
    else:
        bmap = 1.0 - _radial(H, W)
    blend  = (bmap * intensity)[..., np.newaxis]
    result = arr*(1-blend) + blurred*blend
    return _p(np.clip((result-0.5)*(1.0+intensity*0.1)+0.5, 0, 1))


def _anamorphic(img, intensity, streak_color, streak_length, oval_bokeh, blue_bias):
    STREAK_TINTS = {
        "blue":  (0.5, 0.7, 1.0),
        "warm":  (1.0, 0.85, 0.5),
        "white": (1.0, 1.0,  1.0),
    }
    tint = STREAK_TINTS.get(streak_color, STREAK_TINTS["blue"])

    arr  = _f(img)
    H, W = arr.shape[:2]
    lum  = 0.299*arr[...,0] + 0.587*arr[...,1] + 0.114*arr[...,2]
    hi   = np.clip((lum - 0.7) / 0.3, 0, 1)

    flare_layer = np.zeros_like(arr)
    slen  = max(1, int(W * streak_length * 0.5))
    kx    = np.arange(-slen, slen+1)
    kern  = np.exp(-np.abs(kx) / (slen*0.25)).reshape(1, -1).astype(np.float32)
    kern /= kern.sum()
    for c in range(3):
        flare_layer[...,c] = convolve(arr[...,c]*hi, kern) * tint[c]

    result = np.clip(arr + flare_layer*intensity, 0, 1)

    if oval_bokeh > 0:
        hi2   = np.clip((lum-0.6)/0.4, 0, 1)[...,np.newaxis]
        vblur = np.stack([gaussian_filter(arr[...,c],
                          sigma=[6.0*oval_bokeh, 0.5]) for c in range(3)], axis=-1)
        result = result*(1-hi2*oval_bokeh*0.5) + vblur*hi2*oval_bokeh*0.5

    if blue_bias > 0:
        smask = np.clip(1.0 - lum/0.4, 0, 1)
        result[...,2] = np.clip(result[...,2] + smask*blue_bias*0.15, 0, 1)

    return _p(np.clip(result, 0, 1))
