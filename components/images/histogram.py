import numpy as np
from PIL import Image, ImageFilter
import glob
import hashlib
import json
import folder_paths
import os

_HIST_CH_DEFS = {
    "RGB":   [(0, (1.0, 0.22, 0.22)), (1, (0.22, 1.0, 0.22)), (2, (0.22, 0.44, 1.0))],
    "RED":   [(0, (1.0, 0.22, 0.22))],
    "GREEN": [(1, (0.22, 1.0, 0.22))],
    "BLUE":  [(2, (0.22, 0.44, 1.0))],
}

# All valid style names
VALID_STYLES = {
    "gradient",     # original — filled area with top-to-bottom brightness fade
    "bars",         # original — flat filled area
    "lines",        # original — thin outline only (2px)
    "glow",         # original — bars + gaussian bloom
    "waveform",     # center-line oscilloscope, mirrored above/below midpoint
    "heatmap",      # single-channel density map using perceptual colour ramp
    "stacked",      # R/G/B channels stacked (not overlapping)
    "dots",         # vertical dot columns, density proportional to count
    "step",         # raw unsmoothed step function — shows true comb pattern
    "luma",         # gradient + luminosity curve overlaid in white
    "log",          # gradient with log-scale Y axis
    "parade",       # R | G | B panels side by side (ignores channel arg)
    "percentile",   # gradient + vertical lines at 10/25/50/75/90 percentiles
    "inverse",      # light background variant of gradient
}


def _get_raw(arr: np.ndarray, ch_idx: int, precision: bool) -> np.ndarray:
    """Return 256-bin histogram for one channel."""
    if precision:
        arr_16 = arr[:, :, ch_idx] * (65535.0 / 255.0)
        raw_16, _ = np.histogram(arr_16, bins=65536, range=(0, 65536))
        return raw_16.reshape(256, 256).sum(axis=1).astype(np.float32)
    else:
        raw, _ = np.histogram(arr[:, :, ch_idx], bins=256, range=(0, 256))
        return raw.astype(np.float32)


def _make_smooth(sigma: float):
    """Return a closure that gaussian-smooths a 256-element array."""
    size = max(int(sigma * 4) | 1, 3)
    kx   = np.arange(size) - size // 2
    k    = np.exp(-0.5 * (kx / sigma) ** 2); k /= k.sum()
    def _smooth(h: np.ndarray) -> np.ndarray:
        return np.convolve(h.astype(np.float32), k, mode='same')
    return _smooth


def _normalise(h: np.ndarray, smooth_fn, sqrt: bool) -> np.ndarray:
    sm = smooth_fn(h)
    if sqrt:
        sm = np.sqrt(np.maximum(sm, 0))
    return sm / (sm.max() or 1.0)


def _log_normalise(h: np.ndarray, smooth_fn) -> np.ndarray:
    sm = smooth_fn(h)
    sm = np.log1p(np.maximum(sm, 0))
    return sm / (sm.max() or 1.0)


def _dark_canvas(h: int, w: int) -> np.ndarray:
    canvas = np.full((h, w, 3), 18.0 / 255.0, dtype=np.float32)
    for frac in (0.25, 0.5, 0.75):
        canvas[int((1.0 - frac) * (h - 1)), :] = 0.32
        canvas[:, int(frac * (w - 1)), :] = 0.32
    return canvas


def _light_canvas(h: int, w: int) -> np.ndarray:
    canvas = np.full((h, w, 3), 0.92, dtype=np.float32)
    for frac in (0.25, 0.5, 0.75):
        canvas[int((1.0 - frac) * (h - 1)), :] = 0.72
        canvas[:, int(frac * (w - 1)), :] = 0.72
    return canvas


def _cols_heights(norm256: np.ndarray, hist_w: int, hist_h: int):
    """Interpolate normalised 256-bin curve to hist_w display columns."""
    x_idx   = np.linspace(0, 255, hist_w)
    cols    = np.interp(x_idx, np.arange(256), norm256)
    heights = (cols * (hist_h - 1)).astype(int)
    return cols, heights


def _draw_gradient(canvas, heights, color, hist_h, hist_w):
    row_idx  = np.arange(hist_h).reshape(-1, 1)
    fill_mask = row_idx >= (hist_h - heights)
    safe_h    = np.maximum(heights, 1).astype(np.float32)
    dist_b    = (hist_h - 1 - row_idx).astype(np.float32)
    grad      = np.clip(0.28 + 0.72 * (dist_b / safe_h), 0.0, 1.0)
    for ci, cv in enumerate(color):
        canvas[:, :, ci] = np.where(
            fill_mask, np.maximum(canvas[:, :, ci], grad * cv), canvas[:, :, ci])


def _draw_bars(canvas, heights, color, hist_h, hist_w, alpha=1.0):
    row_idx  = np.arange(hist_h).reshape(-1, 1)
    fill_mask = row_idx >= (hist_h - heights)
    for ci, cv in enumerate(color):
        canvas[:, :, ci] = np.where(
            fill_mask, np.maximum(canvas[:, :, ci], cv * alpha), canvas[:, :, ci])


def _draw_lines(canvas, heights, color, hist_h, hist_w):
    for dy in range(2):
        rs = np.clip(hist_h - heights - dy, 0, hist_h - 1)
        xs = np.arange(hist_w)[heights > 0]
        for ci, cv in enumerate(color):
            canvas[rs[xs], xs, ci] = np.maximum(canvas[rs[xs], xs, ci], cv)

def rasterix_histogram_render(
    pil_img:   Image.Image,
    channel:   str  = "RGB",
    style:     str  = "bars",
    precision: bool = False,
) -> Image.Image:

    if style not in VALID_STYLES:
        raise ValueError(f"style must be one of {sorted(VALID_STYLES)}, got '{style}'")

    arr      = np.array(pil_img.convert("RGB"), dtype=np.float32)
    hist_h   = 192
    hist_w   = 512
    sqrt_norm = False

    sigma = 0.75 if style in ("bars", "step", "dots") else 1.0
    smooth   = _make_smooth(sigma)
    channels = _HIST_CH_DEFS.get(channel, _HIST_CH_DEFS["RGB"])

    if style == "parade":
        panel_w  = hist_w // 3   # 341 px each; total = 1023 px
        parade_w = panel_w * 3
        canvas   = np.full((hist_h, parade_w, 3), 18.0 / 255.0, dtype=np.float32)
        for p in range(3):
            ox = p * panel_w
            for frac in (0.25, 0.5, 0.75):
                canvas[int((1.0-frac)*(hist_h-1)), ox:ox+panel_w] = 0.32
                canvas[:, ox + int(frac*(panel_w-1)), :] = 0.32
            if p > 0:
                canvas[:, ox, :] = 0.45
        for p, (ch_idx, color) in enumerate(_HIST_CH_DEFS["RGB"]):
            ox   = p * panel_w
            raw  = _get_raw(arr, ch_idx, precision)
            norm = _normalise(raw, smooth, sqrt_norm)
            _, heights = _cols_heights(norm, panel_w, hist_h)
            sub = canvas[:, ox:ox+panel_w, :]
            _draw_gradient(sub, heights, color, hist_h, panel_w)
            canvas[:, ox:ox+panel_w, :] = sub
        result = Image.fromarray(
            np.clip(canvas * 255, 0, 255).astype(np.uint8), mode="RGB")
        return result

    if style == "inverse":
        canvas = _light_canvas(hist_h, hist_w)
        inv_colors = {0: (0.75, 0.10, 0.10), 1: (0.10, 0.65, 0.10), 2: (0.10, 0.25, 0.85)}
        draw_channels = [(idx, inv_colors.get(idx, col)) for idx, col in channels]
    else:
        canvas = _dark_canvas(hist_h, hist_w)
        draw_channels = channels

    raws  = {ch_idx: _get_raw(arr, ch_idx, precision) for ch_idx, _ in channels}
    luma_raw = None
    if style in ("luma", "heatmap"):
        luma_arr = (0.299 * arr[:,:,0] + 0.587 * arr[:,:,1] + 0.114 * arr[:,:,2])
        if precision:
            l16 = luma_arr * (65535.0/255.0)
            r16, _ = np.histogram(l16, bins=65536, range=(0,65536))
            luma_raw = r16.reshape(256,256).sum(axis=1).astype(np.float32)
        else:
            luma_raw, _ = np.histogram(luma_arr, bins=256, range=(0,256))
            luma_raw = luma_raw.astype(np.float32)

    if style == "heatmap":
        norm = _normalise(luma_raw, smooth, sqrt_norm)
        x_idx   = np.linspace(0, 255, hist_w)
        cols    = np.interp(x_idx, np.arange(256), norm)
        ramp_t  = np.array([0.0,  0.33,  0.66, 1.0])
        ramp_r  = np.array([0.0,  0.05,  0.0,  1.0])
        ramp_g  = np.array([0.0,  0.05,  0.85, 1.0])
        ramp_b  = np.array([0.0,  0.55,  0.85, 1.0])
        cr = np.interp(cols, ramp_t, ramp_r)
        cg = np.interp(cols, ramp_t, ramp_g)
        cb = np.interp(cols, ramp_t, ramp_b)
        row_idx  = np.arange(hist_h).reshape(-1, 1)
        heights  = (cols * (hist_h - 1)).astype(int)
        fill_mask = row_idx >= (hist_h - heights)
        safe_h   = np.maximum(heights, 1).astype(np.float32)
        dist_b   = (hist_h - 1 - row_idx).astype(np.float32)
        grad     = np.clip(0.15 + 0.85 * (dist_b / safe_h), 0.0, 1.0)
        for ci, cramp in enumerate([cr, cg, cb]):
            canvas[:, :, ci] = np.where(fill_mask,
                np.maximum(canvas[:, :, ci], grad * cramp), canvas[:, :, ci])
        result = Image.fromarray(
            np.clip(canvas * 255, 0, 255).astype(np.uint8), mode="RGB")
        return result

    if style == "stacked":
        x_idx    = np.linspace(0, 255, hist_w)
        row_idx  = np.arange(hist_h).reshape(-1, 1)
        norms = []
        for ch_idx, _ in _HIST_CH_DEFS["RGB"]:
            raw = _get_raw(arr, ch_idx, precision)
            norms.append(np.interp(x_idx, np.arange(256), _normalise(raw, smooth, sqrt_norm)))
        norms = np.array(norms)   # (3, hist_w)
        total = norms.sum(axis=0) + 1e-6
        fracs = norms / total     # (3, hist_w), each col sums to 1
        cum_h = np.zeros(hist_w, dtype=np.float32)
        for layer, (ch_idx, color) in enumerate(_HIST_CH_DEFS["RGB"]):
            layer_h = (fracs[layer] * (hist_h - 1) * norms.max(axis=0) / norms.max()).astype(int)
            floor_h  = cum_h.astype(int)
            top_h    = (cum_h + layer_h).astype(int)
            for x in range(hist_w):
                if layer_h[x] > 0:
                    y_lo = hist_h - 1 - top_h[x]
                    y_hi = hist_h - 1 - floor_h[x]
                    y_lo = np.clip(y_lo, 0, hist_h-1)
                    y_hi = np.clip(y_hi, 0, hist_h-1)
                    if y_lo <= y_hi:
                        for ci, cv in enumerate(color):
                            canvas[y_lo:y_hi+1, x, ci] = np.maximum(
                                canvas[y_lo:y_hi+1, x, ci], cv * 0.85)
            cum_h += layer_h
        result = Image.fromarray(
            np.clip(canvas * 255, 0, 255).astype(np.uint8), mode="RGB")
        return result

    if style == "step":
        row_idx = np.arange(hist_h).reshape(-1, 1)
        column_bin = np.minimum((np.arange(hist_w) * 256) // hist_w, 255).astype(np.int32)
        for ch_idx, color in draw_channels:
            raw = _get_raw(arr, ch_idx, precision)
            if sqrt_norm:
                norm256 = np.sqrt(np.maximum(raw, 0)); norm256 /= (norm256.max() or 1.0)
            else:
                norm256 = raw / (raw.max() or 1.0)
            heights = (norm256[column_bin] * (hist_h - 1)).astype(int)
            fill_mask = row_idx >= (hist_h - heights)
            for ci, cv in enumerate(color):
                canvas[:, :, ci] = np.where(
                    fill_mask, np.maximum(canvas[:, :, ci], cv * 0.9), canvas[:, :, ci])
            y = np.clip(hist_h - 1 - heights, 0, hist_h - 1)
            x = np.arange(hist_w)
            canvas[y, x, :] = np.maximum(canvas[y, x, :], 0.95)
        result = Image.fromarray(
            np.clip(canvas * 255, 0, 255).astype(np.uint8), mode="RGB")
        return result

    if style == "waveform":
        x_idx   = np.linspace(0, 255, hist_w)
        mid     = hist_h // 2
        for ch_idx, color in draw_channels:
            raw  = _get_raw(arr, ch_idx, precision)
            norm = _normalise(raw, smooth, sqrt_norm)
            cols = np.interp(x_idx, np.arange(256), norm)
            amp  = (cols * (mid - 2)).astype(int)   # half-amplitude
            for x in range(hist_w):
                if amp[x] == 0: continue
                y_lo = np.clip(mid - amp[x], 0, hist_h-1)
                y_hi = np.clip(mid + amp[x], 0, hist_h-1)
                for y in range(y_lo, y_hi+1):
                    dist = abs(y - mid) / max(amp[x], 1)
                    brightness = max(0.25, 1.0 - dist * 0.7)
                    for ci, cv in enumerate(color):
                        canvas[y, x, ci] = max(canvas[y, x, ci], cv * brightness)
            canvas[mid, :, :] = np.maximum(canvas[mid, :, :], 0.28)
        result = Image.fromarray(
            np.clip(canvas * 255, 0, 255).astype(np.uint8), mode="RGB")
        return result

    if style == "dots":
        x_idx  = np.linspace(0, 255, hist_w)
        n_dots = 32   # max dots per column
        for ch_idx, color in draw_channels:
            raw  = _get_raw(arr, ch_idx, precision)
            norm = _normalise(raw, smooth, sqrt_norm)
            cols = np.interp(x_idx, np.arange(256), norm)
            for x in range(hist_w):
                n = max(1, int(cols[x] * n_dots))
                positions = np.linspace(hist_h - 2, int((1.0 - cols[x]) * (hist_h - 1)), n)
                for pos in positions:
                    y = int(np.clip(pos, 0, hist_h - 1))
                    brightness = 0.5 + 0.5 * (1.0 - pos / hist_h)
                    for ci, cv in enumerate(color):
                        canvas[y, x, ci] = max(canvas[y, x, ci], cv * brightness)
        result = Image.fromarray(
            np.clip(canvas * 255, 0, 255).astype(np.uint8), mode="RGB")
        return result

    if style == "log":
        x_idx = np.linspace(0, 255, hist_w)
        for ch_idx, color in draw_channels:
            raw  = _get_raw(arr, ch_idx, precision)
            norm = _log_normalise(raw, smooth)
            _, heights = _cols_heights(norm, hist_w, hist_h)
            _draw_gradient(canvas, heights, color, hist_h, hist_w)
        result = Image.fromarray(
            np.clip(canvas * 255, 0, 255).astype(np.uint8), mode="RGB")
        return result

    if style == "luma":
        x_idx = np.linspace(0, 255, hist_w)
        for ch_idx, color in draw_channels:
            raw  = _get_raw(arr, ch_idx, precision)
            norm = _normalise(raw, smooth, sqrt_norm)
            _, heights = _cols_heights(norm, hist_w, hist_h)
            dimmed = tuple(v * 0.55 for v in color)
            _draw_gradient(canvas, heights, dimmed, hist_h, hist_w)
        norm_luma = _normalise(luma_raw, smooth, sqrt_norm)
        cols_luma = np.interp(x_idx, np.arange(256), norm_luma)
        heights_luma = (cols_luma * (hist_h - 1)).astype(int)
        white = (1.0, 1.0, 1.0)
        _draw_lines(canvas, heights_luma, white, hist_h, hist_w)
        result = Image.fromarray(
            np.clip(canvas * 255, 0, 255).astype(np.uint8), mode="RGB")
        return result

    if style == "percentile":
        x_idx = np.linspace(0, 255, hist_w)
        for ch_idx, color in draw_channels:
            raw  = _get_raw(arr, ch_idx, precision)
            norm = _normalise(raw, smooth, sqrt_norm)
            _, heights = _cols_heights(norm, hist_w, hist_h)
            _draw_gradient(canvas, heights, color, hist_h, hist_w)
        if len(draw_channels) == 3:
            lum = 0.299*arr[:,:,0] + 0.587*arr[:,:,1] + 0.114*arr[:,:,2]
            flat = lum.ravel()
        else:
            flat = arr[:, :, draw_channels[0][0]].ravel()
        pcts  = [10, 25, 50, 75, 90]
        pvals = np.percentile(flat, pcts)
        pct_colors = [
            (0.60, 0.60, 0.60),  # 10th — grey
            (0.85, 0.85, 0.30),  # 25th — yellow
            (1.00, 1.00, 1.00),  # 50th — white (median)
            (0.85, 0.85, 0.30),  # 75th — yellow
            (0.60, 0.60, 0.60),  # 90th — grey
        ]
        for pval, pcol in zip(pvals, pct_colors):
            x_pos = int(np.interp(pval, [0, 255], [0, hist_w - 1]))
            x_pos = np.clip(x_pos, 0, hist_w - 1)
            for ci, cv in enumerate(pcol):
                canvas[:, x_pos, ci] = cv
        result = Image.fromarray(
            np.clip(canvas * 255, 0, 255).astype(np.uint8), mode="RGB")
        return result

    if style == "inverse":
        x_idx = np.linspace(0, 255, hist_w)
        for ch_idx, color in draw_channels:
            raw  = _get_raw(arr, ch_idx, precision)
            norm = _normalise(raw, smooth, sqrt_norm)
            cols = np.interp(x_idx, np.arange(256), norm)
            heights = (cols * (hist_h - 1)).astype(int)
            row_idx  = np.arange(hist_h).reshape(-1, 1)
            fill_mask = row_idx >= (hist_h - heights)
            safe_h    = np.maximum(heights, 1).astype(np.float32)
            dist_b    = (hist_h - 1 - row_idx).astype(np.float32)
            grad = np.clip(0.15 + 0.85 * (1.0 - dist_b / safe_h), 0.0, 1.0)
            for ci, cv in enumerate(color):
                canvas[:, :, ci] = np.where(
                    fill_mask,
                    np.minimum(canvas[:, :, ci], 1.0 - grad * cv * 0.7),
                    canvas[:, :, ci])
        result = Image.fromarray(
            np.clip(canvas * 255, 0, 255).astype(np.uint8), mode="RGB")
        return result

    for ch_idx, color in draw_channels:
        raw  = _get_raw(arr, ch_idx, precision)
        norm = _normalise(raw, smooth, sqrt_norm)
        _, heights = _cols_heights(norm, hist_w, hist_h)

        if style in ("gradient", "inverse"):
            _draw_gradient(canvas, heights, color, hist_h, hist_w)
        elif style == "lines":
            _draw_lines(canvas, heights, color, hist_h, hist_w)
        else:   # bars, glow
            _draw_bars(canvas, heights, color, hist_h, hist_w)

    result = Image.fromarray(
        np.clip(canvas * 255, 0, 255).astype(np.uint8), mode="RGB")

    if style == "glow":
        bloom  = result.filter(ImageFilter.GaussianBlur(radius=5))
        result = Image.fromarray(
            np.clip(
                np.array(result, dtype=np.float32) +
                np.array(bloom,  dtype=np.float32) * 0.55,
                0, 255).astype(np.uint8), mode="RGB")

    return result

def _safe_node_id(node_id):
    safe = str(node_id or "global")
    return "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in safe)

def rasterix_hist_cache_paths(node_id=None):
    hist_dir = folder_paths.get_temp_directory()
    os.makedirs(hist_dir, exist_ok=True)
    safe_id = _safe_node_id(node_id)
    return hist_dir, os.path.join(hist_dir, f"rasterix_hist_cache_{safe_id}_input.png"), os.path.join(hist_dir, f"rasterix_hist_cache_{safe_id}_output.png"),

def rasterix_hist_meta_path(node_id=None):
    hist_dir, _, _ = rasterix_hist_cache_paths(node_id=node_id)
    return os.path.join(hist_dir, f"rasterix_hist_cache_meta_{_safe_node_id(node_id)}.json")

def rasterix_hist_signature(pil_img):
    return hashlib.sha1(pil_img.tobytes()).hexdigest()

def rasterix_hist_cache_store(pil_input, pil_output, precision, node_id=None):
    hist_dir, in_path, out_path = rasterix_hist_cache_paths(node_id=node_id)
    meta_path = rasterix_hist_meta_path(node_id=node_id)
    safe_id = _safe_node_id(node_id)
    current_key = f"{rasterix_hist_signature(pil_input)}::{rasterix_hist_signature(pil_output)}::{'16' if precision else '8'}"

    previous_key = None
    if os.path.isfile(meta_path):
        try:
            with open(meta_path, 'r', encoding='utf-8') as f:
                previous_key = (json.load(f) or {}).get("cache_key")
        except Exception:
            previous_key = None

    if current_key != previous_key:
        for old_hist in glob.glob(os.path.join(hist_dir, f"History_{safe_id}_*.png")):
            try:
                os.remove(old_hist)
            except OSError:
                pass

    pil_input.save(in_path, compress_level=1)
    pil_output.save(out_path, compress_level=1)
    with open(meta_path, 'w', encoding='utf-8') as f:
        json.dump({"cache_key": current_key}, f)

def rasterix_hist_render_path(node_id, histogram_source, histogram_channel, histogram_style):
    hist_dir, _, _ = rasterix_hist_cache_paths(node_id=node_id)
    source_prefix = "input" if histogram_source else "output"
    safe_id = _safe_node_id(node_id)
    history_filename = f"History_{safe_id}_{source_prefix}_{histogram_channel.lower()}_{histogram_style}.png"
    return os.path.join(hist_dir, history_filename)

def rasterix_hist_render_selected(pil_input, pil_output, precision, histogram_source, histogram_channel, histogram_style, node_id=None):
    target_file = rasterix_hist_render_path(node_id, histogram_source, histogram_channel, histogram_style)
    if os.path.isfile(target_file):
        return Image.open(target_file).convert("RGB")
    source_image = pil_input if histogram_source else pil_output
    rendered = rasterix_histogram_render(source_image, histogram_channel, histogram_style, precision)
    rendered.save(target_file, compress_level=1)
    return rendered