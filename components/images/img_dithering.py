import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view

def _adaptive_dither_amplitude(scale: float, adaptive: bool, max_val: float) -> float:
    """
    Return dither amplitude in output-code units (LSB of 8-bit domain).
    scale=1.0 means wide tonal span, lower values mean tighter span.
    """
    base_lsb = max_val / 255.0
    if not adaptive:
        return 1.0 * base_lsb
    compression = float(np.clip(1.0 - scale, 0.0, 1.0))
    return (0.75 + 1.25 * compression) * base_lsb


def _estimate_global_scale(arr: np.ndarray, max_val: float) -> float:
    """
    Estimate effective tonal span (0..1) from channel min/max.
    Used for adaptive dither strength when the operation is applied
    as a standalone post-process.
    """
    mins = arr.reshape(-1, 3).min(axis=0)
    maxs = arr.reshape(-1, 3).max(axis=0)
    spans = np.clip((maxs - mins) / max_val, 0.0, 1.0)
    return float(np.mean(spans))


def _tpdf_noise(shape: tuple[int, int, int], amplitude: float) -> np.ndarray:
    """Triangular PDF noise in [-amplitude, +amplitude], float32."""
    h, w, c = shape
    rng = np.random.default_rng()
    u1 = rng.random((h, w, c), dtype=np.float32)
    u2 = rng.random((h, w, c), dtype=np.float32)
    return (u1 - u2) * amplitude


def _floyd_steinberg_quantize(arr: np.ndarray, max_val: float) -> np.ndarray:
    """Floyd-Steinberg error-diffusion quantization in current precision domain."""
    work = np.clip(arr, 0.0, max_val).astype(np.float32, copy=True)
    h, w, c = work.shape
    for ch in range(c):
        plane = work[:, :, ch]
        for y in range(h):
            for x in range(w):
                old = plane[y, x]
                new = np.clip(np.rint(old), 0.0, max_val)
                err = old - new
                plane[y, x] = new
                if x + 1 < w:
                    plane[y, x + 1] += err * (7.0 / 16.0)
                if y + 1 < h:
                    if x > 0:
                        plane[y + 1, x - 1] += err * (3.0 / 16.0)
                    plane[y + 1, x] += err * (5.0 / 16.0)
                    if x + 1 < w:
                        plane[y + 1, x + 1] += err * (1.0 / 16.0)
    return np.clip(work, 0.0, max_val)


def _normalize_midpeaks_channel(
    channel: np.ndarray,
    peak_width: int,
    max_val: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """
    Histogram-aware anti-spike smoothing near empty bins (gaps),
    adapted from levels_auto for standalone post-process use.
    """
    n_bins = int(max_val) + 1
    result = channel.copy()
    c_int = np.clip(np.round(result).astype(np.int64), 0, int(max_val))
    c_hist = np.bincount(c_int.ravel(), minlength=n_bins).astype(np.float64)

    gap_arr = (c_hist == 0)
    if not gap_arr.any():
        return result

    pad = peak_width
    padded = np.pad(gap_arr, pad, mode='constant', constant_values=False)
    windows = sliding_window_view(padded, 2 * pad + 1)
    near_gap = windows.any(axis=1)

    # Use a bit-depth-scaled TPDF noise amplitude similar to levels_auto.
    amp = (peak_width / 2.0) * (max_val / 255.0)
    half = amp / 2.0
    noise = (
        rng.uniform(-half, half, result.shape).astype(np.float32) +
        rng.uniform(-half, half, result.shape).astype(np.float32)
    )

    qualify_mask = near_gap[c_int] & (~gap_arr[c_int])
    return np.where(qualify_mask, np.clip(result + noise, 0.0, max_val), result)


def _normalize_gaps_legacy(
    stretched: np.ndarray,
    scale:     float,
    rng_gap:   np.random.Generator,
    max_val:   float = 255.0,
) -> np.ndarray:
    """
    Anti-comb filter — TPDF gap dithering (moved from img_levels_auto).

    Fills quantization gaps from non-integer stretch scale factors.
    """
    print('------------ 5 -------------------')
    amplitude = max(1.0, (scale / 1.275) ** 2.2) * (max_val / 255.0)
    half      = amplitude / 2.0
    noise     = (rng_gap.uniform(-half, half, stretched.shape).astype(np.float32) + rng_gap.uniform(-half, half, stretched.shape).astype(np.float32))

    return np.clip(stretched + noise, 0.0, max_val)


def img_dithering(
    image: Image.Image,
    normalize_gaps_legacy: bool = False,
    stretched_gaps_spike: list = [],
    scale_spike: list = [],
    rng_gap_spike: list = [],
    dither_quantization: bool = True,
    adaptive_dither_strength: bool = True,
    error_diffusion: bool = False,
    normalize_midpeaks: bool = False,
    peak_width: int = 3,
    high_precision: bool = False,
) -> Image.Image:
    if not (1 <= peak_width <= 10):
        raise ValueError(f"peak_width must be 1–10, got {peak_width}")

    print('------------ 2 -------------------')
    arr_8f = np.array(image.convert("RGB"), dtype=np.float32)
    max_val = 65535.0 if high_precision else 255.0
    scale_factor = max_val / 255.0
    arr = arr_8f * scale_factor if high_precision else arr_8f

    # 1) Legacy anti-comb (TPDF gap dither from auto-levels stretch)
    #    - If lists were provided by img_levels_auto → use exact per-channel scale + RNG
    #    - Otherwise (auto-levels was off) → fallback to a useful non-specific scale
    #      estimated from the current image tonal span (exactly as you requested).
    if normalize_gaps_legacy:
        print('------------ 3 -------------------')
        if len(stretched_gaps_spike) > 0 and len(scale_spike) > 0 and len(rng_gap_spike) > 0:
            print('------------ 4a -------------------')
            # Auto-levels was used → use its exact stretch parameters
            for ch in range(3):
                scale = scale_spike[ch]
                rng_gap = rng_gap_spike[ch]
                # Apply to the current image (post-gamma if auto_gamma was enabled).
                # This is the cleanest logical placement now that the function lives here.
                arr[:, :, ch] = _normalize_gaps_legacy(arr[:, :, ch], scale, rng_gap, max_val)
        else:
            print('------------ 4b -------------------')
            # Auto-levels was OFF → run with useful non-specific parameters
            local_scale = _estimate_global_scale(arr, max_val)
            for ch in range(3):
                rng_gap = np.random.default_rng(ch)          # same seeding style as levels_auto
                arr[:, :, ch] = _normalize_gaps_legacy(arr[:, :, ch], local_scale, rng_gap, max_val)

    # 2) Mid-peak smoothing
    if normalize_midpeaks:
        for ch in range(3):
            rng = np.random.default_rng(100 + ch)
            arr[:, :, ch] = _normalize_midpeaks_channel(arr[:, :, ch], peak_width, max_val, rng)

    # 3) Quantization path
    if error_diffusion:
        quantized = _floyd_steinberg_quantize(arr, max_val)
    else:
        quant_input = arr
        if dither_quantization:
            local_scale = _estimate_global_scale(quant_input, max_val)
            amp = _adaptive_dither_amplitude(local_scale, adaptive_dither_strength, max_val)
            quant_input = quant_input + _tpdf_noise(quant_input.shape, amp)
        quantized = np.clip(np.rint(quant_input), 0, max_val)

    out_8f = quantized * (255.0 / max_val) if high_precision else quantized
    out_8 = np.clip(np.rint(out_8f), 0, 255).astype(np.uint8)

    return Image.fromarray(out_8, mode="RGB")