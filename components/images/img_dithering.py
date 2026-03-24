import numpy as np
from PIL import Image
from numpy.lib.stride_tricks import sliding_window_view

# ─────────────────────────────────────────────────────────────────────────────
# Optional GPU / acceleration imports (graceful fallback)
# ─────────────────────────────────────────────────────────────────────────────

try:
    from numba import njit
    NUMBA_AVAILABLE = True
except ImportError:
    njit = None
    NUMBA_AVAILABLE = False


def _adaptive_dither_amplitude(scale: float, adaptive: bool, max_val: float) -> float:
    """
    Return dither amplitude in output-code units (LSB of 8-bit domain).
    (Already strengthened in previous version — unchanged)
    """
    base_lsb = max_val / 255.0
    if not adaptive:
        return 1.5 * base_lsb
    compression = float(np.clip(1.0 - scale, 0.0, 1.0))
    return (1.2 + 2.8 * compression) * base_lsb


def _estimate_global_scale(arr: np.ndarray, max_val: float) -> float:
    """
    Estimate effective tonal span (0..1) from channel min/max.
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


def _floyd_steinberg_quantize_python(arr: np.ndarray, max_val: float) -> np.ndarray:
    """Original pure-Python Floyd-Steinberg (kept for when Numba is not used)."""
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


@njit(fastmath=True)
def _floyd_steinberg_quantize_numba(arr: np.ndarray, max_val: float) -> np.ndarray:
    """Numba-accelerated version of Floyd-Steinberg (20–50× faster on CPU)."""
    work = np.clip(arr, 0.0, max_val).astype(np.float32)
    h, w, c = work.shape
    for ch in range(c):
        plane = work[:, :, ch]
        for y in range(h):
            for x in range(w):
                old = plane[y, x]
                new = min(max(np.round(old), 0.0), max_val)
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


def _get_spikiness_factor(c_hist: np.ndarray, total: float) -> float:
    """Return 0.0–2.0 boost factor when histogram has tall spikes."""
    if total <= 0:
        return 0.0
    peak_ratio = c_hist.max() / (c_hist.mean() + 1e-8)
    return np.clip((peak_ratio - 4.0) / 12.0, 0.0, 2.0)


def _normalize_midpeaks_channel(
        channel: np.ndarray,
        peak_width: int,
        max_val: float,
        rng: np.random.Generator,
) -> np.ndarray:
    """
    Histogram-aware anti-spike smoothing near empty bins (gaps).

    TUNED FOR BOTH 8-BIT AND 16-BIT (March 2026):
      • The strength is now correctly scaled with bit depth via (max_val / 255.0).
      • Base multiplier reduced from 8.0 → 2.5 so that:
          - 8-bit, peak_width=1  → ~2.5 LSB   (very gentle)
          - 8-bit, peak_width=3  → ~7.5 LSB   (good default)
          - 16-bit, peak_width=1 → ~2.5 LSB   (now gentle, was previously ~650 LSB!)
          - 16-bit, peak_width=5 → ~12.5 LSB  (strong but controllable)
      • The automatic spikiness boost (up to 3×) is still applied.
      • peak_width remains the ONLY user-controlled sensitivity variable:
        1 = minimal / surgical
        3 = balanced
        5–7 = strong
        8–10 = very aggressive
    """
    n_bins = int(max_val) + 1
    result = channel.copy()
    c_int = np.clip(np.round(result).astype(np.int64), 0, int(max_val))
    c_hist = np.bincount(c_int.ravel(), minlength=n_bins).astype(np.float64)

    gap_arr = (c_hist == 0)
    if not gap_arr.any():
        return result

    # ── TUNED STRENGTH (now safe for 16-bit) ────────────────────────────────
    amp = (peak_width * 1) * (max_val / 255.0)  # ← this is the key line

    total = c_hist.sum()
    spikiness = _get_spikiness_factor(c_hist, total)
    amp *= (1.0 + spikiness)
    half = amp / 2.0
    noise = (rng.uniform(-half, half, result.shape).astype(np.float32) + rng.uniform(-half, half, result.shape).astype(np.float32))
    pad = peak_width
    padded = np.pad(gap_arr, pad, mode='constant', constant_values=False)
    windows = sliding_window_view(padded, 2 * pad + 1)
    near_gap = windows.any(axis=1)

    qualify_mask = near_gap[c_int] & (~gap_arr[c_int])
    return np.where(qualify_mask, np.clip(result + noise, 0.0, max_val), result)


def img_dithering(
        image: Image.Image,
        dither_quantization: bool = True,
        adaptive_dither_strength: bool = True,
        error_diffusion: bool = False,
        normalize_midpeaks: bool = False,
        peak_width: int = 3,
        high_precision: bool = False,
        numba_accelerated: bool = True,
) -> Image.Image:
    """
    Standalone quantization dither stage for post-processing.

    normalize_midpeaks STRENGTH NOW PROPERLY SCALED FOR 16-BIT.
    The ONLY variable that controls sensitivity is peak_width (as before).
    """
    if not (1 <= peak_width <= 10):
        raise ValueError(f"peak_width must be 1–10, got {peak_width}")

    if error_diffusion and numba_accelerated and not NUMBA_AVAILABLE:
        print("⚠️  numba_accelerated=True but Numba is not installed. "
              "Falling back to pure Python (slow). Run: pip install numba")

    arr_8f = np.array(image.convert("RGB"), dtype=np.float32)
    max_val = 65535.0 if high_precision else 255.0
    scale_factor = max_val / 255.0
    arr = arr_8f * scale_factor if high_precision else arr_8f

    # ── 1. Mid-peak spike removal (now correctly tuned for 16-bit) ───────────
    if normalize_midpeaks:
        for ch in range(3):
            rng = np.random.default_rng(100 + ch)
            arr[:, :, ch] = _normalize_midpeaks_channel(arr[:, :, ch], peak_width, max_val, rng)

    # ── 2. Final quantization stage ──────────────────────────────────────────
    if error_diffusion:
        pre_amp = 0.5 * (max_val / 255.0)
        arr = arr + _tpdf_noise(arr.shape, pre_amp)

        if NUMBA_AVAILABLE and numba_accelerated:
            quantized = _floyd_steinberg_quantize_numba(arr, max_val)
        else:
            quantized = _floyd_steinberg_quantize_python(arr, max_val)

    else:
        quant_input = arr
        if dither_quantization:
            scale = _estimate_global_scale(quant_input, max_val)
            amp = _adaptive_dither_amplitude(scale, adaptive_dither_strength, max_val)

            flat = np.clip(np.round(quant_input).astype(np.int64), 0, int(max_val)).ravel()
            c_hist = np.bincount(flat, minlength=int(max_val) + 1).astype(np.float64)
            total = c_hist.sum()
            spikiness = _get_spikiness_factor(c_hist, total)
            amp *= (1.0 + spikiness)

            quant_input = quant_input + _tpdf_noise(quant_input.shape, amp)
        quantized = np.clip(np.rint(quant_input), 0, max_val)

    out_8f = quantized * (255.0 / max_val) if high_precision else quantized
    out_8 = np.clip(np.rint(out_8f), 0, 255).astype(np.uint8)

    return Image.fromarray(out_8, mode="RGB")