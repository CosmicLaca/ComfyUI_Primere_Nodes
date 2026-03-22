import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

EDGE_SPREAD_RATIO = 8.0 / 255.0   # edge spread as fraction of max value
                                    # 8-bit:  8 bins,  16-bit: 2056 bins
GAMMA_MIN         = 0.25
GAMMA_MAX         = 4.0


# ─────────────────────────────────────────────────────────────────────────────
# Step functions — callable independently
# ─────────────────────────────────────────────────────────────────────────────

def levels_detect_points(
    channel:    np.ndarray,
    threshold:  float,
    max_val:    float = 255.0,
) -> tuple:
    """
    Detect black and white points from a single channel histogram.

    Args:
        channel   : 2D float32 array, values 0–max_val
        threshold : 0.0–100.0, percent of pixels to clip at each end
        max_val   : 255.0 for 8-bit, 65535.0 for 16-bit

    Returns:
        (black_point, white_point, scale)
        scale = max_val / (white_point - black_point)
    """
    n_bins     = int(max_val) + 1
    hist, _    = np.histogram(channel, bins=n_bins, range=(0, max_val + 1))
    cumulative = np.cumsum(hist)
    total      = int(cumulative[-1])
    cutoff     = total * (threshold / 100.0)

    black_point = 0
    for i in range(n_bins):
        if cumulative[i] >= cutoff:
            black_point = i
            break

    white_point = int(max_val)
    for i in range(n_bins - 1, -1, -1):
        if (total - cumulative[i]) >= cutoff:
            white_point = i
            break

    if white_point <= black_point:
        white_point = min(black_point + 1, int(max_val))

    scale = max_val / (white_point - black_point)
    return black_point, white_point, scale


def levels_stretch(
    channel:     np.ndarray,
    black_point: int,
    white_point: int,
    max_val:     float = 255.0,
) -> np.ndarray:
    """
    Linear stretch of channel values to [0 … max_val].

    Args:
        channel     : 2D float32 array
        black_point : input value that maps to 0
        white_point : input value that maps to max_val
        max_val     : 255.0 for 8-bit, 65535.0 for 16-bit

    Returns:
        Stretched float32 array clipped to [0, max_val]
    """
    scale     = max_val / (white_point - black_point)
    stretched = (channel - black_point) * scale
    return np.clip(stretched, 0.0, max_val)


def levels_edge_spread(
    channel:     np.ndarray,
    stretched:   np.ndarray,
    black_point: int,
    white_point: int,
    max_val:     float = 255.0,
) -> np.ndarray:
    """
    Rank-based edge spread — always applied, not gated by any boolean.

    Spreads clipped pixels uniformly across [0 … edge_spread] and
    [max_val-edge_spread … max_val]. Edge spread width scales proportionally
    with max_val so the same fraction of the range is used at any bit depth.

    Args:
        channel     : original 2D float32 channel before stretch
        stretched   : 2D float32 array after stretch
        black_point : black point used in stretch
        white_point : white point used in stretch
        max_val     : 255.0 for 8-bit, 65535.0 for 16-bit

    Returns:
        Float32 array with edge pixels redistributed
    """
    edge_spread = EDGE_SPREAD_RATIO * max_val   # ~8 at 8-bit, ~2056 at 16-bit
    result      = stretched.copy()

    if black_point > 0:
        below_mask = channel < black_point
        if below_mask.any():
            flat_idx   = np.where(below_mask.ravel())[0]
            n          = len(flat_idx)
            rank_order = np.argsort(np.argsort(channel.ravel()[flat_idx]))
            flat_out   = result.ravel().copy()
            flat_out[flat_idx] = edge_spread * rank_order / max(n - 1, 1)
            result = flat_out.reshape(result.shape)

    if white_point < int(max_val):
        above_mask = channel > white_point
        if above_mask.any():
            flat_idx   = np.where(above_mask.ravel())[0]
            n          = len(flat_idx)
            rank_order = np.argsort(np.argsort(channel.ravel()[flat_idx]))
            flat_out   = result.ravel().copy()
            flat_out[flat_idx] = (max_val - edge_spread) + edge_spread * rank_order / max(n - 1, 1)
            result = flat_out.reshape(result.shape)

    return result


def levels_normalize_midpeaks(
    stretched:  np.ndarray,
    peak_width: int,
    rng_spike:  np.random.Generator,
    max_val:    float = 255.0,
) -> np.ndarray:
    """
    Anti-spike filter — smooths histogram bins near quantization gaps.

    A bin qualifies as a peak if it has at least one zero bin within
    peak_width positions. Targeted TPDF dithering is applied to qualifying
    pixels using a single pre-generated noise field (not per-bin), making
    the operation O(1) in the number of qualifying bins.

    Args:
        stretched  : 2D float32 array after edge spread
        peak_width : 1–10, distance from a gap that qualifies a bin
        rng_spike  : np.random.Generator (independent from gap dithering)
        max_val    : 255.0 for 8-bit, 65535.0 for 16-bit

    Returns:
        Float32 array with peak bins redistributed
    """
    n_bins = int(max_val) + 1
    result = stretched.copy()
    s_int  = np.clip(np.round(result).astype(np.int64), 0, int(max_val))
    s_hist = np.bincount(s_int.ravel(), minlength=n_bins).astype(np.float64)

    # Mid-range: exclude edge-spread zones (bins 0–edge and max-edge–max)
    edge_bins = int(EDGE_SPREAD_RATIO * max_val) + 1
    lo = edge_bins
    hi = n_bins - edge_bins

    gap_bins = set(int(b) for b in range(lo, hi) if s_hist[b] == 0)
    if not gap_bins:
        return result

    amp  = (peak_width / 2.0) * (max_val / 255.0)  # scale amplitude with bit depth
    half = amp / 2.0

    # Generate noise once for the full channel
    noise = (rng_spike.uniform(-half, half, result.shape).astype(np.float32) +
             rng_spike.uniform(-half, half, result.shape).astype(np.float32))

    # Vectorized near-gap detection via sliding window
    gap_arr = np.zeros(n_bins, dtype=bool)
    for g in gap_bins:
        gap_arr[g] = True

    from numpy.lib.stride_tricks import sliding_window_view
    pad      = peak_width
    padded   = np.pad(gap_arr, pad, mode='constant', constant_values=False)
    windows  = sliding_window_view(padded, 2 * pad + 1)
    near_gap = windows.any(axis=1)   # shape (n_bins,)

    # Build mask: all pixels in qualifying non-gap bins
    qualify_mask = np.zeros(result.shape, dtype=bool)
    for b in range(lo, hi):
        if s_hist[b] == 0 or not near_gap[b]:
            continue
        qualify_mask |= (s_int == b)

    result = np.where(qualify_mask, np.clip(result + noise, 0.0, max_val), result)
    return result


def levels_normalize_gaps(
    stretched: np.ndarray,
    scale:     float,
    rng_gap:   np.random.Generator,
    max_val:   float = 255.0,
) -> np.ndarray:
    """
    Anti-comb filter — TPDF gap dithering.

    Fills quantization gaps from non-integer stretch scale factors.
    At 16-bit the gaps are far smaller (1/65535 vs 1/255) and largely
    invisible, but dithering is still applied for completeness.

    Amplitude formula is ratio-based so it works at any bit depth:
      amplitude = max(1.0, (scale / 1.275) ^ 2.2) × (max_val / 255)

    Args:
        stretched : 2D float32 array
        scale     : stretch scale factor
        rng_gap   : np.random.Generator
        max_val   : 255.0 for 8-bit, 65535.0 for 16-bit

    Returns:
        Float32 array with gaps filled
    """
    amplitude = max(1.0, (scale / 1.275) ** 2.2) * (max_val / 255.0)
    half      = amplitude / 2.0
    noise     = (rng_gap.uniform(-half, half, stretched.shape).astype(np.float32) +
                 rng_gap.uniform(-half, half, stretched.shape).astype(np.float32))
    return np.clip(stretched + noise, 0.0, max_val)


def levels_auto_gamma(
    stretched:    np.ndarray,
    gamma_target: float,
    max_val:      float = 255.0,
) -> np.ndarray:
    """
    Auto gamma correction — pushes mean brightness toward gamma_target.

    Formula: gamma = log(current_mean_norm) / log(target_norm)
    Applied: output = (input / max_val) ^ (1 / gamma) × max_val
    Black and white stay anchored. gamma_target is always on 0–255 scale
    regardless of bit depth — it is normalised internally.

    Args:
        stretched    : 2D float32 array, values 0–max_val
        gamma_target : target mean brightness 0–255 (normalised internally)
        max_val      : 255.0 for 8-bit, 65535.0 for 16-bit

    Returns:
        Float32 array with gamma correction applied
    """
    current_mean = float(stretched.mean())
    low_guard    = 0.5 * (max_val / 255.0)
    high_guard   = max_val - low_guard

    if not (low_guard < current_mean < high_guard):
        return stretched

    target_norm  = float(np.clip(gamma_target / 255.0, 0.01, 0.99))
    current_norm = float(np.clip(current_mean / max_val, 0.01, 0.99))
    gamma        = np.log(current_norm) / np.log(target_norm)
    gamma        = float(np.clip(gamma, GAMMA_MIN, GAMMA_MAX))

    if abs(gamma - 1.0) <= 0.01:
        return stretched

    norm = np.clip(stretched / max_val, 0.0, 1.0)
    return np.clip(np.power(norm, 1.0 / gamma) * max_val, 0.0, max_val)


# ─────────────────────────────────────────────────────────────────────────────
# Main function
# ─────────────────────────────────────────────────────────────────────────────

def img_levels_auto(
    image:               Image.Image,
    auto_normalize:      bool  = True,
    threshold:           float = 2.0,
    normalize_gaps:      bool  = True,
    normalize_midpeaks:  bool  = False,
    peak_width:          int   = 3,
    auto_gamma:          bool  = True,
    gamma_target:        float = 128.0,
    precision:           bool  = False,
) -> Image.Image:
    """
    Photoshop-style per-channel auto levels normalization.

    Args:
        image              : PIL Image (RGB)

        auto_normalize     : True  = apply auto levels (default).
                             False = passthrough, return image unchanged.

        threshold          : 0.0 … 100.0. Percent of total pixels per channel
                             used to determine black and white points.
                             ~1–2 = subtle,  ~5 = moderate,  ~10+ = aggressive.

        normalize_gaps     : True  = Anti-comb filter. TPDF dithering fills
                             quantization gaps. Independent of normalize_midpeaks.
                             Default: True.

        normalize_midpeaks : True  = Anti-spike filter. Smooths bins near gaps.
                             False = completely skipped regardless of other flags.
                             Default: False.

        peak_width         : 1 … 10. Distance from a gap that qualifies a bin
                             as a peak for smoothing. Only used when
                             normalize_midpeaks=True.
                             1  = only directly adjacent bins (surgical)
                             3  = within 3 bins of any gap (default)
                             10 = wide smoothing

        auto_gamma         : True  = auto per-channel gamma after stretch.
                             Default: True.

        gamma_target       : 0 … 255. Target mean brightness for auto gamma.
                             128 = neutral (default), 110 = moody, 150 = airy.
                             Always specified on 0–255 scale regardless of
                             precision setting.

        precision     : False = 8-bit pipeline, returns PIL Image RGB.
                             True  = 16-bit pipeline (65536 histogram bins),
                             returns PIL Image RGB encoded at 16-bit precision
                             scaled back to 8-bit output. Use for AI-generated
                             tensors where higher internal precision reduces
                             quantization artefacts before final 8-bit output.
                             Default: False.

    Returns:
        PIL Image (RGB)

    Pipeline per channel:
        1. levels_detect_points   — black / white point via threshold
        2. levels_stretch         — linear stretch to [0 … max_val]
        3. levels_edge_spread     — rank-based edge spread (always on)
        4. levels_normalize_midpeaks — peak smoothing (if normalize_midpeaks)
        5. levels_normalize_gaps  — TPDF gap dithering (if normalize_gaps)
        6. levels_auto_gamma      — gamma correction (if auto_gamma)
    """
    img = image.convert("RGB")

    if not auto_normalize:
        return img

    if not (0.0 <= threshold <= 100.0):
        raise ValueError(f"threshold must be 0.0–100.0, got {threshold}")
    if not (0.0 <= gamma_target <= 255.0):
        raise ValueError(f"gamma_target must be 0–255, got {gamma_target}")
    if not (1 <= peak_width <= 10):
        raise ValueError(f"peak_width must be 1–10, got {peak_width}")

    # ── Bit depth configuration ───────────────────────────────────────────────
    max_val = 65535.0 if precision else 255.0

    # ── Load image into float array ───────────────────────────────────────────
    # Always read as 8-bit uint8 from PIL, then scale up to max_val if needed
    arr_8 = np.array(img, dtype=np.float32)          # 0–255 always
    if precision:
        arr = arr_8 * (65535.0 / 255.0)              # scale to 0–65535
    else:
        arr = arr_8

    out = np.empty_like(arr)

    for ch in range(3):
        rng_gap   = np.random.default_rng(ch)
        rng_spike = np.random.default_rng(ch + 100)

        channel = arr[:, :, ch]

        # 1. Detect black / white points
        black_point, white_point, scale = levels_detect_points(channel, threshold, max_val)

        # 2. Stretch
        stretched = levels_stretch(channel, black_point, white_point, max_val)

        # 3. Edge spread (always on)
        stretched = levels_edge_spread(channel, stretched, black_point, white_point, max_val)

        # 4. Peak smoothing (before gap dithering)
        if normalize_midpeaks:
            stretched = levels_normalize_midpeaks(stretched, peak_width, rng_spike, max_val)

        # 5. Gap dithering
        if normalize_gaps:
            stretched = levels_normalize_gaps(stretched, scale, rng_gap, max_val)

        # 6. Auto gamma
        if auto_gamma:
            stretched = levels_auto_gamma(stretched, gamma_target, max_val)

        out[:, :, ch] = stretched

    # ── Convert back to uint8 for PIL output ──────────────────────────────────
    if precision:
        # Scale 16-bit result back to 8-bit for PIL output
        out_8 = np.clip(out * (255.0 / 65535.0), 0, 255).astype(np.uint8)
    else:
        out_8 = np.clip(out, 0, 255).astype(np.uint8)

    return Image.fromarray(out_8, mode="RGB")
