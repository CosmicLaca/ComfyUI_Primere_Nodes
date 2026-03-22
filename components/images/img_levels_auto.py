import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

EDGE_SPREAD   = 8.0    # bins to spread clipped edge pixels across
GAMMA_MIN     = 0.25   # clamp auto gamma to safe range
GAMMA_MAX     = 4.0


# ─────────────────────────────────────────────────────────────────────────────
# Step functions — callable independently
# ─────────────────────────────────────────────────────────────────────────────

def levels_detect_points(
    channel:   np.ndarray,
    threshold: float,
) -> tuple:
    """
    Detect black and white points from a single channel histogram.

    Args:
        channel   : 2D float32 array, values 0–255
        threshold : 0.0–100.0, percent of pixels to clip at each end

    Returns:
        (black_point, white_point, scale)
        scale = 255 / (white_point - black_point)
    """
    hist, _      = np.histogram(channel, bins=256, range=(0, 256))
    cumulative   = np.cumsum(hist)
    total_pixels = int(cumulative[-1])
    abs_cutoff   = total_pixels * (threshold / 100.0)

    black_point = 0
    for i in range(256):
        if cumulative[i] >= abs_cutoff:
            black_point = i
            break

    white_point = 255
    for i in range(255, -1, -1):
        if (total_pixels - cumulative[i]) >= abs_cutoff:
            white_point = i
            break

    if white_point <= black_point:
        white_point = min(black_point + 1, 255)

    scale = 255.0 / (white_point - black_point)
    return black_point, white_point, scale


def levels_stretch(
    channel:     np.ndarray,
    black_point: int,
    white_point: int,
) -> np.ndarray:
    """
    Linear stretch of channel values to [0 … 255].

    Args:
        channel     : 2D float32 array, values 0–255
        black_point : input value that maps to 0
        white_point : input value that maps to 255

    Returns:
        Stretched float32 array clipped to [0, 255]
    """
    scale     = 255.0 / (white_point - black_point)
    stretched = (channel - black_point) * scale
    return np.clip(stretched, 0.0, 255.0)


def levels_edge_spread(
    channel:     np.ndarray,
    stretched:   np.ndarray,
    black_point: int,
    white_point: int,
) -> np.ndarray:
    """
    Rank-based edge spread — always applied, not gated by any boolean.

    Pixels below black_point all clipped to 0 after stretch. Rather than
    piling them into bin 0, they are spread uniformly across [0 … EDGE_SPREAD]
    using rank ordering — flat distribution regardless of input clustering.
    Same for white-clipped pixels spread to [255-EDGE_SPREAD … 255].

    Args:
        channel     : original 2D float32 channel before stretch
        stretched   : 2D float32 array after stretch, values 0–255
        black_point : black point used in stretch
        white_point : white point used in stretch

    Returns:
        Float32 array with edge pixels redistributed
    """
    result = stretched.copy()

    if black_point > 0:
        below_mask = channel < black_point
        if below_mask.any():
            flat_idx   = np.where(below_mask.ravel())[0]
            n          = len(flat_idx)
            rank_order = np.argsort(np.argsort(channel.ravel()[flat_idx]))
            flat_out   = result.ravel().copy()
            flat_out[flat_idx] = EDGE_SPREAD * rank_order / max(n - 1, 1)
            result = flat_out.reshape(result.shape)

    if white_point < 255:
        above_mask = channel > white_point
        if above_mask.any():
            flat_idx   = np.where(above_mask.ravel())[0]
            n          = len(flat_idx)
            rank_order = np.argsort(np.argsort(channel.ravel()[flat_idx]))
            flat_out   = result.ravel().copy()
            flat_out[flat_idx] = (255.0 - EDGE_SPREAD) + EDGE_SPREAD * rank_order / max(n - 1, 1)
            result = flat_out.reshape(result.shape)

    return result


def levels_normalize_midpeaks(
    stretched:  np.ndarray,
    peak_width: int,
    rng_spike:  np.random.Generator,
) -> np.ndarray:
    """
    Anti-spike filter — smooths histogram peaks near quantization gaps.

    After integer stretch with scale > 1, a comb pattern appears: some output
    bins receive no pixels (gaps) while adjacent bins receive the displaced
    pixels and appear as thin peaks visually. This function redistributes
    pixels from peak bins into neighboring gap bins by applying targeted
    TPDF dithering only to pixels in bins within peak_width distance of a gap.

    Detection: a bin qualifies as a peak if it has at least one zero bin
    within peak_width positions on either side. No ratio threshold — the
    user controls sensitivity directly via peak_width.

    Correction: targeted TPDF dithering applied ONLY to pixels in the
    qualifying bin. Amplitude = peak_width / 2 pixels. Wider peak_width
    both catches more bins AND spreads their pixels further — double effect.

    Args:
        stretched  : 2D float32 array, values 0–255, after edge spread
        peak_width : 1–10. Distance from a gap within which a bin is
                     considered a peak and gets smoothed.
                     1  = only bins directly adjacent to gaps (surgical)
                     3  = bins within 3 of any gap (default, balanced)
                     10 = wide smoothing around all gap regions
        rng_spike  : np.random.Generator, kept separate from gap dithering

    Returns:
        Float32 array with peak bins redistributed toward gap bins
    """
    result = stretched.copy()
    s_int  = np.clip(np.round(result).astype(np.int32), 0, 255)
    s_hist = np.bincount(s_int.ravel(), minlength=256).astype(np.float64)

    # Build set of gap bins for fast lookup
    gap_bins = set(int(b) for b in range(9, 247) if s_hist[b] == 0)
    if not gap_bins:
        return result   # no gaps to smooth

    amp  = peak_width / 2.0
    half = amp / 2.0

    # Generate ONE noise array for the entire channel — all qualifying bins
    # use the same amplitude (peak_width / 2) so a single TPDF noise field
    # covers all of them. Each bin's mask selects which pixels receive it.
    # This reduces rng calls from 2 × N_bins to 2 total — the critical fix
    # for performance on large images with many qualifying bins.
    noise = (rng_spike.uniform(-half, half, result.shape).astype(np.float32) +
             rng_spike.uniform(-half, half, result.shape).astype(np.float32))

    # Build qualifying bin mask vectorized using numpy
    # A bin qualifies if any bin within peak_width distance is a gap.
    gap_arr   = np.zeros(256, dtype=bool)
    for g in gap_bins:
        gap_arr[g] = True

    # For each bin b, check if any position in [b-pw, b+pw] is a gap
    # Equivalent to convolving gap_arr with a window of width 2*peak_width+1
    from numpy.lib.stride_tricks import sliding_window_view
    pad       = peak_width
    padded    = np.pad(gap_arr, pad, mode='constant', constant_values=False)
    windows   = sliding_window_view(padded, 2 * pad + 1)  # shape (256, 2*pad+1)
    near_gap  = windows.any(axis=1)                        # shape (256,)

    # Build combined mask: all pixels in qualifying non-gap bins
    qualify_mask = np.zeros(result.shape, dtype=bool)
    for b in range(9, 247):
        if s_hist[b] == 0 or not near_gap[b]:
            continue
        qualify_mask |= (s_int == b)

    # Apply noise only to qualifying pixels
    result = np.where(qualify_mask, np.clip(result + noise, 0.0, 255.0), result)

    return result


def levels_normalize_gaps(
    stretched: np.ndarray,
    scale:     float,
    rng_gap:   np.random.Generator,
) -> np.ndarray:
    """
    Anti-comb filter — TPDF gap dithering.

    Fills quantization gaps (zero bins) created by integer rounding when
    the stretch scale factor is non-integer. Applied to ALL pixels including
    the edge-spread region.

    Noise model: TPDF — sum of two uniform distributions. Zero mean,
    max change = ±amplitude.

    Amplitude auto-scales:
      amplitude = max(1.0, (scale / 1.275) ^ 2.2)

      threshold=2  → scale≈1.28 → amplitude=1.00  (±1.0 px max)
      threshold=6  → scale≈1.43 → amplitude=1.28  (±1.3 px max)
      threshold=10 → scale≈1.53 → amplitude=1.49  (±1.5 px max)
      threshold=20 → scale≈2.02 → amplitude=2.76  (±2.8 px max)

    Args:
        stretched : 2D float32 array, values 0–255
        scale     : stretch scale factor (255 / tonal_range)
        rng_gap   : np.random.Generator for gap dithering noise

    Returns:
        Float32 array with quantization gaps filled
    """
    amplitude = max(1.0, (scale / 1.275) ** 2.2)
    half      = amplitude / 2.0
    noise     = (rng_gap.uniform(-half, half, stretched.shape).astype(np.float32) +
                 rng_gap.uniform(-half, half, stretched.shape).astype(np.float32))
    return np.clip(stretched + noise, 0.0, 255.0)


def levels_auto_gamma(
    stretched:    np.ndarray,
    gamma_target: float,
) -> np.ndarray:
    """
    Auto gamma correction — pushes mean brightness toward gamma_target.

    Formula: gamma = log(current_mean_norm) / log(target_norm)
    Applied as: output = (input / 255) ^ (1 / gamma) × 255
    Black (0) and white (255) stay anchored.

    Args:
        stretched    : 2D float32 array, values 0–255
        gamma_target : target mean brightness 0–255 (128 = neutral 50% grey)

    Returns:
        Float32 array with gamma correction applied
    """
    current_mean = float(stretched.mean())
    if not (0.5 < current_mean < 254.5):
        return stretched

    target_norm  = float(np.clip(gamma_target / 255.0, 0.01, 0.99))
    current_norm = float(np.clip(current_mean / 255.0, 0.01, 0.99))
    gamma        = np.log(current_norm) / np.log(target_norm)
    gamma        = float(np.clip(gamma, GAMMA_MIN, GAMMA_MAX))

    if abs(gamma - 1.0) <= 0.01:
        return stretched

    norm = np.clip(stretched / 255.0, 0.0, 1.0)
    return np.clip(np.power(norm, 1.0 / gamma) * 255.0, 0.0, 255.0)


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
                             quantization gaps created by integer rounding.
                             Amplitude auto-scales with stretch factor.
                             Independent of normalize_midpeaks.
                             Default: True.

        normalize_midpeaks : True  = Anti-spike filter. Smooths histogram bins
                             that are near quantization gaps, reducing the thin-
                             peak appearance of the comb pattern. Operates by
                             targeted dithering on qualifying bins only.
                             False = function is completely skipped, regardless
                             of normalize_gaps state.
                             Default: False.

        peak_width         : 1 … 10. Controls which bins qualify as peaks and
                             how far their pixels are spread.
                             1  = only bins directly adjacent to a gap
                             3  = bins within 3 positions of any gap (default)
                             10 = wide smoothing around all gap regions
                             Larger values catch more bins AND spread pixels
                             further (double effect). Only used when
                             normalize_midpeaks=True.

        auto_gamma         : True  = auto per-channel gamma after stretch to
                             push mean brightness toward gamma_target.
                             Default: True.

        gamma_target       : 0 … 255. Target mean brightness.
                             128 = neutral (default), 110 = moody, 150 = airy.

    Returns:
        PIL Image (RGB)

    Pipeline per channel:
        1. levels_detect_points   — black / white point via threshold
        2. levels_stretch         — linear stretch to [0 … 255]
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

    arr = np.array(img, dtype=np.float32)
    out = np.empty_like(arr)

    for ch in range(3):
        # Independent RNGs per channel — seeded by channel index so that
        # spike correction firing on one channel cannot shift the noise
        # sequence of gap dithering on any other channel.
        rng_gap   = np.random.default_rng(ch)
        rng_spike = np.random.default_rng(ch + 100)

        channel = arr[:, :, ch]

        # 1. Detect black / white points
        black_point, white_point, scale = levels_detect_points(channel, threshold)

        # 2. Stretch
        stretched = levels_stretch(channel, black_point, white_point)

        # 3. Edge spread (always on)
        stretched = levels_edge_spread(channel, stretched, black_point, white_point)

        # 4. Peak smoothing — completely skipped when normalize_midpeaks=False
        if normalize_midpeaks:
            stretched = levels_normalize_midpeaks(stretched, peak_width, rng_spike)

        # 5. Gap dithering — independent of normalize_midpeaks
        if normalize_gaps:
            stretched = levels_normalize_gaps(stretched, scale, rng_gap)

        # 6. Auto gamma
        if auto_gamma:
            stretched = levels_auto_gamma(stretched, gamma_target)

        out[:, :, ch] = stretched

    return Image.fromarray(out.astype(np.uint8), mode="RGB")
