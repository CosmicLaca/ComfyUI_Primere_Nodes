import numpy as np
from PIL import Image


# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

EDGE_SPREAD    = 8.0    # bins to spread clipped edge pixels across
SPIKE_WIDTH    = 8      # max run width (bins) for classic isolated spike
SPIKE_RATIO    = 3.0    # classic spike: peak must exceed this × context median
SPIKE_RATIO_GN = 1.5    # gap-neighbor spike: lower threshold (single bin adj. to gap)
SPIKE_SEARCH   = 8      # bins each side used to measure neighbor context
SPIKE_AMP_MAX  = 4.0    # max dither amplitude for spike correction (pixels)
GAMMA_MIN      = 0.25   # clamp auto gamma to safe range
GAMMA_MAX      = 4.0


# ─────────────────────────────────────────────────────────────────────────────
# Step functions — callable independently
# ─────────────────────────────────────────────────────────────────────────────

def levels_detect_points(
    channel:    np.ndarray,
    threshold:  float,
) -> tuple[int, int, float]:
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
    piling them into bin 0 (creating an artificial edge spike), they are
    spread uniformly across [0 … EDGE_SPREAD] using rank ordering.
    Same for white-clipped pixels spread to [255-EDGE_SPREAD … 255].

    Rank ordering guarantees flat distribution regardless of how tightly
    input pixels cluster near the clipping boundary.

    Args:
        channel     : original 2D float32 channel (before stretch)
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
    stretched: np.ndarray,
    rng_spike: np.random.Generator,
) -> np.ndarray:
    """
    Anti-spike filter — mid-histogram spike removal.

    Runs on the raw stretched histogram BEFORE gap dithering so that
    gap-neighbor peaks are still detectable (dithering fills gaps and makes
    those peaks indistinguishable from the surrounding distribution).

    Detects two types of spikes in bins 9–246:

      Classic isolated spike:
        A run of consecutive non-zero bins where:
          - run width ≤ SPIKE_WIDTH (narrow — not a legitimate tonal concentration)
          - peak > SPIKE_RATIO × context median (significantly taller than context)

      Gap-neighbor peak:
        A single bin adjacent to a zero bin where:
          - peak > SPIKE_RATIO_GN × context median
        These are the "comb teeth" — bins inflated by the quantization pattern.
        They have a lower threshold because they are always isolated (one bin wide)
        and their context is artificially suppressed by the neighboring gap.

    Correction: targeted TPDF dithering applied ONLY to pixels in the detected
    bin. Amplitude = min((ratio − 1) × 1.5, SPIKE_AMP_MAX). Spike pixels spread
    into neighboring bins — the peak flattens without removing pixels from the image.

    Args:
        stretched : 2D float32 array, values 0–255, after edge spread
        rng_spike : np.random.Generator for spike dithering noise
                    (kept separate from gap dithering RNG)

    Returns:
        Float32 array with spike bins redistributed
    """
    result = stretched.copy()
    s_int  = np.clip(np.round(result).astype(np.int32), 0, 255)
    s_hist = np.bincount(s_int.ravel(), minlength=256).astype(np.float64)

    b = 9
    while b < 247:
        if s_hist[b] == 0:
            b += 1
            continue

        # Find extent of the current non-zero run
        run_start = b
        while b < 247 and s_hist[b] > 0:
            b += 1
        run_end   = b          # first zero bin after run (or 247)
        run_width = run_end - run_start

        # Context: SPIKE_SEARCH bins each side, non-zero only, median
        left_ctx  = s_hist[max(0, run_start - SPIKE_SEARCH):run_start]
        right_ctx = s_hist[run_end:min(256, run_end + SPIKE_SEARCH)]
        ctx_nz    = np.concatenate([left_ctx, right_ctx])
        ctx_nz    = ctx_nz[ctx_nz > 0]
        ctx_med   = float(np.median(ctx_nz)) if len(ctx_nz) > 0 else 1.0

        run_peak = s_hist[run_start:run_end].max()
        ratio    = run_peak / (ctx_med + 1e-6)

        # Classic isolated spike: narrow AND significantly above context
        is_classic = (run_width <= SPIKE_WIDTH and ratio > SPIKE_RATIO)

        # Gap-neighbor peaks: run is flanked by at least one zero bin AND
        # above the lower gap-neighbor threshold. Only single bins qualify
        # (run_width == 1) to avoid flattening multi-bin tonal concentrations.
        gap_left    = (run_start > 0   and s_hist[run_start - 1] == 0)
        gap_right   = (run_end   < 255 and s_hist[run_end]        == 0)
        is_gap_nbr  = (run_width == 1 and (gap_left or gap_right)
                       and ratio > SPIKE_RATIO_GN)

        if is_classic or is_gap_nbr:
            spike_amp = min((ratio - 1.0) * 1.5, SPIKE_AMP_MAX)
            half      = spike_amp / 2.0
            # Apply dithering to each bin in the run independently
            for bin_b in range(run_start, run_end):
                mask = (s_int == bin_b)
                if not mask.any():
                    continue
                extra = (rng_spike.uniform(-half, half, result.shape).astype(np.float32) +
                         rng_spike.uniform(-half, half, result.shape).astype(np.float32))
                result = np.where(mask, np.clip(result + extra, 0.0, 255.0), result)

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
    the edge-spread region so that edge jaggedness at higher threshold values
    is also smoothed.

    Noise model: TPDF (Triangular Probability Density Function) — sum of two
    independent uniform distributions. Zero mean → no brightness drift.
    Max pixel change = ±amplitude.

    Amplitude auto-scales from the stretch factor:
      amplitude = max(1.0, (scale / 1.275) ^ 2.2)

    Anchored at threshold=2 (scale≈1.275) where amplitude=1.0 (±1 px max).
    Power of 2.2 derived empirically: threshold=10 needs amplitude≈1.5 to
    reduce visible roughness to the same order as threshold=2.

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

    Computes per-channel gamma from the current mean and target:
      gamma = log(current_mean_norm) / log(target_norm)

    Applied as a power curve:
      output = (input / 255) ^ (1 / gamma) × 255

    Black (0) and white (255) stay anchored. Gamma is clamped to
    [GAMMA_MIN, GAMMA_MAX] to prevent extreme corrections on unusual images.

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
                             quantization gaps created by integer rounding
                             during stretch. Amplitude auto-scales with scale
                             factor. Default: True.

        normalize_midpeaks : True  = Anti-spike filter. Detects and flattens
                             isolated spikes in mid-histogram (bins 9–246)
                             BEFORE gap dithering. Two detection modes:
                             classic narrow spike (SPIKE_WIDTH, SPIKE_RATIO)
                             and single-bin gap-neighbor peak (SPIKE_RATIO_GN).
                             Default: False.

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
        4. levels_normalize_midpeaks — spike removal (if normalize_midpeaks)
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

        # 3. Edge spread (always on — eliminates boundary spikes)
        stretched = levels_edge_spread(channel, stretched, black_point, white_point)

        # 4. Spike removal (before gap dithering — gaps still visible in histogram)
        if normalize_midpeaks:
            stretched = levels_normalize_midpeaks(stretched, rng_spike)

        # 5. Gap dithering (after spike removal — fills remaining gaps cleanly)
        if normalize_gaps:
            stretched = levels_normalize_gaps(stretched, scale, rng_gap)

        # 6. Auto gamma
        if auto_gamma:
            stretched = levels_auto_gamma(stretched, gamma_target)

        out[:, :, ch] = stretched

    return Image.fromarray(out.astype(np.uint8), mode="RGB")
