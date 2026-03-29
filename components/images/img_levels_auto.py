import numpy as np
from PIL import Image

EDGE_SPREAD_RATIO = 8.0 / 255.0   # edge spread as fraction of max value
                                    # 8-bit:  8 bins,  16-bit: 2056 bins
GAMMA_MIN         = 0.25
GAMMA_MAX         = 4.0

def levels_detect_points(
    channel:    np.ndarray,
    threshold:  float,
    max_val:    float = 255.0,
) -> tuple:
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
    n_bins = int(max_val) + 1
    result = stretched.copy()
    s_int  = np.clip(np.round(result).astype(np.int64), 0, int(max_val))
    s_hist = np.bincount(s_int.ravel(), minlength=n_bins).astype(np.float64)

    edge_bins = int(EDGE_SPREAD_RATIO * max_val) + 1
    lo = edge_bins
    hi = n_bins - edge_bins

    gap_bins = set(int(b) for b in range(lo, hi) if s_hist[b] == 0)
    if not gap_bins:
        return result

    amp  = (peak_width / 2.0) * (max_val / 255.0)  # scale amplitude with bit depth
    half = amp / 2.0

    noise = (rng_spike.uniform(-half, half, result.shape).astype(np.float32) +
             rng_spike.uniform(-half, half, result.shape).astype(np.float32))

    gap_arr = np.zeros(n_bins, dtype=bool)
    for g in gap_bins:
        gap_arr[g] = True

    from numpy.lib.stride_tricks import sliding_window_view
    pad      = peak_width
    padded   = np.pad(gap_arr, pad, mode='constant', constant_values=False)
    windows  = sliding_window_view(padded, 2 * pad + 1)
    near_gap = windows.any(axis=1)   # shape (n_bins,)

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
    amplitude = max(1.0, (scale / 1.275) ** 2.2) * (max_val / 255.0)
    half      = amplitude / 2.0
    noise     = (rng_gap.uniform(-half, half, stretched.shape).astype(np.float32) + rng_gap.uniform(-half, half, stretched.shape).astype(np.float32))
    return np.clip(stretched + noise, 0.0, max_val)


def levels_auto_gamma(
    stretched:    np.ndarray,
    gamma_target: float,
    max_val:      float = 255.0,
) -> np.ndarray:
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
    seed:                int | None = None,
) -> Image.Image:
    img = image.convert("RGB")

    if not auto_normalize:
        return img

    if not (0.0 <= threshold <= 100.0):
        raise ValueError(f"threshold must be 0.0–100.0, got {threshold}")
    if not (0.0 <= gamma_target <= 255.0):
        raise ValueError(f"gamma_target must be 0–255, got {gamma_target}")
    if not (1 <= peak_width <= 10):
        raise ValueError(f"peak_width must be 1–10, got {peak_width}")

    max_val = 65535.0 if precision else 255.0
    arr_8 = np.array(img, dtype=np.float32)          # 0–255 always
    if precision:
        arr = arr_8 * (65535.0 / 255.0)              # scale to 0–65535
    else:
        arr = arr_8
    out = np.empty_like(arr)
    base_rng = np.random.default_rng(seed)
    for ch in range(3):
        channel_seed_gap = int(base_rng.integers(0, 2**31 - 1))
        channel_seed_spike = int(base_rng.integers(0, 2**31 - 1))
        rng_gap = np.random.default_rng(channel_seed_gap)
        rng_spike = np.random.default_rng(channel_seed_spike)
        channel = arr[:, :, ch]
        black_point, white_point, scale = levels_detect_points(channel, threshold, max_val)
        stretched = levels_stretch(channel, black_point, white_point, max_val)
        stretched = levels_edge_spread(channel, stretched, black_point, white_point, max_val)
        if normalize_midpeaks:
            stretched = levels_normalize_midpeaks(stretched, peak_width, rng_spike, max_val)
        if normalize_gaps:
            stretched = levels_normalize_gaps(stretched, scale, rng_gap, max_val)
        if auto_gamma:
            stretched = levels_auto_gamma(stretched, gamma_target, max_val)
        out[:, :, ch] = stretched

    if precision:
        out_8 = np.clip(out * (255.0 / 65535.0), 0, 255).astype(np.uint8)
    else:
        out_8 = np.clip(out, 0, 255).astype(np.uint8)

    return Image.fromarray(out_8, mode="RGB")
