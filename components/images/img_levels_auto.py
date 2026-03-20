import numpy as np
from PIL import Image


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

        normalize_gaps     : True  = apply TPDF dithering to fill quantization
                                     gaps (zero bins) created by integer rounding
                                     during stretch. Max pixel change ±1, zero
                                     mean bias. Default: True.
                             False = skip gap filling, keep raw stretch result.

        normalize_midpeaks : True  = detect and flatten isolated thin spikes in
                                     the middle of the histogram (bins 9–246).
                                     A spike is a narrow run (≤ SPIKE_WIDTH bins)
                                     that is significantly taller than its
                                     surrounding context. Flattened to the mean
                                     of immediate neighbors.
                             False = leave mid-histogram peaks unchanged (default).
                                     Use False if your peaks may be valid tonal
                                     concentrations from the source image.

        auto_gamma         : True  = automatically compute per-channel gamma
                                     correction after stretch to push the mean
                                     brightness toward gamma_target.
                             False = no gamma correction (gamma = 1.0).

        gamma_target       : 0 … 255. Target mean brightness for auto gamma.
                             128  = neutral 50% grey (default).
                             110  = slightly dark / moody.
                             150  = bright / airy.
                             Only used when auto_gamma = True.

    Returns:
        PIL Image (RGB)

    Pipeline order (each step builds on the previous):
        1. Black / white point detection via threshold
        2. Float stretch to [0 … 255]
        3. Rank-based edge spread (always on — eliminates clipping spikes)
        4. TPDF gap dithering (normalize_gaps)
        5. Mid-histogram spike removal (normalize_midpeaks)
        6. Auto gamma correction (auto_gamma)
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

    EDGE_SPREAD      = 8.0    # bins to spread clipped edge pixels across
    SPIKE_WIDTH      = 8      # max run width (bins) to be considered a spike
    SPIKE_RATIO      = 3.0    # spike peak must exceed this × neighbor mean
    SPIKE_SEARCH     = 8      # bins each side used to measure neighbor context
    GAMMA_MIN        = 0.25   # clamp auto gamma to safe range
    GAMMA_MAX        = 4.0

    rng = np.random.default_rng(0)

    for ch in range(3):
        channel      = arr[:, :, ch]
        hist, _      = np.histogram(channel, bins=256, range=(0, 256))
        cumulative   = np.cumsum(hist)
        total_pixels = cumulative[-1]
        abs_cutoff   = total_pixels * (threshold / 100.0)

        # ── Black / white point detection ─────────────────────────────────────
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

        # ── Step 1: Float stretch ──────────────────────────────────────────────
        scale     = 255.0 / (white_point - black_point)
        stretched = (channel - black_point) * scale
        stretched = np.clip(stretched, 0.0, 255.0)

        # ── Step 2: Rank-based edge spread (always on) ────────────────────────
        # Pixels below black_point clipped to 0 → spread to [0 … EDGE_SPREAD]
        # Pixels above white_point clipped to 255 → spread to [255-EDGE_SPREAD … 255]
        # Rank ordering guarantees flat distribution regardless of how pixels
        # cluster near the clipping boundary.
        if black_point > 0:
            below_mask = channel < black_point
            if below_mask.any():
                flat_idx    = np.where(below_mask.ravel())[0]
                n           = len(flat_idx)
                rank_order  = np.argsort(np.argsort(channel.ravel()[flat_idx]))
                flat_out    = stretched.ravel().copy()
                flat_out[flat_idx] = EDGE_SPREAD * rank_order / max(n - 1, 1)
                stretched   = flat_out.reshape(stretched.shape)

        if white_point < 255:
            above_mask = channel > white_point
            if above_mask.any():
                flat_idx    = np.where(above_mask.ravel())[0]
                n           = len(flat_idx)
                rank_order  = np.argsort(np.argsort(channel.ravel()[flat_idx]))
                flat_out    = stretched.ravel().copy()
                flat_out[flat_idx] = (255.0 - EDGE_SPREAD) + EDGE_SPREAD * rank_order / max(n - 1, 1)
                stretched   = flat_out.reshape(stretched.shape)

        # ── Step 3: TPDF gap dithering ────────────────────────────────────────
        # Applied only to in-range pixels (not the edge-spread region).
        # Triangular noise: zero mean, max ±1 pixel change.
        if normalize_gaps:
            in_range = np.ones(channel.shape, dtype=bool)
            if black_point > 0:
                in_range &= (channel >= black_point)
            if white_point < 255:
                in_range &= (channel <= white_point)

            noise = (rng.uniform(-0.5, 0.5, channel.shape).astype(np.float32) +
                     rng.uniform(-0.5, 0.5, channel.shape).astype(np.float32))
            stretched = np.where(in_range, stretched + noise, stretched)
            stretched = np.clip(stretched, 0.0, 255.0)

        # ── Step 4: Mid-histogram spike removal ───────────────────────────────
        # Operates on bins 9–246 only (edge bins handled by edge spread above).
        # Detects runs that are narrow AND significantly taller than context.
        # Correction: build a per-bin output LUT and remap pixels.
        if normalize_midpeaks:
            s_int  = np.clip(np.round(stretched).astype(np.int32), 0, 255)
            s_hist = np.bincount(s_int.ravel(), minlength=256).astype(np.float64)

            # Build identity LUT, then modify spike bins
            lut = np.arange(256, dtype=np.float64)

            i = 9   # start away from edge-spread zone
            while i < 247:
                if s_hist[i] > 0:
                    run_start = i
                    while i < 247 and s_hist[i] > 0:
                        i += 1
                    run_end   = i
                    run_width = run_end - run_start

                    if run_width <= SPIKE_WIDTH:
                        left_ctx  = s_hist[max(0, run_start - SPIKE_SEARCH):run_start]
                        right_ctx = s_hist[run_end:min(256, run_end + SPIKE_SEARCH)]
                        ctx_mean  = (left_ctx.mean() + right_ctx.mean()) / 2.0 + 1e-6

                        if s_hist[run_start:run_end].max() > SPIKE_RATIO * ctx_mean:
                            # Spike: replace with linear interpolation between
                            # immediate outer neighbors
                            v_left  = lut[run_start - 1]
                            v_right = lut[min(run_end, 255)]
                            for g in range(run_start, run_end):
                                t      = (g - run_start + 1) / float(run_width + 1)
                                lut[g] = v_left * (1.0 - t) + v_right * t
                else:
                    i += 1

            # Apply LUT with linear interpolation
            idx_f = np.clip(np.floor(stretched).astype(np.int32), 0, 254)
            frac  = stretched - np.floor(stretched)
            stretched = np.clip(lut[idx_f]*(1.0-frac) + lut[idx_f+1]*frac, 0, 255)

        # ── Step 5: Auto gamma correction ─────────────────────────────────────
        # Find gamma such that mean(output^(1/gamma)) ≈ gamma_target / 255.
        # Formula: gamma = log(current_mean_norm) / log(target_norm)
        # Applied as a power curve: output = (input/255)^(1/gamma) * 255
        # Black (0) and white (255) stay anchored.
        if auto_gamma:
            current_mean = stretched.mean()
            if current_mean > 0.5 and current_mean < 254.5:
                target_norm  = np.clip(gamma_target / 255.0, 0.01, 0.99)
                current_norm = np.clip(current_mean / 255.0, 0.01, 0.99)
                gamma = np.log(current_norm) / np.log(target_norm)
                gamma = float(np.clip(gamma, GAMMA_MIN, GAMMA_MAX))

                if abs(gamma - 1.0) > 0.01:   # skip if negligible
                    norm    = np.clip(stretched / 255.0, 0.0, 1.0)
                    stretched = np.power(norm, 1.0 / gamma) * 255.0
                    stretched = np.clip(stretched, 0.0, 255.0)

        out[:, :, ch] = stretched

    return Image.fromarray(out.astype(np.uint8), mode="RGB")
