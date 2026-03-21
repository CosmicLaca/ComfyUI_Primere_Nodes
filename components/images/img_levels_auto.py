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
                             gaps (zero bins) created by integer rounding during
                             stretch. Amplitude auto-scales with the stretch
                             factor so higher thresholds are smoothed correctly.
                             Default: True.  Label: Anti-comb filter.

        normalize_midpeaks : True  = detect and flatten isolated spikes in the
                             mid-histogram (bins 9–246) BEFORE gap dithering.
                             Detects two kinds:
                               - Classic isolated spikes: narrow run
                                 (≤ 8 bins) AND peak > 3.0× context median
                               - Gap-neighbor peaks: bin adjacent to a zero
                                 bin AND peak > 1.5× context median
                             Correction: targeted dithering on spike-bin pixels
                             only, amplitude proportional to excess ratio.
                             Default: False.  Label: Anti-spike filter.

        auto_gamma         : True  = auto per-channel gamma after stretch to
                             push mean brightness toward gamma_target.
                             Default: True.

        gamma_target       : 0 … 255. Target mean brightness for auto gamma.
                             128  = neutral 50% grey (default).
                             110  = slightly dark / moody.
                             150  = bright / airy.

    Returns:
        PIL Image (RGB)

    Pipeline order:
        1. Black / white point detection via cumulative histogram threshold
        2. Float stretch to [0 … 255]
        3. Rank-based edge spread (always on)
        4. Mid-histogram spike removal (normalize_midpeaks)
        5. TPDF gap dithering (normalize_gaps)
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

    # Constants — defined once, never reassigned inside the loop
    EDGE_SPREAD   = 8.0
    SPIKE_WIDTH   = 8
    SPIKE_RATIO   = 3.0
    SPIKE_RATIO_GN = 1.5   # gap-neighbor threshold (lower = more sensitive)
    SPIKE_SEARCH  = 8
    SPIKE_AMP_MAX = 4.0
    GAMMA_MIN     = 0.25
    GAMMA_MAX     = 4.0

    for ch in range(3):
        # Independent RNGs per channel — seeded by channel index so that
        # spike correction firing on one channel does not shift the noise
        # sequence of gap dithering on any other channel.
        rng_gap   = np.random.default_rng(ch)          # gap dithering
        rng_spike = np.random.default_rng(ch + 100)    # spike correction

        channel = arr[:, :, ch]

        # ── Step 1: Black / white point detection ─────────────────────────────
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

        # ── Step 2: Float stretch ──────────────────────────────────────────────
        scale     = 255.0 / (white_point - black_point)
        stretched = (channel - black_point) * scale
        stretched = np.clip(stretched, 0.0, 255.0)

        # ── Step 3: Rank-based edge spread (always on) ────────────────────────
        # Pixels below black_point clipped to 0 → spread uniformly to
        # [0 … EDGE_SPREAD] using rank ordering so the distribution is flat
        # regardless of how tightly the clipped pixels cluster near the boundary.
        # Same logic for white-clipped pixels spread to [255-EDGE_SPREAD … 255].
        if black_point > 0:
            below_mask = channel < black_point
            if below_mask.any():
                flat_idx   = np.where(below_mask.ravel())[0]
                n          = len(flat_idx)
                rank_order = np.argsort(np.argsort(channel.ravel()[flat_idx]))
                flat_out   = stretched.ravel().copy()
                flat_out[flat_idx] = EDGE_SPREAD * rank_order / max(n - 1, 1)
                stretched  = flat_out.reshape(stretched.shape)

        if white_point < 255:
            above_mask = channel > white_point
            if above_mask.any():
                flat_idx   = np.where(above_mask.ravel())[0]
                n          = len(flat_idx)
                rank_order = np.argsort(np.argsort(channel.ravel()[flat_idx]))
                flat_out   = stretched.ravel().copy()
                flat_out[flat_idx] = (255.0 - EDGE_SPREAD) + EDGE_SPREAD * rank_order / max(n - 1, 1)
                stretched  = flat_out.reshape(stretched.shape)

        # ── Step 4: Mid-histogram spike removal (BEFORE dithering) ────────────
        # Runs on the raw stretched histogram so gap-neighbor peaks are still
        # visible. After dithering, gaps fill and those peaks become part of
        # the continuous distribution — they would no longer be detectable.
        #
        # Detects two types:
        #   Classic spike:    narrow run (≤ SPIKE_WIDTH) AND
        #                     peak > SPIKE_RATIO × context median
        #   Gap-neighbor:     adjacent to a zero bin AND
        #                     peak > SPIKE_RATIO_GN × context median
        #
        # Correction: targeted TPDF dithering applied ONLY to pixels in the
        # detected bin. Amplitude = min((ratio-1) × 1.5, SPIKE_AMP_MAX).
        # Spike pixels spread into neighboring bins; the peak flattens.
        # Uses rng_spike — independent of gap dithering rng_gap.
        if normalize_midpeaks:
            s_int  = np.clip(np.round(stretched).astype(np.int32), 0, 255)
            s_hist = np.bincount(s_int.ravel(), minlength=256).astype(np.float64)

            for b in range(9, 247):
                if s_hist[b] == 0:
                    continue

                left_ctx = s_hist[max(0, b - SPIKE_SEARCH):b]
                right_ctx = s_hist[b + 1:min(256, b + SPIKE_SEARCH + 1)]
                ctx_nz   = np.concatenate([left_ctx, right_ctx])
                ctx_nz   = ctx_nz[ctx_nz > 0]
                ctx_med  = float(np.median(ctx_nz)) if len(ctx_nz) > 0 else 1.0
                ratio    = s_hist[b] / (ctx_med + 1e-6)

                gap_left  = (b > 0   and s_hist[b - 1] == 0)
                gap_right = (b < 255 and s_hist[b + 1] == 0)
                adjacent  = gap_left or gap_right

                is_classic    = (ratio > SPIKE_RATIO)
                is_gap_nbr    = (adjacent and ratio > SPIKE_RATIO_GN)

                if is_classic or is_gap_nbr:
                    spike_amp = min((ratio - 1.0) * 1.5, SPIKE_AMP_MAX)
                    half      = spike_amp / 2.0
                    mask      = (s_int == b)
                    if not mask.any():
                        continue
                    extra = (rng_spike.uniform(-half, half, channel.shape).astype(np.float32) +
                             rng_spike.uniform(-half, half, channel.shape).astype(np.float32))
                    stretched = np.where(mask,
                                         np.clip(stretched + extra, 0.0, 255.0),
                                         stretched)

        # ── Step 5: TPDF gap dithering ────────────────────────────────────────
        # Fills quantization gaps and smooths edge jaggedness.
        # Applied to ALL pixels (including edge-spread region).
        # Amplitude auto-scales with the stretch factor:
        #
        #   amplitude = max(1.0, (scale / 1.275) ^ 2.2)
        #
        # Anchored: threshold=2 (scale≈1.275) → amplitude=1.0 (±1 px).
        # Scales up for higher thresholds — thr=10 → amplitude≈1.5.
        # Zero mean bias (TPDF) confirmed at all amplitudes.
        if normalize_gaps:
            amplitude = max(1.0, (scale / 1.275) ** 2.2)
            half      = amplitude / 2.0
            noise     = (rng_gap.uniform(-half, half, channel.shape).astype(np.float32) +
                         rng_gap.uniform(-half, half, channel.shape).astype(np.float32))
            stretched = np.clip(stretched + noise, 0.0, 255.0)

        # ── Step 6: Auto gamma correction ─────────────────────────────────────
        # Computes per-channel gamma to push mean brightness toward gamma_target.
        # Formula: gamma = log(current_mean_norm) / log(target_norm)
        # Applied as power curve: output = (input/255)^(1/gamma) × 255
        # Black (0) and white (255) stay anchored.
        if auto_gamma:
            current_mean = float(stretched.mean())
            if 0.5 < current_mean < 254.5:
                target_norm  = float(np.clip(gamma_target / 255.0, 0.01, 0.99))
                current_norm = float(np.clip(current_mean / 255.0, 0.01, 0.99))
                gamma = np.log(current_norm) / np.log(target_norm)
                gamma = float(np.clip(gamma, GAMMA_MIN, GAMMA_MAX))
                if abs(gamma - 1.0) > 0.01:
                    norm      = np.clip(stretched / 255.0, 0.0, 1.0)
                    stretched = np.power(norm, 1.0 / gamma) * 255.0
                    stretched = np.clip(stretched, 0.0, 255.0)

        out[:, :, ch] = stretched

    return Image.fromarray(out.astype(np.uint8), mode="RGB")
