import numpy as np
from PIL import Image


def img_levels_compress(
    image:           Image.Image,
    black_offset:    float = 0.0,
    white_offset:    float = 0.0,
    skip_if_no_clip: bool  = False,
    high_precision:  bool  = False,
) -> Image.Image:
    """
    Proportional histogram compression — the inverse of auto-levels stretch.

    Shifts the black point upward and the white point downward by the specified
    offsets, compressing all pixel values proportionally into the narrower range
    without clipping any pixels. The original tonal relationships are preserved.

    This is the manual counterpart to img_levels_auto: while auto-levels
    automatically STRETCHES the histogram to fill [0, 255], this function
    COMPRESSES the histogram to avoid pure black (0) and pure white (255),
    which is useful for preventing full-black / full-white areas in compositing
    or for adding a subtle lift before further processing.

    Args:
        image           : PIL Image (RGB)

        black_offset    : 0.0 … 25.0 (0–10% of 255).
                          Lifts the output black point.
                          0   = no change (pixels can still reach 0).
                          10  = darkest pixel maps to 10 instead of 0.
                          25  = darkest pixel maps to 25 (maximum lift).

        white_offset    : 0.0 … 25.0 (0–10% of 255).
                          Lowers the output white point.
                          0   = no change (pixels can still reach 255).
                          5   = brightest pixel maps to 250 instead of 255.
                          25  = brightest pixel maps to 230 (maximum pull-down).

        skip_if_no_clip : Controls per-side, per-channel behaviour when the
                          channel's data does not actually reach the extreme.

                          False (default) — Always apply:
                            Both offsets are applied to every channel regardless
                            of whether the channel data reaches 0 or 255.
                            Use this for a uniform look across all channels.

                          True — Skip if no data reaches the extreme:
                            Each offset is applied to a channel side only if
                            the channel's data reaches that side threshold:
                              Black side: apply only if channel_min <= black_offset
                              White side: apply only if channel_max >= (255 - white_offset)
                            If a channel's brightest pixel is 245 and white_offset=5
                            (threshold = 250), the white compression is skipped for
                            that channel — it has no pixels to protect.
                            Each side is evaluated independently.
                            Use this when channels have unequal tonal ranges and
                            you only want to protect sides that actually clip.

        high_precision  : False = 8-bit pipeline (default, 0–255 range).
                          True  = 16-bit pipeline (0–65535 range, offsets
                                  scale proportionally with max_val).
                          Both return PIL Image RGB (uint8). 16-bit precision
                          is internal — more accurate intermediate computation.

    Returns:
        PIL Image (RGB)

    Formula (per active side):
        Both sides active:
            output = black_offset + input × (max_val − white_offset − black_offset) / max_val

        Black side only:
            output = black_offset + input × (max_val − black_offset) / max_val

        White side only:
            output = input × (max_val − white_offset) / max_val

        Neither side active (skip_if_no_clip=True, no data at extremes):
            output = input (passthrough)

    Passthrough conditions:
        - black_offset == 0 AND white_offset == 0
        - OR skip_if_no_clip=True and no channel side reaches its threshold
    """
    if not (0.0 <= black_offset <= 25.0):
        raise ValueError(f"black_offset must be 0.0–25.0, got {black_offset}")
    if not (0.0 <= white_offset <= 25.0):
        raise ValueError(f"white_offset must be 0.0–25.0, got {white_offset}")

    if black_offset == 0.0 and white_offset == 0.0:
        return image.convert("RGB")

    img     = image.convert("RGB")
    max_val = 65535.0 if high_precision else 255.0

    # Scale offsets from 8-bit units to internal precision
    scale_factor = max_val / 255.0
    bo = black_offset * scale_factor   # e.g. 10 → 10 (8-bit) or 2570 (16-bit)
    wo = white_offset * scale_factor

    # Load as float32, scale to internal range
    arr_8 = np.array(img, dtype=np.float32)          # 0–255 always
    arr   = arr_8 * scale_factor if high_precision else arr_8.copy()

    out = np.empty_like(arr)

    for ch in range(3):
        channel = arr[:, :, ch]

        if skip_if_no_clip:
            # Per-side threshold check:
            # Black side fires only if the channel has pixels at or below the
            # new black point threshold.
            # White side fires only if the channel has pixels at or above the
            # new white point threshold.
            ch_min = float(channel.min())
            ch_max = float(channel.max())
            black_active = ch_min <= bo
            white_active = ch_max >= (max_val - wo)
        else:
            black_active = True
            white_active = True

        # Determine effective offsets for this channel
        bo_eff = bo if black_active else 0.0
        wo_eff = wo if white_active else 0.0

        if bo_eff == 0.0 and wo_eff == 0.0:
            # Neither side active — passthrough this channel
            out[:, :, ch] = channel
            continue

        # Proportional compression formula
        # output = bo_eff + input × (max_val − wo_eff − bo_eff) / max_val
        compress_range = max_val - wo_eff - bo_eff
        scale          = compress_range / max_val
        out[:, :, ch]  = bo_eff + channel * scale

    # Convert back to uint8
    if high_precision:
        out_8 = np.clip(out * (255.0 / max_val), 0, 255).astype(np.uint8)
    else:
        out_8 = np.clip(out, 0, 255).astype(np.uint8)

    return Image.fromarray(out_8, mode="RGB")
