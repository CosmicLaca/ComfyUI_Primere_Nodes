import numpy as np
from PIL import Image


def img_levels_compress(
    image:           Image.Image,
    black_offset:    float = 0.0,
    white_offset:    float = 0.0,
    skip_if_no_clip: bool  = False,
    high_precision:  bool  = False,
) -> Image.Image:
    if not (0.0 <= black_offset <= 25.0):
        raise ValueError(f"black_offset must be 0.0–25.0, got {black_offset}")
    if not (0.0 <= white_offset <= 25.0):
        raise ValueError(f"white_offset must be 0.0–25.0, got {white_offset}")

    if black_offset == 0.0 and white_offset == 0.0:
        return image.convert("RGB")

    img     = image.convert("RGB")
    max_val = 65535.0 if high_precision else 255.0

    scale_factor = max_val / 255.0
    bo = black_offset * scale_factor   # e.g. 10 → 10 (8-bit) or 2570 (16-bit)
    wo = white_offset * scale_factor

    arr_8 = np.array(img, dtype=np.float32)          # 0–255 always
    arr   = arr_8 * scale_factor if high_precision else arr_8.copy()

    out = np.empty_like(arr)

    for ch in range(3):
        channel = arr[:, :, ch]
        if skip_if_no_clip:
            ch_min = float(channel.min())
            ch_max = float(channel.max())
            black_active = ch_min <= bo
            white_active = ch_max >= (max_val - wo)
        else:
            black_active = True
            white_active = True

        bo_eff = bo if black_active else 0.0
        wo_eff = wo if white_active else 0.0

        if bo_eff == 0.0 and wo_eff == 0.0:
            out[:, :, ch] = channel
            continue

        compress_range = max_val - wo_eff - bo_eff
        scale          = compress_range / max_val
        out[:, :, ch]  = bo_eff + channel * scale

    if high_precision:
        out_8f = np.clip(out * (255.0 / max_val), 0, 255)
    else:
        out_8f = np.clip(out, 0, 255)
    out_8 = np.clip(np.rint(out_8f), 0, 255).astype(np.uint8)

    return Image.fromarray(out_8, mode="RGB")
