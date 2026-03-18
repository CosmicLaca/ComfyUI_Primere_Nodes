import numpy as np
from PIL import Image


def img_levels_auto(
    image:          Image.Image,
    auto_normalize: bool  = True,
    threshold:      float = 2.0,
) -> Image.Image:

    img = image.convert("RGB")

    if not auto_normalize:
        return img

    if not (0.0 <= threshold <= 100.0):
        raise ValueError(f"threshold must be 0.0–100.0, got {threshold}")

    arr = np.array(img, dtype=np.float32)
    out = np.empty_like(arr)

    for ch in range(3):
        channel = arr[:, :, ch]

        hist, _ = np.histogram(channel, bins=256, range=(0, 256))

        peak         = hist.max()
        cutoff_count = peak * (threshold / 100.0)

        cumulative   = np.cumsum(hist)
        total_pixels = cumulative[-1]
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

        stretched = (channel - black_point) * (255.0 / (white_point - black_point))
        out[:, :, ch] = np.clip(stretched, 0, 255)

    return Image.fromarray(out.astype(np.uint8), mode="RGB")
