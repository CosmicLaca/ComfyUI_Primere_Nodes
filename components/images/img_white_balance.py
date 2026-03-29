import numpy as np
from PIL import Image


def img_white_balance(
    image:       Image.Image,
    temperature: float = 6500,
    tint:        float = 0,
) -> Image.Image:
    if temperature == 6500 and tint == 0:
        return image.convert("RGB")

    if not (2000 <= temperature <= 12000):
        raise ValueError(f"temperature must be 2000–12000K, got {temperature}")
    if not (-100 <= tint <= 100):
        raise ValueError(f"tint must be -100 … +100, got {tint}")

    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32) / 255.0

    def kelvin_to_rgb(K):
        K = K / 100.0
        if K <= 66:
            R = 255.0
        else:
            R = 329.698727446 * ((K - 60) ** -0.1332047592)
        R = np.clip(R, 0, 255)

        if K <= 66:
            G = 99.4708025861 * np.log(K) - 161.1195681661
        else:
            G = 288.1221695283 * ((K - 60) ** -0.0755148492)
        G = np.clip(G, 0, 255)

        if K >= 66:
            B = 255.0
        elif K <= 19:
            B = 0.0
        else:
            B = 138.5177312231 * np.log(K - 10) - 305.0447927307
        B = np.clip(B, 0, 255)

        return np.array([R, G, B]) / 255.0

    rgb_target  = kelvin_to_rgb(temperature)
    rgb_neutral = kelvin_to_rgb(6500)

    with np.errstate(divide='ignore', invalid='ignore'):
        gain = np.where(rgb_neutral > 0, rgb_target / rgb_neutral, 1.0)

    arr = arr * gain

    if tint != 0:
        tint_gain = 1.0 + (tint / 100.0) * 0.3
        arr[..., 1] = arr[..., 1] * tint_gain

    arr = np.clip(arr, 0, 1)
    return Image.fromarray((arr * 255).astype(np.uint8), mode="RGB")
