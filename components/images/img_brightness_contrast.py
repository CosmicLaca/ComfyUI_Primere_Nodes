import numpy as np
from PIL import Image

def img_brightness_contrast(image: Image.Image, brightness: float = 0, contrast: float = 0, use_legacy: bool  = False,) -> Image.Image:
    img = image.convert("RGB")
    arr = np.array(img, dtype=np.float32)

    if use_legacy:
        if brightness != 0:
            arr = arr + brightness

        if contrast != 0:
            if contrast >= 0:
                scale = 1.0 + (contrast / 100.0)
            else:
                scale = 1.0 + (contrast / 50.0)
                scale = max(scale, 0.0)
            arr = (arr - 128.0) * scale + 128.0

        arr = np.clip(arr, 0, 255)

    else:
        arr /= 255.0

        if brightness != 0:
            b = brightness / 150.0
            gamma = 1.0 - b * 0.9
            gamma = max(gamma, 0.01)
            arr = np.power(np.clip(arr, 0, 1), gamma)

        if contrast != 0:
            c = contrast / 100.0
            if c >= 0:
                factor = 1.0 + c * 1.5
            else:
                factor = 1.0 + c
                factor = max(factor, 0.1)
            arr = (arr - 0.5) * factor + 0.5

        arr = np.clip(arr * 255.0, 0, 255)

    return Image.fromarray(arr.astype(np.uint8), mode="RGB")
