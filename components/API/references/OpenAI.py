from __future__ import annotations

from typing import Any
import random
from PIL import Image
import io
import numpy as np

def handle_reference_images(img_binary_api: Any = None, single_image: Any = None, temp_file_ref: str = "", loaded_client_for_upload: Any = None, **_: Any):
    output = img_binary_api if isinstance(img_binary_api, list) else []

    if single_image is None or type(single_image).__name__ != "Tensor":
        return output

    r1 = random.randint(10000, 99999)
    image_np = (single_image[0].numpy() * 255).astype(np.uint8)
    img = Image.fromarray(image_np)
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format="PNG")
    img_byte_arr.seek(0)
    img_byte_arr.name = f"image_{r1}.png"
    output.append(img_byte_arr)

    return output