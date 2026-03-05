from __future__ import annotations

from typing import Any
import random
from PIL import Image
import io
import numpy as np

def handle_reference_images(source_images: Any = None, temp_file_ref: str = "", loaded_client_for_upload: Any = None, **_: Any):
    output = source_images if isinstance(source_images, list) else []
    if not isinstance(source_images, list) or len(source_images) == 0:
        return []

    print('----------------------')
    print(len(source_images))
    print(type(source_images).__name__)

    for single_image in source_images:
        if single_image is not None and type(single_image).__name__ == "Tensor":
            # print(type(single_image).__name__)
            # print(type(single_image[0]).__name__)
            # raise RuntimeError(f"OAI test")
            r1 = random.randint(10000, 99999)
            image_np = (single_image[0].numpy() * 255).astype(np.uint8)
            img = Image.fromarray(image_np)
            img_byte_arr = io.BytesIO()
            img.save(img_byte_arr, format="PNG")
            img_byte_arr.seek(0)
            img_binary = img_byte_arr
            img_binary.name = f"image_{r1}.png"
            print(img_binary)
            output.append(img_binary)

    print('----------------------')
    print(len(output))
    print(type(output).__name__)
    print('----------------------')

    return output