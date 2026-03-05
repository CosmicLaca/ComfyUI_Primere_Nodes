from __future__ import annotations

import base64
from io import BytesIO
from typing import Any
from PIL import Image

def handle_reference_images(source_images: Any = None, temp_file_ref: str = "", loaded_client_for_upload: Any = None, **_: Any):
    if not isinstance(source_images, list) or len(source_images) == 0:
        return []

    if type(source_images).__name__ == "list" and len(source_images) > 0:
        bf_source_image = source_images[0]
    else:
        bf_source_image = source_images

    single_image = bf_source_image[0]
    image_np = (single_image.numpy() * 255).astype("uint8")
    img_bf = Image.fromarray(image_np)
    img_byte_arr_bf = BytesIO()
    img_bf.save(img_byte_arr_bf, format="PNG")
    img_byte_arr_bf.seek(0)
    encoded_string = base64.b64encode(img_byte_arr_bf.read())
    return encoded_string.decode("ascii")
