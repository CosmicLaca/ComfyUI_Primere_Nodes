from __future__ import annotations

import base64
from io import BytesIO
from typing import Any

from PIL import Image


def _encode_tensor_to_base64(image_tensor: Any) -> str | None:
    if image_tensor is None or type(image_tensor).__name__ != "Tensor":
        return None

    image_data = image_tensor[0].numpy() if len(image_tensor.shape) == 4 else image_tensor.numpy()
    image_np = (image_data * 255).astype("uint8")
    img_bf = Image.fromarray(image_np)
    img_byte_arr_bf = BytesIO()
    img_bf.save(img_byte_arr_bf, format="PNG")
    img_byte_arr_bf.seek(0)
    encoded_string = base64.b64encode(img_byte_arr_bf.read())
    return encoded_string.decode("ascii")


def handle_reference_images(source_images: Any = None, **_: Any):
    if not isinstance(source_images, list) or len(source_images) == 0:
        return []

    encoded_images: list[str] = []
    for source_image in source_images:
        encoded = _encode_tensor_to_base64(source_image)
        if encoded:
            encoded_images.append(encoded)
        if len(encoded_images) >= 8:
            break

    if len(encoded_images) == 0:
        return []

    payload: dict[str, str] = {}
    for index, encoded in enumerate(encoded_images, start=1):
        key_name = "input_image" if index == 1 else f"input_image_{index}"
        payload[key_name] = encoded

    return payload
