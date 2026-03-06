from __future__ import annotations

import io
from typing import Any
import numpy as np
from PIL import Image
from google.genai import types


def handle_reference_images(img_binary_api: Any = None, temp_file_ref: str = "", loaded_client_for_upload: Any = None, **_: Any):
    if img_binary_api is None:
        return {}
    if type(img_binary_api).__name__ == "Tensor":
        img_array = (img_binary_api[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_array)
        buffer = io.BytesIO()
        pil_image.save(buffer, format="PNG")
        image_bytes = buffer.getvalue()
        return {"first_image": types.Image(image_bytes=image_bytes, mime_type="image/png")}
    return {}
