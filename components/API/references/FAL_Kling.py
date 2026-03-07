from __future__ import annotations

import os
import tempfile
from typing import Any

import numpy as np
from PIL import Image


def handle_reference_images(img_binary_api: Any = None, temp_file_ref: str = "", loaded_client_for_upload: Any = None, target_key: str = "first_image", **_: Any):
    if temp_file_ref:
        output = img_binary_api if isinstance(img_binary_api, list) else []
        if hasattr(loaded_client_for_upload, "upload_file"):
            output.append(loaded_client_for_upload.upload_file(temp_file_ref))
        else:
            output.append(temp_file_ref)
        return output

    if img_binary_api is not None and type(img_binary_api).__name__ == "Tensor":
        img_array = (img_binary_api[0].cpu().numpy() * 255).astype(np.uint8)
        pil_image = Image.fromarray(img_array)
        with tempfile.NamedTemporaryFile(suffix=".png", delete=False) as tmp:
            tmp_path = tmp.name
        pil_image.save(tmp_path, format="PNG")
        try:
            uploaded = loaded_client_for_upload.upload_file(tmp_path) if hasattr(loaded_client_for_upload, "upload_file") else tmp_path
        finally:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
        return {target_key: uploaded}

    if img_binary_api is not None and isinstance(img_binary_api, str) and img_binary_api:
        if os.path.isfile(img_binary_api) and hasattr(loaded_client_for_upload, "upload_file"):
            uploaded = loaded_client_for_upload.upload_file(img_binary_api)
        else:
            uploaded = img_binary_api
        return {target_key: uploaded}

    return img_binary_api if isinstance(img_binary_api, list) else []
