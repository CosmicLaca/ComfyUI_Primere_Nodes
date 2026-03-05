from __future__ import annotations

from typing import Any

def handle_reference_images(img_binary_api: Any = None, temp_file_ref: str = "", loaded_client_for_upload: Any = None, **_: Any):
    output = img_binary_api if isinstance(img_binary_api, list) else []
    if temp_file_ref:
        output.append(temp_file_ref)
    return output
