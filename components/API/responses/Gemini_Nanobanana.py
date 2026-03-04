from __future__ import annotations

from . import response_helper

def handle_response(api_result, schema=None, loaded_client=None, response_url=None):
    candidates = getattr(api_result, "candidates", None) or []
    if not candidates:
        return None

    content = getattr(candidates[0], "content", None)
    parts = getattr(content, "parts", None) or []
    tensors = []
    for part in parts:
        inline_data = getattr(part, "inline_data", None)
        image_bytes = getattr(inline_data, "data", None)
        tensor = response_helper.image_bytes_to_tensor(image_bytes)
        if tensor is not None:
            tensors.append(tensor)

    return response_helper.merge_image_tensors(tensors)