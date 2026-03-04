from __future__ import annotations

from . import response_helper

def handle_response(api_result, schema=None, loaded_client=None, response_url=None):
    data = getattr(api_result, "data", None) or []
    tensors = []
    for item in data:
        tensor = response_helper.base64_to_tensor(getattr(item, "b64_json", None))
        if tensor is not None:
            tensors.append(tensor)

    return response_helper.merge_image_tensors(tensors)