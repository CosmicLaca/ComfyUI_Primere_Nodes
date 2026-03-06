from __future__ import annotations

import base64
from . import response_helper

def handle_response(api_result, schema=None, loaded_client=None, response_url=None):
    data_items = getattr(api_result, "data", None)
    if not data_items:
        return [None, None]

    image_tensors = []
    first_bytes = None
    for item in data_items:
        image_base64 = getattr(item, "b64_json", None)
        if not image_base64:
            continue

        image_bytes = base64.b64decode(image_base64)
        tensor = response_helper.bytes_to_tensor(image_bytes)
        if tensor is not None:
            image_tensors.append(tensor)
            if first_bytes is None:
                first_bytes = image_bytes

        return [response_helper.stack_image_tensors(image_tensors), first_bytes]
