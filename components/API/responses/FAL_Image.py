from __future__ import annotations

from . import response_helper
import json

def handle_response(api_result, schema=None, loaded_client=None, response_url=None):
    request_id = api_result.request_id
    loaded_client.status(response_url, request_id, with_logs=False)
    result = loaded_client.result(response_url, request_id)

    try:
        json_object = json.loads(json.dumps(result))
    except ValueError as exc:
        raise RuntimeError(f"Invalid JSON response received: {api_result}") from exc

    remote_images = json_object.get("images", [])
    image_tensors = []
    for remote_image in remote_images:
        image_url = remote_image.get("url") if isinstance(remote_image, dict) else None
        if not image_url:
            continue

        tensor = response_helper.url_to_tensor(image_url)
        if tensor is not None:
            image_tensors.append(tensor)

    return response_helper.stack_image_tensors(image_tensors)