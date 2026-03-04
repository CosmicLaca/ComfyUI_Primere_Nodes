from __future__ import annotations

from . import response_helper

def handle_response(api_result, schema=None, loaded_client=None, response_url=None):
    json_object = response_helper.load_json_object(getattr(api_result, "text", api_result), "Input object failed")
    polling_url = json_object.get("polling_url")
    if not polling_url:
        raise RuntimeError(f"No polling_url in response: {json_object}")

    result_payload = response_helper.poll_result_endpoint(polling_url=polling_url, response_url=response_url)
    image_url = ((result_payload.get("result") or {}).get("sample"))
    if not image_url:
        raise RuntimeError(f"No result sample image in response: {result_payload}")

    tensor = response_helper.image_url_to_tensor(image_url)
    if tensor is None:
        raise RuntimeError("No result image...")

    return response_helper.merge_image_tensors([tensor])