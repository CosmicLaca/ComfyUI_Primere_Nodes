from __future__ import annotations

from . import response_helper

def handle_response(api_result, schema=None, loaded_client=None, response_url=None):
    request_id = api_result.request_id
    loaded_client.status(response_url, request_id, with_logs=False)
    result = loaded_client.result(response_url, request_id)
    json_object = response_helper.load_json_object(result, "Invalid FAL response received")
    remote_images = json_object.get("images") or []
    image_urls = [item.get("url") for item in remote_images if isinstance(item, dict) and item.get("url")]
    return response_helper.merge_image_tensors(response_helper.image_urls_to_tensors(image_urls))