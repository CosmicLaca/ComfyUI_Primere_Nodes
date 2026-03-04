from __future__ import annotations

from . import response_helper

def handle_response(api_result, schema=None, loaded_client=None, response_url=None):
    generated_images = getattr(api_result, "generated_images", None) or []
    if not generated_images:
        return None
    image_obj = getattr(generated_images[0], "image", None)
    image_bytes = getattr(image_obj, "image_bytes", None)
    return response_helper.image_bytes_to_tensor(image_bytes)
