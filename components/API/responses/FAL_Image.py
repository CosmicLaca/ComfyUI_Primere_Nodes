from __future__ import annotations

from typing import Any
import json
import requests
import comfy.utils
import numpy as np
import torch
from PIL import Image
import fal_client
from io import BytesIO

def handle_response(api_result, schema=None, loaded_client=None, response_url=None):
    result_image = None
    image_list = []
    final_batch_img = []

    request_id = api_result.request_id
    status = loaded_client.status(response_url, request_id, with_logs=False)
    result = loaded_client.result(response_url, request_id)

    try:
        json_d = json.dumps(result)
        json_object = json.loads(json_d)
    except ValueError as e:
        raise RuntimeError(f"Invalid JSON response received: {api_result}")

    if 'images' in json_object and 'url' in json_object['images'][0]:
        remote_images = json_object['images']
        for remote_image in remote_images:
            if 'url' in remote_image:
                response = requests.get(remote_image['url'])
                result_image = Image.open(BytesIO(response.content))
                if result_image is not None:
                    result_image = result_image.convert("RGB")
                    result_image = np.array(result_image).astype(np.float32) / 255.0
                    result_image = torch.from_numpy(result_image)[None,]
                    final_batch_img.append(result_image)

        if type(final_batch_img).__name__ == "list" and len(final_batch_img) > 1:
            image_list = final_batch_img
            single_image_start = final_batch_img[0]
            batch_count = 0
            s = None
            for single_image in final_batch_img:
                if (batch_count + 1) < len(final_batch_img):
                    current_single_image = final_batch_img[batch_count + 1]
                    if single_image_start.shape[1:] != current_single_image.shape[1:]:
                        current_single_image = comfy.utils.common_upscale(current_single_image.movedim(-1, 1), single_image_start.shape[2], single_image_start.shape[1], "bilinear", "center").movedim(1, -1)
                    batch_count = batch_count + 1
                    if s is not None:
                        single_image = s
                    s = torch.cat((current_single_image, single_image), dim=0)
                    result_image = s

    return result_image