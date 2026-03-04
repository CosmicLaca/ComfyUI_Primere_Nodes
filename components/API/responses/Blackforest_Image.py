from __future__ import annotations

from typing import Any
import json
import requests
import comfy.utils
import numpy as np
import torch
from PIL import Image
from io import BytesIO
import time
from urllib.parse import urlparse

def handle_response(api_result, schema=None, loaded_client=None, response_url=None):
    result_image = None
    image_list = []
    final_batch_img = []
    status_accepted = ['Ready', 'Pending', 'Task not found']
    parsed_url = urlparse(response_url)
    blackforest_api_region = parsed_url.netloc or None
    path_parts = [part for part in parsed_url.path.split("/") if part]
    blackforest_api_version = path_parts[0] if len(path_parts) > 0 else None

    try:
        json_object = json.loads(api_result.text)
    except ValueError as e:
        raise RuntimeError(f"Input object failed: {api_result}")

    if 'polling_url' in json_object:
        resp = requests.get(json_object['polling_url'])
        resp_json_object = json.loads(resp.text)

        status = 'Start'
        request_tryout = 0
        error_tryout = 0

        while error_tryout <= 1:
            while status != 'Ready' or request_tryout <= 20:
                url_res = f"https://{blackforest_api_region}/{blackforest_api_version}/get_result"
                querystring = {"id": resp_json_object['id']}
                response = requests.request("GET", url_res, params=querystring)
                resp_json_object = json.loads(response.text)
                status = resp_json_object['status']
                if status not in status_accepted:
                    resp_error = requests.get(json_object['polling_url'])
                    resp_error_json_object = json.loads(resp_error.text)
                    error_status = resp_error_json_object['status']
                    raise RuntimeError(f"Response status: {status}, error status: {error_status}")
                if status == 'Ready':
                    break
                time.sleep(2)
                request_tryout = request_tryout + 1
            time.sleep(1)
            if status == 'Ready':
                break
            error_tryout = error_tryout + 1

        if status == 'Ready':
            image_url = resp_json_object['result']['sample']
            response = requests.get(image_url)
            result_image = Image.open(BytesIO(response.content))
            if result_image is not None:
                result_image = result_image.convert("RGB")
                result_image = np.array(result_image).astype(np.float32) / 255.0
                result_image = torch.from_numpy(result_image)[None,]
                final_batch_img.append(result_image)
            else:
                raise RuntimeError(f"No result image...")
    else:
        raise RuntimeError(f"No polling_url in response: {json_object}")

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