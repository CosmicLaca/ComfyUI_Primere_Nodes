from __future__ import annotations

from typing import Any
import json
import requests
import comfy.utils
import numpy as np
import torch
from PIL import Image
from io import BytesIO
import base64

def handle_response(api_result, schema=None, loaded_client=None, response_url=None):
    result_image = None
    image_list = []
    final_batch_img = []

    if type(api_result.data).__name__ == "list" and len(api_result.data) > 1:
        batch_images = []
        for single_result in result.data:
            image_base64 = single_result.b64_json
            image_bytes = base64.b64decode(image_base64)
            result_image = Image.open(BytesIO(image_bytes))
            if result_image is not None:
                result_image = result_image.convert("RGB")
                result_image = np.array(result_image).astype(np.float32) / 255.0
                result_image = torch.from_numpy(result_image)[None,]
                batch_images.append(result_image)

        if type(batch_images).__name__ == "list" and len(batch_images) > 1:
            image_list = batch_images
            single_image_start = batch_images[0]
            batch_count = 0
            s = None
            for single_image in batch_images:
                if (batch_count + 1) < len(batch_images):
                    current_single_image = batch_images[batch_count + 1]
                    if single_image_start.shape[1:] != current_single_image.shape[1:]:
                        current_single_image = comfy.utils.common_upscale(current_single_image.movedim(-1, 1), single_image_start.shape[2], single_image_start.shape[1], "bilinear", "center").movedim(1, -1)
                    batch_count = batch_count + 1
                    if s is not None:
                        single_image = s
                    s = torch.cat((current_single_image, single_image), dim=0)
                    result_image = s
    else:
        image_base64 = api_result.data[0].b64_json
        image_bytes = base64.b64decode(image_base64)
        result_image = Image.open(BytesIO(image_bytes))
        if result_image is not None:
            result_image = result_image.convert("RGB")
            result_image = np.array(result_image).astype(np.float32) / 255.0
            result_image = torch.from_numpy(result_image)[None,]

    return result_image