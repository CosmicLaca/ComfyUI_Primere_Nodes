from __future__ import annotations

from io import BytesIO
from typing import Any

import comfy.utils
import numpy as np
import torch
from PIL import Image

def handle_response(api_result: Any):
    result_image = None
    image_list = []
    final_batch_img = []
    if api_result.candidates[0].content is not None and api_result.candidates[0].content.parts is not None:
        image_parts = [
            part.inline_data.data
            for part in api_result.candidates[0].content.parts
            if part.inline_data
        ]
        if image_parts:
            result_image = Image.open(BytesIO(image_parts[0]))
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
