from __future__ import annotations

from io import BytesIO
from typing import Any

import numpy as np
import torch
from PIL import Image

def handle_response(api_result, schema=None, loaded_client=None, response_url=None):
    result_image = None
    if api_result.generated_images[0].image is not None and api_result.generated_images[0].image.image_bytes is not None:
        generated_image = api_result.generated_images[0].image.image_bytes
        result_image = Image.open(BytesIO(generated_image))
        if result_image is not None:
            result_image = result_image.convert("RGB")
            result_image = np.array(result_image).astype(np.float32) / 255.0
            result_image = torch.from_numpy(result_image)[None,]

    return result_image
