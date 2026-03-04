from __future__ import annotations

from io import BytesIO
from typing import Iterable

import comfy.utils
import numpy as np
import requests
import torch
from PIL import Image

def pil_image_to_tensor(image: Image.Image | None) -> torch.Tensor | None:
    if image is None:
        return None

    rgb_image = image.convert("RGB")
    image_array = np.array(rgb_image).astype(np.float32) / 255.0
    return torch.from_numpy(image_array)[None,]


def bytes_to_tensor(image_bytes: bytes | bytearray | None) -> torch.Tensor | None:
    if not image_bytes:
        return None

    image = Image.open(BytesIO(image_bytes))
    return pil_image_to_tensor(image)


def fetch_url_bytes(url: str, timeout: int = 60) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def url_to_tensor(url: str, timeout: int = 60) -> torch.Tensor | None:
    return bytes_to_tensor(fetch_url_bytes(url, timeout=timeout))


def stack_image_tensors(images: Iterable[torch.Tensor]) -> torch.Tensor | None:
    prepared: list[torch.Tensor] = [img for img in images if isinstance(img, torch.Tensor)]
    if len(prepared) == 0:
        return None
    if len(prepared) == 1:
        return prepared[0]

    target = prepared[0]
    aligned: list[torch.Tensor] = [target]
    for image in prepared[1:]:
        if target.shape[1:] != image.shape[1:]:
            image = comfy.utils.common_upscale(
                image.movedim(-1, 1),
                target.shape[2],
                target.shape[1],
                "bilinear",
                "center",
            ).movedim(1, -1)
        aligned.append(image)

    return torch.cat(aligned, dim=0)
