from __future__ import annotations

import base64
import json
import time
from io import BytesIO
from typing import Any, Iterable
from urllib.parse import urlparse

import comfy.utils
import numpy as np
import requests
import torch
from PIL import Image

DEFAULT_REQUEST_TIMEOUT = 60

def load_json_object(payload: Any, error_prefix: str = "Invalid JSON response") -> dict[str, Any]:
    """Normalize payload into a JSON object dict."""
    try:
        if isinstance(payload, dict):
            return payload
        if isinstance(payload, str):
            parsed = json.loads(payload)
        else:
            parsed = json.loads(json.dumps(payload))
    except (TypeError, ValueError) as exc:
        raise RuntimeError(f"{error_prefix}: {payload}") from exc

    if not isinstance(parsed, dict):
        raise RuntimeError(f"{error_prefix}: expected JSON object, got {type(parsed).__name__}")
    return parsed


def image_bytes_to_tensor(image_bytes: bytes) -> torch.Tensor | None:
    if not image_bytes:
        return None

    image = Image.open(BytesIO(image_bytes))
    image = image.convert("RGB")
    image_np = np.array(image).astype(np.float32) / 255.0
    return torch.from_numpy(image_np)[None,]


def base64_to_tensor(image_base64: str | None) -> torch.Tensor | None:
    if not image_base64:
        return None
    return image_bytes_to_tensor(base64.b64decode(image_base64))


def download_image_bytes(url: str, timeout: int = DEFAULT_REQUEST_TIMEOUT) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def image_url_to_tensor(url: str, timeout: int = DEFAULT_REQUEST_TIMEOUT) -> torch.Tensor | None:
    if not url:
        return None
    return image_bytes_to_tensor(download_image_bytes(url, timeout=timeout))


def image_urls_to_tensors(urls: Iterable[str], timeout: int = DEFAULT_REQUEST_TIMEOUT) -> list[torch.Tensor]:
    tensors: list[torch.Tensor] = []
    for url in urls:
        tensor = image_url_to_tensor(url, timeout=timeout)
        if tensor is not None:
            tensors.append(tensor)
    return tensors


def merge_image_tensors(tensors: list[torch.Tensor]) -> torch.Tensor | None:
    if not tensors:
        return None
    if len(tensors) == 1:
        return tensors[0]

    base = tensors[0]
    normalized = [base]
    for tensor in tensors[1:]:
        current = tensor
        if base.shape[1:] != current.shape[1:]:
            current = comfy.utils.common_upscale(
                current.movedim(-1, 1),
                base.shape[2],
                base.shape[1],
                "bilinear",
                "center",
            ).movedim(1, -1)
        normalized.append(current)

    return torch.cat(normalized, dim=0)


def build_result_endpoint_from_response_url(response_url: str) -> str:
    parsed_url = urlparse(response_url)
    region = parsed_url.netloc or None
    path_parts = [part for part in parsed_url.path.split("/") if part]
    version = path_parts[0] if path_parts else None
    if not region or not version:
        raise RuntimeError(f"Invalid response_url: {response_url}")
    return f"https://{region}/{version}/get_result"


def poll_result_endpoint(
    polling_url: str,
    response_url: str,
    accepted_statuses: set[str] | None = None,
    max_attempts: int = 20,
    max_error_retries: int = 1,
    sleep_seconds: float = 2.0,
) -> dict[str, Any]:
    accepted = accepted_statuses or {"Ready", "Pending", "Task not found"}

    polling_response = requests.get(polling_url, timeout=DEFAULT_REQUEST_TIMEOUT)
    polling_response.raise_for_status()
    result_payload = load_json_object(polling_response.text, "Invalid polling response")

    result_endpoint = build_result_endpoint_from_response_url(response_url)
    status = "Start"
    error_tryout = 0

    while error_tryout <= max_error_retries:
        request_tryout = 0
        while status != "Ready" and request_tryout <= max_attempts:
            response = requests.request(
                "GET",
                result_endpoint,
                params={"id": result_payload.get("id")},
                timeout=DEFAULT_REQUEST_TIMEOUT,
            )
            response.raise_for_status()
            result_payload = load_json_object(response.text, "Invalid result endpoint response")
            status = result_payload.get("status")

            if status not in accepted:
                error_response = requests.get(polling_url, timeout=DEFAULT_REQUEST_TIMEOUT)
                error_response.raise_for_status()
                error_payload = load_json_object(error_response.text, "Invalid error polling response")
                error_status = error_payload.get("status")
                raise RuntimeError(f"Response status: {status}, error status: {error_status}")

            if status == "Ready":
                return result_payload

            time.sleep(sleep_seconds)
            request_tryout += 1

        if status == "Ready":
            return result_payload

        error_tryout += 1
        time.sleep(1)

    raise RuntimeError("Polling timed out before status became Ready")
