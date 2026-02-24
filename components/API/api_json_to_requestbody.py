from __future__ import annotations
from typing import Any
from . import external_api_backend

def make_fake_values(spec: dict[str, Any]) -> dict[str, Any]:
    request = spec.get("request", {})
    placeholders = external_api_backend.list_placeholders(
        {
            "endpoint": request.get("endpoint", ""),
            "method": request.get("method", "POST"),
            "headers": request.get("headers", {}),
            "query": request.get("query", {}),
            "body": request.get("body"),
            "sdk_call": request.get("sdk_call"),
        }
    )

    def fake(name: str) -> Any:
        k = name.lower()
        if "prompt" in k:
            return "cute cat walking in the street of futuristic metropolis"
        if "model" in k:
            return "gemini-3-pro-image-preview"
        if "aspect_ratio" in k:
            return "1:1"
        if "resolution" in k or "image_size" in k:
            return "1K"
        if "number" in k or "count" in k or "width" in k or "height" in k:
            return 1
        if "response_modalities_0" in k:
            return "IMAGE"
        if "response_modalities_1" in k:
            return "TEXT"
        return f"fake_{name}"

    return {k: fake(k) for k in placeholders}


def render_from_schema(spec: dict[str, Any], values: dict[str, Any] | None = None):
    used_values = values or make_fake_values(spec)
    return external_api_backend.build_request(spec, used_values), used_values
