from __future__ import annotations
from typing import Any
from . import external_api_backend

def _canonical_param_name(name: str) -> str:
    low = name.lower()
    if "aspect_ratio" in low:
        return "aspect_ratio"
    if "resolution" in low or "image_size" in low:
        return "resolution"
    if low == "model" or low.endswith("_model"):
        return "model"
    if "prompt" in low or "contents" in low:
        return "prompt"
    if "response_modalities" in low:
        return "response_modalities"
    return name

def _default_value(name: str) -> Any:
    k = name.lower()
    if "prompt" in k:
        return "default prompt"
    if "response_modalities" in k:
        return "IMAGE"
    if "model" in k:
        return "gemini-3-pro-image-preview"
    if "aspect_ratio" in k:
        return "1:1"
    if "resolution" in k:
        return "1K"
    if "number" in k or "count" in k or "width" in k or "height" in k:
        return 1
    return f"default_{name}"

def _build_values(spec: dict[str, Any], values: dict[str, Any] | None = None) -> dict[str, Any]:
    user_values = values or {}
    request = spec.get("request", {})
    possible_parameters = spec.get("possible_parameters", {}) if isinstance(spec.get("possible_parameters"), dict) else {}

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

    resolved: dict[str, Any] = {}
    for key in placeholders:
        canonical = _canonical_param_name(key)
        selected = None

        if key in user_values and user_values[key] not in (None, ""):
            selected = user_values[key]
        elif canonical in user_values and user_values[canonical] not in (None, ""):
            selected = user_values[canonical]
        elif key in possible_parameters and isinstance(possible_parameters[key], list) and len(possible_parameters[key]) > 0:
            selected = possible_parameters[key][0]
        elif canonical in possible_parameters and isinstance(possible_parameters[canonical], list) and len(possible_parameters[canonical]) > 0:
            selected = possible_parameters[canonical][0]
        else:
            selected = _default_value(canonical)

        resolved[key] = selected

    return resolved

def render_from_schema(spec: dict[str, Any], values: dict[str, Any] | None = None):
    used_values = _build_values(spec, values)
    return external_api_backend.build_request(spec, used_values), used_values