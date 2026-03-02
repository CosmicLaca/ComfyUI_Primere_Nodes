from __future__ import annotations
from typing import Any
import os
from . import external_api_backend
from . import api_helper

def _canonical_param_name(name: str) -> str:
    low = name.lower()
    if "aspect_ratio" in low:
        return "aspect_ratio"
    if "resolution" in low or "image_size" in low:
        return "resolution"
    if low == "model" or low.endswith("_model"):
        return "model"
    if low in {"prompt", "contents"} or low.endswith("_prompt"):
        return "prompt"
    if "response_modalities" in low:
        return "response_modalities"
    return name

def _provider_api_key(spec: dict[str, Any]) -> Any:
    provider = str(spec.get("provider") or "").strip()
    if not provider:
        return None
    try:
        config = api_helper.get_api_config("apiconfig.json")
    except Exception:
        return None
    entry = config.get(provider, {}) if isinstance(config, dict) else {}
    if not isinstance(entry, dict):
        return None
    value = entry.get("APIKEY")
    return value if value not in (None, "") else None


def _secret_placeholder_default(key: str, spec: dict[str, Any]) -> Any:
    normalized = str(key or "").strip()
    low = normalized.lower()

    # If placeholder itself looks like an env-var token, keep token name.
    # This preserves schemas that intentionally use:
    # {"$call": "os.environ.get", "$args": ["{{BFL_API_KEY}}"]}
    # so runtime executes os.environ.get("BFL_API_KEY") correctly.
    if normalized.upper() == normalized and "_" in normalized:
        if low.endswith(("_key", "_token", "_secret", "_password")):
            return normalized

    direct_env = os.environ.get(normalized) or os.environ.get(normalized.upper())
    if direct_env not in (None, ""):
        return direct_env

    if low.endswith("_api_key") or low in {"api_key", "apikey", "x_api_key", "provider_api_key", "authorization", "auth_token", "access_token", "bearer_token", "token"}:
        return _provider_api_key(spec)

    return None

def _marker_default(marker: Any) -> Any:
    if not isinstance(marker, str):
        return None
    value = marker.strip().upper()
    if value == "INT":
        return 1
    if value == "FLOAT":
        return 1.0
    if value == "STRING":
        return ""
    if value == "BOOLEAN":
        return False
    return None

def _default_value(name: str) -> Any:
    k = name.lower()
    if "prompt" in k:
        return "cute cat walking in the futuristic metropolis"
    if "response_modalities" in k:
        return "IMAGE"
    if "model" in k:
        return "valid-model-name"
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
            secret_selected = _secret_placeholder_default(key, spec)
            if secret_selected is not None:
                selected = secret_selected
            else:
                marker_selected = _marker_default(possible_parameters.get(key)) if key in possible_parameters else None
                if marker_selected is None and canonical in possible_parameters:
                    marker_selected = _marker_default(possible_parameters.get(canonical))
                selected = marker_selected if marker_selected is not None else _default_value(canonical)

        resolved[key] = selected

    return resolved

def render_from_schema(spec: dict[str, Any], values: dict[str, Any] | None = None):
    used_values = _build_values(spec, values)
    return external_api_backend.build_request(spec, used_values), used_values