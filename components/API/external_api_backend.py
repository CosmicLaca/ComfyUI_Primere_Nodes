from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any
from PIL import Image
from io import BytesIO
import numpy as np
import torch
import comfy.utils

PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")

class ExternalAPIError(RuntimeError):
    pass


@dataclass
class RenderResult:
    provider: str
    endpoint: str
    method: str
    headers: dict[str, Any]
    query: dict[str, Any]
    body: dict[str, Any] | list[Any] | None
    sdk_call: dict[str, Any] | None


def _replace_string_template(template: str, values: dict[str, Any]) -> Any:
    matches = list(PLACEHOLDER_RE.finditer(template))
    if not matches:
        return template
    if len(matches) == 1 and matches[0].span() == (0, len(template)):
        key = matches[0].group(1)
        if key not in values:
            raise ExternalAPIError(f"Missing value for placeholder '{key}'")
        return values[key]
    out = template
    for match in matches:
        key = match.group(1)
        if key not in values:
            raise ExternalAPIError(f"Missing value for placeholder '{key}'")
        out = out.replace(match.group(0), str(values[key]))
    return out


def render_template(template: Any, values: dict[str, Any]) -> Any:
    if isinstance(template, dict):
        return {k: render_template(v, values) for k, v in template.items()}
    if isinstance(template, list):
        return [render_template(v, values) for v in template]
    if isinstance(template, str):
        return _replace_string_template(template, values)
    return template


def list_placeholders(template: Any) -> list[str]:
    found: set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for v in node.values():
                walk(v)
            return
        if isinstance(node, list):
            for v in node:
                walk(v)
            return
        if isinstance(node, str):
            for m in PLACEHOLDER_RE.finditer(node):
                found.add(m.group(1))

    walk(template)
    return sorted(found)


def build_request(spec: dict[str, Any], values: dict[str, Any]) -> RenderResult:
    request = spec.get("request", {})
    return RenderResult(
        provider=spec.get("provider", "custom"),
        endpoint=render_template(request.get("endpoint", ""), values),
        method=render_template(request.get("method", "POST"), values),
        headers=render_template(request.get("headers", {}), values),
        query=render_template(request.get("query", {}), values),
        body=render_template(request.get("body"), values) if request.get("body") is not None else None,
        sdk_call=render_template(request.get("sdk_call"), values) if request.get("sdk_call") is not None else None,
    )


def normalize_sdk_call(sdk_call: dict[str, Any] | None) -> tuple[list[Any], dict[str, Any]]:
    if sdk_call is None:
        return [], {}
    if "args" in sdk_call or "kwargs" in sdk_call:
        return list(sdk_call.get("args", [])), dict(sdk_call.get("kwargs", {}))
    return [], dict(sdk_call)


def _resolve_dotted_from_context(path: str, context: dict[str, Any], allowed_roots: set[str]) -> Any:
    if not path:
        raise ExternalAPIError("Empty dotted path")
    parts = path.split(".")
    root = parts[0]
    if root not in allowed_roots:
        raise ExternalAPIError(f"Root '{root}' is not allowed")
    if root not in context:
        raise ExternalAPIError(f"Root '{root}' not found in context")
    value = context[root]
    for part in parts[1:]:
        if part.startswith("__"):
            raise ExternalAPIError("Dunder access is not allowed")
        value = getattr(value, part)
    return value


def _materialize_sdk_value(value: Any, context: dict[str, Any], allowed_roots: set[str]) -> Any:
    if isinstance(value, dict) and "$call" in value:
        fn = _resolve_dotted_from_context(str(value.get("$call", "")), context, allowed_roots)
        args = [_materialize_sdk_value(v, context, allowed_roots) for v in value.get("$args", [])]
        kwargs = {k: _materialize_sdk_value(v, context, allowed_roots) for k, v in value.get("$kwargs", {}).items()}
        return fn(*args, **kwargs)
    if isinstance(value, list):
        return [_materialize_sdk_value(v, context, allowed_roots) for v in value]
    if isinstance(value, dict):
        return {k: _materialize_sdk_value(v, context, allowed_roots) for k, v in value.items()}
    return value


def execute_sdk_request(rendered: RenderResult, context: dict[str, Any], allowed_roots: set[str] | None = None) -> Any:
    if rendered.method.upper() != "SDK":
        raise ExternalAPIError("execute_sdk_request expects SDK method")
    roots = allowed_roots or set(context.keys())
    fn = _resolve_dotted_from_context(str(rendered.endpoint), context, roots)
    args, kwargs = normalize_sdk_call(rendered.sdk_call)
    safe_args = [_materialize_sdk_value(a, context, roots) for a in args]
    safe_kwargs = {k: _materialize_sdk_value(v, context, roots) for k, v in kwargs.items()}
    return fn(*safe_args, **safe_kwargs)

def default_provider_service(node_data):
    if isinstance(node_data.API_SCHEMA_REGISTRY, dict) and len(node_data.API_SCHEMA_REGISTRY) > 0:
        first_provider = next(iter(node_data.API_SCHEMA_REGISTRY))
        provider_services = node_data.API_SCHEMA_REGISTRY.get(first_provider, {})
        if isinstance(provider_services, dict) and len(provider_services) > 0:
            first_service = next(iter(provider_services))
            return first_provider, first_service
        return first_provider, "default"

    providers = list(node_data.API_RESULT.keys()) if isinstance(node_data.API_RESULT, dict) else []
    if len(providers) > 0:
        return providers[0], "default"

    return "custom", "default"

def provider_list(node_data):
    default_provider, _ = default_provider_service(node_data)
    config_providers = []
    if isinstance(node_data.API_RESULT, dict):
        config_providers = [str(provider) for provider in node_data.API_RESULT.keys()]
    schema_provider_set = set()
    if isinstance(node_data.API_SCHEMA_REGISTRY, dict):
        schema_provider_set = {str(provider) for provider in node_data.API_SCHEMA_REGISTRY.keys()}

    common_providers = [provider for provider in config_providers if provider in schema_provider_set]

    if len(common_providers) == 0:
        return [default_provider]
    ordered_providers = []
    if default_provider in common_providers:
        ordered_providers.append(default_provider)
    for provider in common_providers:
        if provider not in ordered_providers:
            ordered_providers.append(provider)

    return ordered_providers

def service_list(node_data):
    default_provider, default_service = default_provider_service(node_data)
    services = []
    if isinstance(node_data.API_SCHEMA_REGISTRY, dict):
        provider_services = node_data.API_SCHEMA_REGISTRY.get(default_provider, {})
        if isinstance(provider_services, dict):
            services = [str(service) for service in provider_services.keys()]
    ordered_services = [default_service]
    for service in services:
        if service not in ordered_services:
            ordered_services.append(service)

    return ordered_services

def canonical_parameter_key(name: Any) -> str:
    text = str(name or "").strip().lower()
    # canonical matching: remove underscores (and other non-alnum separators), then compare.
    normalized = re.sub(r"[^a-z0-9]+", "", text)
    return normalized

def schema_possible_values(node_data, provider: str, service: str, parameter_name: str) -> list[Any]:
    registry = node_data.API_SCHEMA_REGISTRY if isinstance(node_data.API_SCHEMA_REGISTRY, dict) else {}
    provider_services = registry.get(str(provider), {}) if isinstance(registry, dict) else {}
    schema = provider_services.get(str(service), {}) if isinstance(provider_services, dict) else {}
    possible = schema.get("possible_parameters", {}) if isinstance(schema, dict) else {}
    if not isinstance(possible, dict):
        return []

    if parameter_name in possible and isinstance(possible.get(parameter_name), list):
        return list(possible.get(parameter_name) or [])

    expected = _canonical_parameter_key(parameter_name)
    for key, values in possible.items():
        if _canonical_parameter_key(key) == expected and isinstance(values, list):
            return list(values)

    return []

def parameter_options(node_data):
    default_provider, default_service = default_provider_service(node_data)
    provider_services = node_data.API_SCHEMA_REGISTRY.get(default_provider, {}) if isinstance(node_data.API_SCHEMA_REGISTRY, dict) else {}
    schema = provider_services.get(default_service, {}) if isinstance(provider_services, dict) else {}

    options: dict[str, list[str]] = {}

    possible = schema.get("possible_parameters", {}) if isinstance(schema, dict) else {}
    if not isinstance(possible, dict):
        return options
    for key, values in possible.items():
        key_name = str(key)
        if key_name == "prompt":
            continue
        value_list = [str(v) for v in values] if isinstance(values, list) else []
        if len(value_list) == 0:
            value_list = [f"default_{key_name}"]
        options[key_name] = value_list

    return options

def redact_reference_images(node):
    if isinstance(node, dict):
        sanitized = {}
        for key, value in node.items():
            if key == "reference_images":
                sanitized[key] = "[reference_images omitted]"
            else:
                sanitized[key] = redact_reference_images(value)
        return sanitized
    if isinstance(node, list):
        return [redact_reference_images(value) for value in node]
    return node

def parse_ratio(value):
    if not isinstance(value, str):
        return None
    cleaned = value.strip()
    if ":" not in cleaned:
        return None
    left, right = cleaned.split(":", 1)
    try:
        numerator = float(left)
        denominator = float(right)
    except ValueError:
        return None
    if denominator == 0:
        return None
    return numerator / denominator
def closest_valid_ratio(value, valid_ratios):
    if not isinstance(valid_ratios, list) or len(valid_ratios) == 0:
        return value

    normalized_valid = [str(ratio) for ratio in valid_ratios]
    candidate = str(value).strip() if value is not None else ""
    if candidate in normalized_valid:
        return candidate

    candidate_ratio = parse_ratio(candidate)
    if candidate_ratio is None:
        return normalized_valid[0]

    best_value = normalized_valid[0]
    best_diff = float("inf")
    for ratio_text in normalized_valid:
        parsed_ratio = parse_ratio(ratio_text)
        if parsed_ratio is None:
            continue
        diff = abs(parsed_ratio - candidate_ratio)
        if diff < best_diff:
            best_diff = diff
            best_value = ratio_text

    return best_value

def get_gemini_nanobanana(api_result):
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

def get_gemini_imagen(api_result):
    result_image = None
    if api_result.generated_images[0].image is not None and api_result.generated_images[0].image.image_bytes is not None:
        generated_image = api_result.generated_images[0].image.image_bytes
        result_image = Image.open(BytesIO(generated_image))
        if result_image is not None:
            result_image = result_image.convert("RGB")
            result_image = np.array(result_image).astype(np.float32) / 255.0
            result_image = torch.from_numpy(result_image)[None,]

    return result_image