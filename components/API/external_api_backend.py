from __future__ import annotations
from ...components.tree import PRIMERE_ROOT

import re
import sys
import importlib
import os
import json
from dataclasses import dataclass
from typing import Any
from pathlib import Path
import importlib.util

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

def _collect_sdk_roots(node: Any, roots: set[str]) -> None:
    if isinstance(node, dict):
        call_path = node.get("$call")
        if isinstance(call_path, str) and "." in call_path:
            roots.add(call_path.split(".", 1)[0])
        for value in node.values():
            _collect_sdk_roots(value, roots)
        return
    if isinstance(node, list):
        for value in node:
            _collect_sdk_roots(value, roots)


def _resolve_context_root(root_name: str, provider_client: Any) -> Any:
    if root_name == "client":
        return provider_client
    if root_name in sys.modules:
        return sys.modules[root_name]

    module_candidates = [root_name]
    client_module = getattr(getattr(provider_client, "__class__", None), "__module__", "")
    module_parts = [part for part in str(client_module).split(".") if part]
    for i in range(len(module_parts), 0, -1):
        module_candidates.append(".".join(module_parts[:i] + [root_name]))

    tried = set()
    for module_name in module_candidates:
        if module_name in tried:
            continue
        tried.add(module_name)
        try:
            return importlib.import_module(module_name)
        except Exception:
            continue
    raise ImportError(f"Unable to resolve SDK context root '{root_name}'")


def build_sdk_context(rendered: RenderResult, client: Any) -> tuple[dict[str, Any], set[str]]:
    context: dict[str, Any] = {"client": client}
    allowed_roots: set[str] = {"client"}

    required_roots = set()
    if isinstance(rendered.endpoint, str) and "." in rendered.endpoint:
        required_roots.add(rendered.endpoint.split(".", 1)[0])
    _collect_sdk_roots(rendered.sdk_call, required_roots)

    for root_name in sorted(required_roots):
        if root_name in context:
            continue
        context[root_name] = _resolve_context_root(root_name, client)
        allowed_roots.add(root_name)

    return context, allowed_roots

def load_import_modules(import_modules: list[str] | None) -> tuple[dict[str, Any], set[str]]:
    """Load schema-defined imports into SDK execution context."""
    context: dict[str, Any] = {}
    allowed_roots: set[str] = set()

    for import_line in import_modules or []:
        if not isinstance(import_line, str):
            continue
        line = import_line.strip()
        if not line:
            continue

        if line.startswith("import "):
            module_specs = [part.strip() for part in line[len("import "):].split(",") if part.strip()]
            for spec in module_specs:
                if " as " in spec:
                    module_name, alias = [part.strip() for part in spec.split(" as ", 1)]
                else:
                    module_name = spec
                    alias = module_name.split(".")[-1]
                module_obj = importlib.import_module(module_name)
                context[alias] = module_obj
                allowed_roots.add(alias)
            continue

        if line.startswith("from ") and " import " in line:
            module_name, imported = line[len("from "):].split(" import ", 1)
            module_name = module_name.strip()
            module_obj = importlib.import_module(module_name)
            symbol_specs = [part.strip() for part in imported.split(",") if part.strip()]
            for spec in symbol_specs:
                if " as " in spec:
                    symbol_name, alias = [part.strip() for part in spec.split(" as ", 1)]
                else:
                    symbol_name = spec
                    alias = symbol_name
                context[alias] = getattr(module_obj, symbol_name)
                allowed_roots.add(alias)
            continue

        raise ExternalAPIError(f"Unsupported import syntax in schema import_modules: {import_line}")

    return context, allowed_roots

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
    ordered_services = [default_service]
    registry = node_data.API_SCHEMA_REGISTRY if isinstance(node_data.API_SCHEMA_REGISTRY, dict) else {}
    for provider_services in registry.values():
        if not isinstance(provider_services, dict):
            continue
        for service in provider_services.keys():
            service_name = str(service)
            if service_name not in ordered_services:
                ordered_services.append(service_name)

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

    expected = canonical_parameter_key(parameter_name)
    for key, values in possible.items():
        if canonical_parameter_key(key) == expected and isinstance(values, list):
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

def _parse_ratio_parts(value: Any) -> tuple[float, float] | None:
    text = str(value).strip() if value is not None else ""
    if ":" not in text:
        return None

    left, right = text.split(":", 1)
    try:
        a = float(left.strip())
        b = float(right.strip())
    except ValueError:
        return None

    if a <= 0 or b <= 0:
        return None
    return a, b


def _ratio_orientation(a: float, b: float) -> str:
    if a > b:
        return "horizontal"
    if a < b:
        return "vertical"
    return "square"

def closest_valid_ratio(value, valid_ratios):
    if not isinstance(valid_ratios, (list, tuple)) or len(valid_ratios) == 0:
        return value

    candidate = str(value).strip() if value is not None else ""
    normalized_valid = [str(ratio).strip() for ratio in valid_ratios if str(ratio).strip()]
    if len(normalized_valid) == 0:
        return value
    if candidate in normalized_valid:
        return candidate

    candidate_parts = _parse_ratio_parts(candidate)
    if candidate_parts is None:
        return normalized_valid[0]

    input_a, input_b = candidate_parts
    input_product = input_a * input_b
    input_orientation = _ratio_orientation(input_a, input_b)

    same_orientation_matches: list[tuple[str, float]] = []
    fallback_matches: list[tuple[str, float]] = []

    for ratio_text in normalized_valid:
        ratio_parts = _parse_ratio_parts(ratio_text)
        if ratio_parts is None:
            continue

        valid_a, valid_b = ratio_parts
        valid_product = valid_a * valid_b
        product_diff = abs(valid_product - input_product)
        valid_orientation = _ratio_orientation(valid_a, valid_b)

        fallback_matches.append((ratio_text, product_diff))
        if valid_orientation == input_orientation:
            same_orientation_matches.append((ratio_text, product_diff))

    pool = same_orientation_matches if len(same_orientation_matches) > 0 else fallback_matches
    if len(pool) == 0:
        return normalized_valid[0]

    pool.sort(key=lambda item: item[1])
    return pool[0][0]

def _safe_response_handler_filename(name: str) -> str:
    filename = str(name or "").strip()
    if not filename:
        return ""
    if filename.startswith("/") or ".." in filename or "/" in filename or "\\" in filename:
        raise ExternalAPIError(f"Invalid response handler filename: {filename}")
    if not filename.endswith(".py"):
        raise ExternalAPIError(f"Response handler must be a .py file: {filename}")
    return filename


def _load_response_handler(filename: str):
    safe_name = _safe_response_handler_filename(filename)
    base_dir = os.path.join(PRIMERE_ROOT, 'components', 'API', 'responses')
    module_path = os.path.join(base_dir, safe_name)
    if not Path(module_path).exists():
        raise ExternalAPIError(f"Response handler file not found: {safe_name}")

    module_name = f"{safe_name[:-3].replace('.', '_').replace('-', '_')}"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ExternalAPIError(f"Cannot import response handler: {safe_name}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    handler = getattr(module, "handle_response", None)
    if not callable(handler):
        raise ExternalAPIError(f"Response handler '{safe_name}' must define callable handle_response(api_result, schema)")

    return handler

def apply_response_handler(schema: dict[str, Any] | None, api_result: Any, provider: str = "", service: str = "") -> Any:
    if api_result is None:
        return None

    configured_handler = schema.get("response_handler") if isinstance(schema, dict) else None
    handler_file = str(configured_handler).strip() if configured_handler not in (None, "") else f'{provider}_{service}.py'
    handler = _load_response_handler(handler_file)
    return handler(api_result)