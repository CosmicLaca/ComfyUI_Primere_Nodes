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
import inspect
import dataclasses

from PIL import Image
from io import BytesIO
import numpy as np
import torch
import comfy.utils
import types

from . import request_exceptions

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
    request_exclusions: list[dict[str, Any]] | None


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
    found_order: list[str] = []
    found_set: set[str] = set()
    def add_placeholder(name: str) -> None:
        if name in found_set:
            return
        found_set.add(name)
        found_order.append(name)

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
                add_placeholder(m.group(1))

    walk(template)
    return found_order


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
        request_exclusions=spec.get("request_exclusions") if isinstance(spec.get("request_exclusions"), list) else []
    )

def _prepare_used_value_exclusions(exclusions: Any) -> list[dict[str, Any]]:
    if not isinstance(exclusions, list):
        return []

    prepared: list[dict[str, Any]] = []
    for rule in exclusions:
        if not isinstance(rule, dict):
            continue

        normalized_rule = dict(rule)
        condition = normalized_rule.get("when") if isinstance(normalized_rule.get("when"), dict) else normalized_rule.get("if")
        if isinstance(condition, dict):
            normalized_condition = dict(condition)
            path = normalized_condition.get("path") or normalized_condition.get("key")
            if isinstance(path, str):
                leaf = [part for part in path.split(".") if part]
                if len(leaf) > 0:
                    normalized_condition["path"] = leaf[-1]
            if isinstance(normalized_rule.get("when"), dict):
                normalized_rule["when"] = normalized_condition
            else:
                normalized_rule["if"] = normalized_condition

        remove_spec = normalized_rule.get("remove")
        normalized_remove: list[str] = []
        if isinstance(remove_spec, str):
            remove_spec = [remove_spec]
        if isinstance(remove_spec, list):
            for path in remove_spec:
                if not isinstance(path, str):
                    continue
                leaf = [part for part in path.split(".") if part]
                if len(leaf) > 0:
                    normalized_remove.append(leaf[-1])

        normalized_rule["remove"] = normalized_remove
        prepared.append(normalized_rule)

    return prepared


def remove_excluded_used_values(used_values: dict[str, Any], exclusions: Any) -> dict[str, Any]:
    filtered = dict(used_values) if isinstance(used_values, dict) else {}
    prepared_exclusions = _prepare_used_value_exclusions(exclusions)

    return request_exceptions.apply_exclusions_to_payload(
        filtered,
        prepared_exclusions,
        use_kwargs_fallback=False,
        canonicalize_key=canonical_param_name,
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

def _apply_auth_header_fallback(kwargs: dict[str, Any], context: dict[str, Any]) -> None:
    headers = kwargs.get("headers") if isinstance(kwargs, dict) else None
    if not isinstance(headers, dict):
        return

    provider_api_key = context.get("provider_api_key")
    if provider_api_key in (None, ""):
        return

    for auth_key in ("x-key", "x_api_key", "api-key", "authorization", "Authorization"):
        if auth_key not in headers:
            continue
        if headers.get(auth_key) in (None, "", "null"):
            headers[auth_key] = provider_api_key

def execute_sdk_request(rendered: RenderResult, context: dict[str, Any], allowed_roots: set[str] | None = None, match_context: dict[str, Any] | None = None) -> Any:
    if rendered.method.upper() != "SDK":
        raise ExternalAPIError("execute_sdk_request expects SDK method")
    roots = allowed_roots or set(context.keys())
    fn = _resolve_dotted_from_context(str(rendered.endpoint), context, roots)
    args, kwargs = normalize_sdk_call(rendered.sdk_call)
    filtered_args, filtered_kwargs = request_exceptions.apply_sdk_request_exclusions(args=list(args), kwargs=dict(kwargs), exclusions=rendered.request_exclusions, match_context=match_context)
    safe_args = [_materialize_sdk_value(a, context, roots) for a in filtered_args]
    safe_kwargs = {k: _materialize_sdk_value(v, context, roots) for k, v in filtered_kwargs.items()}
    _apply_auth_header_fallback(safe_kwargs, context)
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
            key_name = str(key or "").lower()
            if key_name == "reference_images" or key_name == "input_image" or key_name.startswith("input_image_"):
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

    # module_name = f"{safe_name[:-3].replace('.', '_').replace('-', '_')}"
    package_name = "primere_response_handlers"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [base_dir]
        sys.modules[package_name] = package

    module_stem = safe_name[:-3].replace('.', '_').replace('-', '_')
    module_name = f"{package_name}.{module_stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ExternalAPIError(f"Cannot import response handler: {safe_name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    handler = getattr(module, "handle_response", None)
    if not callable(handler):
        raise ExternalAPIError(f"Response handler '{safe_name}' must define callable handle_response(api_result, schema)")

    return handler

def _load_reference_images_handler(filename: str):
    safe_name = _safe_response_handler_filename(filename)
    base_dir = os.path.join(PRIMERE_ROOT, 'components', 'API', 'references')
    module_path = os.path.join(base_dir, safe_name)
    if not Path(module_path).exists():
        return None

    package_name = "primere_reference_handlers"
    package = sys.modules.get(package_name)
    if package is None:
        package = types.ModuleType(package_name)
        package.__path__ = [base_dir]
        sys.modules[package_name] = package

    module_stem = safe_name[:-3].replace('.', '_').replace('-', '_')
    module_name = f"{package_name}.{module_stem}"
    spec = importlib.util.spec_from_file_location(module_name, str(module_path))
    if spec is None or spec.loader is None:
        raise ExternalAPIError(f"Cannot import reference images handler: {safe_name}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    handler = getattr(module, "handle_reference_images", None)
    if not callable(handler):
        raise ExternalAPIError(f"Reference images handler '{safe_name}' must define callable handle_reference_images(**kwargs)")

    return handler


def apply_reference_images_handler(schema: dict[str, Any] | None, provider: str, handler_context: dict[str, Any] | None = None) -> Any:
    safe_provider = str(provider or "").strip()
    configured_handler = schema.get("reference_images_handler") if isinstance(schema, dict) else None
    handler_file = str(configured_handler).strip() if configured_handler not in (None, "") else f"{safe_provider}.py"
    handler = _load_reference_images_handler(handler_file)
    if handler is None:
        handler = _load_reference_images_handler("default.py")
    if handler is None:
        raise ExternalAPIError("Reference images handler file not found: default.py")

    context = handler_context if isinstance(handler_context, dict) else {}
    return handler(**context)

def apply_response_handler(schema: dict[str, Any] | None, api_result: Any, provider: str = "", service: str = "", response_context: dict[str, Any] | None = None) -> Any:
    if api_result is None:
        return None

    configured_handler = schema.get("response_handler") if isinstance(schema, dict) else None
    handler_file = str(configured_handler).strip() if configured_handler not in (None, "") else f'{provider}_{service}.py'
    handler = _load_response_handler(handler_file)
    safe_schema = schema if isinstance(schema, dict) else {}
    context = response_context if isinstance(response_context, dict) else {}

    try:
        signature = inspect.signature(handler)
        accepted = set(signature.parameters.keys())
    except (TypeError, ValueError):
        accepted = set()

    kwargs: dict[str, Any] = {}
    if "schema" in accepted:
        kwargs["schema"] = safe_schema

    for key, value in context.items():
        if key in accepted:
            kwargs[key] = value

    return handler(api_result, **kwargs)

def sanitize_debug_value(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return f"[torch.Tensor omitted: shape={tuple(value.shape)}, dtype={value.dtype}]"

    if isinstance(value, dict):
        sanitized_dict = {}
        for k, v in value.items():
            key_name = str(k or "").lower()
            if key_name == "b64_json":
                encoded_size = len(v) if isinstance(v, (str, bytes, bytearray, memoryview)) else 0
                sanitized_dict[k] = f"[base64 omitted: {encoded_size} chars]"
                continue
            if key_name in {"reference_images", "input_image"} or key_name.startswith("input_image_"):
                if isinstance(v, str):
                    sanitized_dict[k] = f"[image payload omitted: {len(v)} chars]"
                elif isinstance(v, list):
                    sanitized_dict[k] = f"[image payload list omitted: {len(v)} items]"
                elif isinstance(v, tuple):
                    sanitized_dict[k] = f"[image payload tuple omitted: {len(v)} items]"
                elif isinstance(v, dict):
                    sanitized_dict[k] = f"[image payload object omitted: {len(v)} keys]"
                else:
                    sanitized_dict[k] = "[image payload omitted]"
                continue
            sanitized_dict[k] = sanitize_debug_value(v)
        return sanitized_dict
    if isinstance(value, list):
        tensor_count = sum(1 for item in value if isinstance(item, torch.Tensor))
        if tensor_count == len(value) and tensor_count > 0:
            return f"[tensor list omitted: {tensor_count} tensors]"
        return [sanitize_debug_value(v) for v in value]
    if isinstance(value, tuple):
        tensor_count = sum(1 for item in value if isinstance(item, torch.Tensor))
        if tensor_count == len(value) and tensor_count > 0:
            return f"[tensor tuple omitted: {tensor_count} tensors]"
        return tuple(sanitize_debug_value(v) for v in value)
    if isinstance(value, (bytes, bytearray, memoryview)):
        return f"[binary data omitted: {len(value)} bytes]"
    if isinstance(value, Image.Image):
        return f"[PIL.Image omitted: mode={value.mode}, size={value.size}]"
    if isinstance(value, np.ndarray):
        return f"[numpy.ndarray omitted: shape={value.shape}, dtype={value.dtype}]"

    if dataclasses.is_dataclass(value):
        return {
            "_type": value.__class__.__name__,
            **{field.name: sanitize_debug_value(getattr(value, field.name)) for field in dataclasses.fields(value)},
        }

    if hasattr(value, "__dict__") and not isinstance(value, (str, int, float, bool)):
        safe_fields = {}
        for key, field_value in vars(value).items():
            if key.startswith("_"):
                continue
            key_name = str(key or "").lower()
            if key_name == "b64_json":
                encoded_size = len(field_value) if isinstance(field_value, (str, bytes, bytearray, memoryview)) else 0
                safe_fields[key] = f"[base64 omitted: {encoded_size} chars]"
                continue
            safe_fields[key] = sanitize_debug_value(field_value)
        if len(safe_fields) > 0:
            return {"_type": value.__class__.__name__, **safe_fields}

    return value

def sanitize_api_debug_payload(value: Any) -> Any:
    return redact_reference_images(sanitize_debug_value(value))
def canonical_param_name(name: str, *, number_of_images_as_seed: bool = False) -> str:
    low = str(name or "").lower()
    if "aspect_ratio" in low:
        return "aspect_ratio"
    if "resolution" in low or "image_size" in low:
        return "resolution"
    if low == "model" or low.endswith("_model"):
        return "model"
    if number_of_images_as_seed and low == "number_of_images":
        return "seed"
    if low in {"prompt", "contents"} or low.endswith("_prompt"):
        return "prompt"
    if "response_modalities" in low:
        return "response_modalities"
    return str(name)