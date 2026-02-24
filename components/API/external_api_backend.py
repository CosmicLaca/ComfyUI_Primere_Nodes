from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

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
