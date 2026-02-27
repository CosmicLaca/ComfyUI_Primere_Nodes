"""Convert `snippet.py` into grouped provider->service API schema JSON."""
# python api_snippet_to_json.py --provider Gemini --service Imagen --replace


from __future__ import annotations

import argparse
import ast
import json
import re
from pathlib import Path
from typing import Any


class SnippetParseError(RuntimeError):
    """Raised when call extraction fails."""


SNIPPET_FILENAME = "snippet.py"
RESULT_FILENAME = "result.json"
DEFAULT_PROVIDER = ""
DEFAULT_SERVICE = ""
PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")

KNOWN_PARAM_OPTIONS: dict[str, list[str]] = {
    "model": ["example-model-1", "example-model-2"],
    "resolution": ["1K", "2K", "4K"],
}

EXCLUDED_PARAMETER_KEYS = {"prompt", "response_modalities", "aspect_ratio", "width", "height", "seed", "reference_images", "first_image", "last_image", "negative_prompt"}


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        return f"{base}.{node.attr}"
    return ast.unparse(node)


def placeholder(name: str) -> str:
    clean = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    while "__" in clean:
        clean = clean.replace("__", "_")
    return f"{{{{{clean.strip('_') or 'value'}}}}}"


def node_to_template(node: ast.AST, path: str) -> Any:
    if isinstance(node, ast.Constant):
        return placeholder(path)

    if isinstance(node, ast.Name):
        return placeholder(node.id)

    if isinstance(node, ast.Dict):
        out: dict[str, Any] = {}
        for key_node, value_node in zip(node.keys, node.values):
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                key = key_node.value
            else:
                key = ast.unparse(key_node)
            out[key] = node_to_template(value_node, f"{path}_{key}")
        return out

    if isinstance(node, ast.List):
        return [node_to_template(item, f"{path}_{idx}") for idx, item in enumerate(node.elts)]

    if isinstance(node, ast.Tuple):
        return [node_to_template(item, f"{path}_{idx}") for idx, item in enumerate(node.elts)]

    if isinstance(node, ast.Call):
        call_name = dotted_name(node.func)
        args = [node_to_template(arg, f"{path}_arg{idx}") for idx, arg in enumerate(node.args)]
        kwargs = {
            kw.arg if kw.arg else f"kw_{idx}": node_to_template(kw.value, f"{path}_{kw.arg if kw.arg else f'kw_{idx}'}")
            for idx, kw in enumerate(node.keywords)
        }
        return {
            "$call": call_name,
            "$args": args,
            "$kwargs": kwargs,
        }

    return placeholder(path)


def _last_call_in_node(node: ast.AST) -> ast.Call | None:
    found: ast.Call | None = None
    for child in ast.walk(node):
        if isinstance(child, ast.Call):
            found = child
    return found


def find_first_call(tree: ast.AST) -> ast.Call:
    if isinstance(tree, ast.Module) and tree.body:
        for stmt in reversed(tree.body):
            if isinstance(stmt, ast.Assign) and isinstance(stmt.value, ast.Call):
                return stmt.value
            if isinstance(stmt, ast.AnnAssign) and isinstance(stmt.value, ast.Call):
                return stmt.value
            if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                return stmt.value
            found = _last_call_in_node(stmt)
            if found is not None:
                return found
    found_any = _last_call_in_node(tree)
    if found_any is not None:
        return found_any
    raise SnippetParseError("No function call found in snippet.")


def _collect_placeholders(node: Any) -> set[str]:
    found: set[str] = set()

    def walk(item: Any) -> None:
        if isinstance(item, dict):
            for v in item.values():
                walk(v)
            return
        if isinstance(item, list):
            for v in item:
                walk(v)
            return
        if isinstance(item, str):
            for m in PLACEHOLDER_RE.finditer(item):
                found.add(m.group(1))

    walk(node)
    return found


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


def build_possible_parameters(request_schema: dict[str, Any]) -> dict[str, list[str]]:
    placeholders = sorted(_collect_placeholders(request_schema))
    possible: dict[str, list[str]] = {}

    for name in placeholders:
        canonical = _canonical_param_name(name)
        if canonical in EXCLUDED_PARAMETER_KEYS:
            continue
        if canonical in KNOWN_PARAM_OPTIONS:
            possible[canonical] = KNOWN_PARAM_OPTIONS[canonical]
        elif canonical not in possible:
            possible[canonical] = [f"fake_{canonical}_value_1", f"fake_{canonical}_value_2"]

    return possible


def build_service_schema(snippet: str, provider: str = DEFAULT_PROVIDER, service: str = DEFAULT_SERVICE) -> dict[str, Any]:
    tree = ast.parse(snippet)
    call = find_first_call(tree)

    endpoint = dotted_name(call.func)
    args_template = [node_to_template(arg, f"arg{idx}") for idx, arg in enumerate(call.args)]
    kwargs_template = {
        kw.arg if kw.arg else f"kw_{idx}": node_to_template(kw.value, kw.arg if kw.arg else f"kw_{idx}")
        for idx, kw in enumerate(call.keywords)
    }

    request_schema = {
        "method": "SDK",
        "endpoint": endpoint,
        "sdk_call": {
            "args": args_template,
            "kwargs": kwargs_template,
        },
    }

    return {
        "provider": provider,
        "service": service,
        "possible_parameters": build_possible_parameters(request_schema),
        "request": request_schema,
    }


def _ensure_mapping(node: Any) -> dict[str, Any]:
    return node if isinstance(node, dict) else {}


def upsert_service_schema(registry: dict[str, Any], service_schema: dict[str, Any]) -> dict[str, Any]:
    provider = str(service_schema.get("provider") or "").strip()
    service = str(service_schema.get("service") or "").strip()

    if not provider:
        raise SnippetParseError("Provider is required. Use --provider (e.g. --provider Gemini).")
    if not service:
        raise SnippetParseError("Service is required. Use --service (e.g. --service text2image).")

    out = _ensure_mapping(registry).copy()
    provider_map = _ensure_mapping(out.get(provider)).copy()
    provider_map[service] = service_schema
    out[provider] = provider_map
    return out


def convert_default_files(
    base_dir: Path | None = None,
    provider: str = DEFAULT_PROVIDER,
    service: str = DEFAULT_SERVICE,
    append: bool = True,
) -> Path:
    root = base_dir or Path.cwd()
    snippet_path = root / SNIPPET_FILENAME
    if not snippet_path.exists():
        raise SnippetParseError(f"Missing {SNIPPET_FILENAME} in {root}")

    snippet = snippet_path.read_text(encoding="utf-8")
    service_schema = build_service_schema(snippet, provider=provider, service=service)

    result_path = root / RESULT_FILENAME
    if append and result_path.exists():
        existing = json.loads(result_path.read_text(encoding="utf-8"))
    else:
        existing = {}

    output = upsert_service_schema(existing, service_schema)
    encoded = json.dumps(output, ensure_ascii=False, indent=2)

    result_path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="Provider name (top-level key), required")
    parser.add_argument("--service", default=DEFAULT_SERVICE, help="Service name (nested key), required")
    parser.add_argument("--replace", action="store_true", help="Replace result.json instead of append/upsert")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    convert_default_files(provider=args.provider, service=args.service, append=not args.replace)


if __name__ == "__main__":
    main()
