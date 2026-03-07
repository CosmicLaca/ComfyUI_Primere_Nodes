"""Convert `snippet.py` into grouped provider->service API schema JSON."""
# Basic usage:
#   python api_snippet_to_json.py --provider Gemini --service Imagen --replace
# Dry-run (preview only, no files written):
#   python api_snippet_to_json.py --provider Gemini --service Imagen --dry-run
# Validate generated schema against api_schemas.json:
#   python api_snippet_to_json.py --provider Gemini --service Imagen --validate
# List all registered provider/service pairs in result.json:
#   python api_snippet_to_json.py --list
# List all registered provider/service pairs in api_schemas.json:
#   python api_snippet_to_json.py --prodlist
# Custom snippet file path:
#   python api_snippet_to_json.py --snippet /path/to/my_snippet.py --provider Gemini --service Imagen
# ==========================================================================
# Manual: https://github.com/CosmicLaca/ComfyUI_Primere_Nodes/blob/master/Workflow/Manual/nodes/uniapi.md
# ==========================================================================

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from pathlib import Path
from typing import Any

PRIMERE_ROOT = Path(__file__).parent.parent.absolute()


class SnippetParseError(RuntimeError):
    """Raised when call extraction fails."""

SNIPPET_FILENAME = "snippet.py"
RESULT_FILENAME = "result.json"
DEFAULT_PROVIDER = ""
DEFAULT_SERVICE = ""
PLACEHOLDER_RE = re.compile(r"\{\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}\}")
PLACEHOLDER_ALIASES = {"number_of_images": "seed", "aspectRatio": "aspect_ratio"}

KNOWN_PARAM_OPTIONS: dict[str, list[str]] = {
    "model": ["example-model-1", "example-model-2"],
    "resolution": ["1K", "2K", "4K"],
    "regions": ["api.bfl.ai", "api.eu.bfl.ai", "api.us.bfl.ai"],
    "gen_method": ["generate", "edit"],
}

TYPE_MARKERS = {"INT", "FLOAT", "STRING", "BOOLEAN"}
INLINE_PLACEHOLDER_RE = re.compile(r"(?<!\{)\{\s*([a-zA-Z_][a-zA-Z0-9_]*)\s*\}(?!\})")
EXCLUDED_PARAMETER_KEYS = {"prompt", "batch", "response_modalities", "width", "height", "seed", "reference_images", "first_image", "last_image", "negative_prompt"}

DEFAULT_IMPORT_MODULES: dict[str, list[str]] = {
    "generic": [
        "import your_provider_sdk",
        "from your_provider_sdk import types",
    ]
}


def dotted_name(node: ast.AST) -> str:
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        base = dotted_name(node.value)
        return f"{base}.{node.attr}"
    return ast.unparse(node)


def placeholder(name: str) -> str:
    name = PLACEHOLDER_ALIASES.get(name, name)
    clean = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in name)
    while "__" in clean:
        clean = clean.replace("__", "_")
    return f"{{{{{clean.strip('_') or 'value'}}}}}"


def normalize_inline_placeholders(value: str) -> str:
    return INLINE_PLACEHOLDER_RE.sub(lambda m: placeholder(m.group(1)), value)


def node_to_template(node: ast.AST, path: str) -> Any:
    if isinstance(node, ast.Constant):
        if node.value is None or isinstance(node.value, bool):
            return node.value
        if isinstance(node.value, str):
            if node.value in TYPE_MARKERS:
                return placeholder(path)
            return normalize_inline_placeholders(node.value)
        return node.value

    if isinstance(node, ast.Name):
        special_literals = {"null": None, "true": True, "false": False}
        if node.id in special_literals:
            return special_literals[node.id]
        return placeholder(node.id)

    if isinstance(node, ast.Dict):
        out: dict[str, Any] = {}
        for key_node, value_node in zip(node.keys, node.values):
            if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                key = key_node.value
            else:
                key = ast.unparse(key_node)
            out[key] = node_to_template(value_node, key)
        return out

    if isinstance(node, ast.List):
        return [node_to_template(item, f"{path}_{idx}") for idx, item in enumerate(node.elts)]

    if isinstance(node, ast.Tuple):
        return [node_to_template(item, f"{path}_{idx}") for idx, item in enumerate(node.elts)]

    if isinstance(node, ast.Call):
        call_name = dotted_name(node.func)
        args = [node_to_template(arg, f"arg{idx}") for idx, arg in enumerate(node.args)]
        kwargs = {
            kw.arg if kw.arg else f"kw_{idx}": node_to_template(kw.value, kw.arg if kw.arg else f"kw_{idx}")
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


def find_main_call(tree: ast.AST) -> ast.Call:
    """Return the call in the last statement of the snippet (where the main API call lives)."""
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


def canonical_param_name(name: str) -> str:
    low = str(name or "").lower()
    if "aspect_ratio" in low:
        return "aspect_ratio"
    if "resolution" in low or "image_size" in low:
        return "resolution"
    if low == "model" or low.endswith("_model"):
        return "model"
    if low in {"negative_prompt", "multi_prompt", "system_prompt"}:
        return low
    if low in {"prompt", "contents"} or low.endswith("_prompt"):
        return "prompt"
    if "response_modalities" in low:
        return "response_modalities"
    return str(name)


def _collect_type_markers(node: ast.AST) -> dict[str, str]:
    marked: dict[str, str] = {}

    def walk(item: ast.AST) -> None:
        if isinstance(item, ast.Dict):
            for key_node, value_node in zip(item.keys, item.values):
                if isinstance(key_node, ast.Constant) and isinstance(key_node.value, str):
                    key = key_node.value
                else:
                    key = ast.unparse(key_node)

                if isinstance(value_node, ast.Constant) and isinstance(value_node.value, str) and value_node.value in TYPE_MARKERS:
                    marked[key] = value_node.value

                walk(value_node)
            return

        if isinstance(item, (ast.List, ast.Tuple, ast.Set)):
            for child in item.elts:
                walk(child)
            return

        if isinstance(item, ast.Call):
            for child in item.args:
                walk(child)
            for kw in item.keywords:
                walk(kw.value)

    walk(node)
    return marked


def build_possible_parameters(request_schema: dict[str, Any], type_markers: dict[str, str] | None = None) -> dict[str, Any]:
    placeholders = sorted(_collect_placeholders(request_schema))
    possible: dict[str, Any] = {}
    marker_map = {canonical_param_name(k): v for k, v in (type_markers or {}).items()}

    for name in placeholders:
        canonical = canonical_param_name(name)
        if canonical in EXCLUDED_PARAMETER_KEYS:
            continue
        marker = marker_map.get(canonical)
        if marker == "BOOLEAN":
            possible[canonical] = [False, True]
            continue
        if marker in {"INT", "FLOAT", "STRING"}:
            possible[canonical] = marker
            continue
        if canonical in KNOWN_PARAM_OPTIONS:
            possible[canonical] = KNOWN_PARAM_OPTIONS[canonical]
        elif canonical not in possible:
            possible[canonical] = [f"fake_{canonical}_value_1", f"fake_{canonical}_value_2"]

    return possible


def build_import_modules(tree: ast.Module, provider: str = "") -> list[str]:
    """Extract import statements so schema keeps service-specific dependencies editable."""
    imports: list[str] = []

    for node in tree.body:
        if isinstance(node, ast.Import):
            rendered = ", ".join(
                f"{alias.name} as {alias.asname}" if alias.asname else alias.name
                for alias in node.names
            )
            imports.append(f"import {rendered}")
        elif isinstance(node, ast.ImportFrom):
            module_name = "." * node.level + (node.module or "")
            rendered = ", ".join(
                f"{alias.name} as {alias.asname}" if alias.asname else alias.name
                for alias in node.names
            )
            imports.append(f"from {module_name} import {rendered}")

    # Preserve order while dropping accidental duplicates.
    unique_imports = list(dict.fromkeys(imports))
    if unique_imports:
        return unique_imports

    return DEFAULT_IMPORT_MODULES["generic"]


def build_service_schema(snippet: str, provider: str = DEFAULT_PROVIDER, service: str = DEFAULT_SERVICE) -> dict[str, Any]:
    tree = ast.parse(snippet)
    call = find_main_call(tree)

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

    type_markers = _collect_type_markers(call)

    service_schema = {
        "provider": provider,
        "service": service,
        "response_handler": response_handler_filename(provider, service),
        "reference_images_handler": reference_images_handler_filename(provider),
        "import_modules": build_import_modules(tree, provider=provider),
        "possible_parameters": build_possible_parameters(request_schema, type_markers=type_markers),
        "request": request_schema,
    }

    return service_schema


def _sanitize_name(value: str) -> str:
    clean = re.sub(r"[^A-Za-z0-9_]+", "_", str(value or "").strip())
    clean = re.sub(r"_+", "_", clean).strip("_")
    return clean or "default"


def response_handler_filename(provider: str, service: str) -> str:
    return f"{_sanitize_name(provider)}_{_sanitize_name(service)}.py"


def reference_images_handler_filename(provider: str) -> str:
    return f"{_sanitize_name(provider)}.py"


def _response_handlers_dir() -> Path:
    return Path(os.path.join(PRIMERE_ROOT, "components", "API", "responses"))


def _reference_handlers_dir() -> Path:
    return Path(os.path.join(PRIMERE_ROOT, "components", "API", "references"))


def _api_schemas_path() -> Path:
    return Path(os.path.join(PRIMERE_ROOT, "front_end", "api_schemas.json"))


def _apiconfig_path() -> Path:
    return Path(os.path.join(PRIMERE_ROOT, "json", "apiconfig.json"))


def check_provider_in_apiconfig(provider: str) -> None:
    """Raise SnippetParseError if provider is not found in apiconfig.json."""
    path = _apiconfig_path()
    if not path.exists():
        raise SnippetParseError(
            f"apiconfig.json not found at {path}.\n"
            "  Rename json/apiconfig.example.json to json/apiconfig.json and add your provider credentials."
        )
    try:
        apiconfig = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SnippetParseError(f"apiconfig.json is malformed and cannot be read: {e}") from e

    if provider not in apiconfig:
        registered = ", ".join(sorted(apiconfig.keys())) or "(none)"
        raise SnippetParseError(
            f"Provider '{provider}' not found in apiconfig.json.\n"
            f"  Registered providers: {registered}\n"
            f"  Add '{provider}' to json/apiconfig.json before writing this schema."
        )


def ensure_response_handler_file(filename: str) -> Path:
    responses_dir = _response_handlers_dir()
    responses_dir.mkdir(parents=True, exist_ok=True)
    target = Path(os.path.join(responses_dir, filename))
    if target.exists():
        return target

    template = (
        "from __future__ import annotations\n\n"
        "from typing import Any\n\n\n"
        "def handle_response(api_result: Any, schema: dict[str, Any] | None = None):\n"
        "    return None\n"
    )
    target.write_text(template, encoding="utf-8")
    return target


def ensure_reference_images_handler_file(filename: str) -> Path:
    references_dir = _reference_handlers_dir()
    references_dir.mkdir(parents=True, exist_ok=True)
    target = Path(os.path.join(references_dir, filename))
    if target.exists():
        return target

    template = (
        "from __future__ import annotations\n\n"
        "from typing import Any\n\n\n"
        "def handle_reference_images(img_binary_api: Any = None, temp_file_ref: str = '', loaded_client_for_upload: Any = None, **_: Any):\n"
        "    output = img_binary_api if isinstance(img_binary_api, list) else []\n"
        "    if temp_file_ref:\n"
        "        output.append(temp_file_ref)\n"
        "    return output\n"
    )
    target.write_text(template, encoding="utf-8")
    return target


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
    if service in provider_map:
        print(f"[WARNING] Overwriting existing '{provider}/{service}' schema in result.json.")
    provider_map[service] = service_schema
    out[provider] = provider_map
    return out


def list_production_services() -> None:
    """Print all provider/service pairs registered in api_schemas.json."""
    path = _api_schemas_path()
    if not path.exists():
        print(f"api_schemas.json not found at {path}")
        return
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SnippetParseError(f"api_schemas.json is malformed and cannot be read: {e}") from e

    if not registry:
        print("No services registered.")
        return

    print(f"Registered services in {path}:")
    for provider in sorted(registry.keys()):
        services = registry[provider]
        if isinstance(services, dict):
            for service in sorted(services.keys()):
                print(f"  {provider} / {service}")
        else:
            print(f"  {provider} (malformed entry)")


def list_registered_services(result_path: Path | None = None) -> None:
    """Print all provider/service pairs registered in result.json."""
    path = result_path or Path(os.path.join(Path.cwd(), RESULT_FILENAME))
    if not path.exists():
        print(f"No {RESULT_FILENAME} found in {path.parent}")
        return
    try:
        registry = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as e:
        raise SnippetParseError(f"result.json is malformed and cannot be read: {e}") from e

    if not registry:
        print("No services registered.")
        return

    print(f"Registered services in {path}:")
    for provider in sorted(registry.keys()):
        services = registry[provider]
        if isinstance(services, dict):
            for service in sorted(services.keys()):
                print(f"  {provider} / {service}")
        else:
            print(f"  {provider} (malformed entry)")


def validate_service_schema(service_schema: dict[str, Any]) -> list[str]:
    """Check schema for required fields and conflicts. Returns list of warning strings."""
    issues: list[str] = []

    required_fields = ["provider", "service", "response_handler", "reference_images_handler", "import_modules", "possible_parameters", "request"]
    for field in required_fields:
        if not service_schema.get(field):
            issues.append(f"Missing or empty required field: '{field}'")

    request = service_schema.get("request", {})
    if isinstance(request, dict):
        if not request.get("method"):
            issues.append("request.method is missing")
        if not request.get("endpoint"):
            issues.append("request.endpoint is missing")

    provider = str(service_schema.get("provider") or "")
    service = str(service_schema.get("service") or "")
    schemas_path = _api_schemas_path()
    if schemas_path.exists():
        try:
            api_schemas = json.loads(schemas_path.read_text(encoding="utf-8"))
            if provider in api_schemas and service in api_schemas.get(provider, {}):
                issues.append(f"CONFLICT: '{provider}/{service}' already exists in api_schemas.json")
        except json.JSONDecodeError:
            issues.append("Could not read api_schemas.json for conflict check")
    else:
        issues.append(f"api_schemas.json not found at {schemas_path}")

    return issues


def _prompt_if_empty(value: str, label: str) -> str:
    """Return value as-is if non-empty, otherwise prompt the user interactively."""
    if value.strip():
        return value
    try:
        entered = input(f"{label}: ").strip()
    except (EOFError, KeyboardInterrupt):
        raise SnippetParseError("Input cancelled.")
    if not entered:
        raise SnippetParseError(f"{label} is required.")
    return entered


def convert_default_files(
    base_dir: Path | None = None,
    provider: str = DEFAULT_PROVIDER,
    service: str = DEFAULT_SERVICE,
    append: bool = True,
    dry_run: bool = False,
    validate: bool = False,
    snippet_override: Path | None = None,
) -> Path:
    root = base_dir or Path.cwd()

    if snippet_override is not None:
        snippet_path = snippet_override.resolve()
    else:
        snippet_path = Path(os.path.join(root, SNIPPET_FILENAME))

    if not snippet_path.exists():
        raise SnippetParseError(f"Missing snippet file: {snippet_path}")

    snippet = snippet_path.read_text(encoding="utf-8")
    service_schema = build_service_schema(snippet, provider=provider, service=service)

    if validate:
        issues = validate_service_schema(service_schema)
        if issues:
            print("[VALIDATE] Issues found:")
            for issue in issues:
                print(f"  - {issue}")
        else:
            print("[VALIDATE] Schema looks good.")
        check_provider_in_apiconfig(provider)

    if dry_run:
        print("[DRY RUN] No files written.")
        print(json.dumps(service_schema, ensure_ascii=False, indent=2))
        return Path(os.path.join(root, RESULT_FILENAME))

    ensure_response_handler_file(str(service_schema.get("response_handler") or ""))
    ensure_reference_images_handler_file(str(service_schema.get("reference_images_handler") or ""))

    result_path = Path(os.path.join(root, RESULT_FILENAME))
    if append and result_path.exists():
        try:
            existing = json.loads(result_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as e:
            raise SnippetParseError(f"result.json is malformed and cannot be read: {e}") from e
    else:
        existing = {}

    output = upsert_service_schema(existing, service_schema)
    encoded = json.dumps(output, ensure_ascii=False, indent=2)

    result_path.write_text(encoded + "\n", encoding="utf-8")
    print(encoded)
    return result_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--provider", default=DEFAULT_PROVIDER, help="Provider name (top-level key) — prompted if omitted")
    parser.add_argument("--service", default=DEFAULT_SERVICE, help="Service name (nested key) — prompted if omitted")
    parser.add_argument("--replace", action="store_true", help="Replace result.json instead of append/upsert")
    parser.add_argument("--dry-run", action="store_true", dest="dry_run", help="Print generated schema without writing any files")
    parser.add_argument("--validate", action="store_true", help="Check schema for issues and conflicts with api_schemas.json")
    parser.add_argument("--list", action="store_true", dest="list_services", help="List all registered provider/service pairs in result.json")
    parser.add_argument("--prodlist", action="store_true", dest="prod_list", help="List all registered provider/service pairs in api_schemas.json")
    parser.add_argument("--snippet", default=None, metavar="PATH", help="Path to snippet file (default: snippet.py in current directory)")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.list_services:
        list_registered_services()
        return

    if args.prod_list:
        list_production_services()
        return

    provider = _prompt_if_empty(args.provider, "Provider name (e.g. Gemini)")
    service = _prompt_if_empty(args.service, "Service name (e.g. Imagen)")
    snippet_override = Path(args.snippet) if args.snippet else None

    try:
        convert_default_files(
            provider=provider,
            service=service,
            append=not args.replace,
            dry_run=args.dry_run,
            validate=args.validate,
            snippet_override=snippet_override,
        )
    except SnippetParseError as e:
        print(f"ERROR: {e}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
