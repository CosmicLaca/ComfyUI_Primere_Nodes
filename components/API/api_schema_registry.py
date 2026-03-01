from __future__ import annotations

from copy import deepcopy
from typing import Any
from pathlib import Path
import json


def _is_service_schema(node: Any) -> bool:
    return isinstance(node, dict) and isinstance(node.get("request"), dict)


def normalize_registry(raw: dict[str, Any] | None) -> dict[str, dict[str, dict[str, Any]]]:
    if not isinstance(raw, dict):
        return {}

    if _is_service_schema(raw):
        provider = str(raw.get("provider") or "custom")
        service = str(raw.get("service") or "default")
        return {provider: {service: deepcopy(raw)}}

    registry: dict[str, dict[str, dict[str, Any]]] = {}
    for provider, services in raw.items():
        if not isinstance(services, dict):
            continue
        service_map: dict[str, dict[str, Any]] = {}
        for service_name, schema in services.items():
            if not _is_service_schema(schema):
                continue
            item = deepcopy(schema)
            item.setdefault("provider", str(provider))
            item.setdefault("service", str(service_name))
            service_map[str(service_name)] = item
        if service_map:
            registry[str(provider)] = service_map
    return registry


def list_providers(registry: dict[str, dict[str, dict[str, Any]]]) -> list[str]:
    return sorted(registry.keys())


def list_services(registry: dict[str, dict[str, dict[str, Any]]], provider: str) -> list[str]:
    return sorted(registry.get(provider, {}).keys())


def get_schema(
    registry: dict[str, dict[str, dict[str, Any]]],
    provider: str,
    service: str | None,
) -> tuple[dict[str, Any] | None, str | None]:
    service_map = registry.get(provider, {})
    if not service_map:
        return None, None

    if service and service in service_map:
        return deepcopy(service_map[service]), service

    first_service = next(iter(service_map.keys()))
    return deepcopy(service_map[first_service]), first_service

def _load_json_file(file_path: str, label: str) -> Any:
    path = str(file_path or "").strip()
    if not path:
        raise RuntimeError(f"Missing path for {label} JSON file")
    if not Path(path).exists():
        raise RuntimeError(f"{label} JSON file not found: {path}")

    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise RuntimeError(f"Invalid JSON syntax in {label} file '{path}' at line {exc.lineno}, column {exc.colno}: {exc.msg}") from exc


def load_and_validate_api_schema_registry(schema_path: str, apiconfig_path: str | None = None) -> dict[str, dict[str, dict[str, Any]]]:
    schema_data = _load_json_file(schema_path, "API schema registry")
    if not isinstance(schema_data, dict):
        raise RuntimeError("API schema registry root must be a JSON object mapping providers to services.")

    config_path = apiconfig_path or os.path.join(PRIMERE_ROOT, "json", "apiconfig.example.json")
    apiconfig_data = _load_json_file(config_path, "API config")
    if not isinstance(apiconfig_data, dict):
        raise RuntimeError("API config root must be a JSON object mapping provider names.")

    allowed_providers = {str(name).strip() for name in apiconfig_data.keys() if str(name).strip()}
    if len(allowed_providers) == 0:
        raise RuntimeError("API config does not define any provider names.")

    validated: dict[str, dict[str, dict[str, Any]]] = {}

    for provider_key, services in schema_data.items():
        provider_name = str(provider_key).strip()
        if not provider_name:
            raise RuntimeError("Schema provider key cannot be empty.")
        if provider_name not in allowed_providers:
            allowed = ", ".join(sorted(allowed_providers))
            raise RuntimeError(f"Provider '{provider_name}' in API schema is not registered in API config '{config_path}'. Allowed providers: {allowed}")
        if not isinstance(services, dict):
            raise RuntimeError(f"Provider '{provider_name}' value must be an object mapping services to schemas.")

        validated_services: dict[str, dict[str, Any]] = {}
        for service_key, schema in services.items():
            service_name = str(service_key).strip()
            if not service_name:
                raise RuntimeError(f"Provider '{provider_name}' contains an empty service key.")
            if not isinstance(schema, dict):
                raise RuntimeError(f"Schema for provider '{provider_name}' service '{service_name}' must be a JSON object.")

            inner_provider = str(schema.get("provider") or "").strip()
            inner_service = str(schema.get("service") or "").strip()
            if inner_provider != provider_name:
                raise RuntimeError(f"Provider/service mismatch at '{provider_name}/{service_name}': schema field 'provider' must equal '{provider_name}', got '{inner_provider or '<missing>'}'.")
            if inner_service != service_name:
                raise RuntimeError(f"Provider/service mismatch at '{provider_name}/{service_name}': schema field 'service' must equal '{service_name}', got '{inner_service or '<missing>'}'.")

            request = schema.get("request")
            if not isinstance(request, dict):
                raise RuntimeError(f"Schema '{provider_name}/{service_name}' must contain object key 'request'.")

            method = str(request.get("method") or "").strip().upper()
            if not method:
                raise RuntimeError(f"Schema '{provider_name}/{service_name}' request.method is required.")
            endpoint = str(request.get("endpoint") or "").strip()
            if not endpoint:
                raise RuntimeError(f"Schema '{provider_name}/{service_name}' request.endpoint is required.")

            possible_parameters = schema.get("possible_parameters", {})
            if not isinstance(possible_parameters, dict):
                raise RuntimeError(f"Schema '{provider_name}/{service_name}' key 'possible_parameters' must be an object.")

            import_modules = schema.get("import_modules", [])
            if not isinstance(import_modules, list):
                raise RuntimeError(f"Schema '{provider_name}/{service_name}' key 'import_modules' must be a list of import statements." )
            for idx, import_line in enumerate(import_modules):
                if not isinstance(import_line, str) or not import_line.strip():
                    raise RuntimeError(f"Schema '{provider_name}/{service_name}' import_modules[{idx}] must be a non-empty string.")

            validated_services[service_name] = schema
        validated[provider_name] = validated_services

    return validated