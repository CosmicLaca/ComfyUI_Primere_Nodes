from __future__ import annotations

from copy import deepcopy
from typing import Any


def _is_service_schema(node: Any) -> bool:
    return isinstance(node, dict) and isinstance(node.get("request"), dict)


def normalize_registry(raw: dict[str, Any] | None) -> dict[str, dict[str, dict[str, Any]]]:
    """Normalize API schema JSON to provider->service->schema mapping.

    Supports legacy single-schema format:
      {"provider": "Gemini", "request": {...}}

    And new grouped format:
      {"Gemini": {"Text2Image Nanobanana": {"provider": "Gemini", "service": "...", "request": {...}}}}
    """
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