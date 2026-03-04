from __future__ import annotations
from typing import Any

def _walk_dict_path(payload: dict[str, Any], parts: list[str]) -> tuple[bool, Any]:
    current: Any = payload
    for part in parts:
        if not isinstance(current, dict) or part not in current:
            return False, None
        current = current.get(part)
    return True, current

def _get_by_path(payload: dict[str, Any], path: str) -> Any:
    parts = [part for part in str(path or "").split(".") if part]
    if len(parts) == 0:
        return None

    found, value = _walk_dict_path(payload, parts)
    if found:
        return value

    # Support schema SDK-call shape where real fields are under "$kwargs".
    expanded_parts: list[str] = []
    for index, part in enumerate(parts):
        expanded_parts.append(part)
        if index < len(parts) - 1:
            expanded_parts.append("$kwargs")

    found, value = _walk_dict_path(payload, expanded_parts)
    return value if found else None

def _remove_by_path(payload: dict[str, Any], path: str) -> bool:
    parts = [part for part in str(path or "").split(".") if part]
    if len(parts) == 0:
        return False

    direct_parent_parts = parts[:-1]
    direct_last = parts[-1]

    found, parent = _walk_dict_path(payload, direct_parent_parts)
    if found and isinstance(parent, dict) and direct_last in parent:
        del parent[direct_last]
        return True

    expanded_parts: list[str] = []
    for index, part in enumerate(parts):
        expanded_parts.append(part)
        if index < len(parts) - 1:
            expanded_parts.append("$kwargs")

    found, parent = _walk_dict_path(payload, expanded_parts[:-1])
    if found and isinstance(parent, dict) and expanded_parts[-1] in parent:
        del parent[expanded_parts[-1]]
        return True

    return False

def _condition_match(payload: dict[str, Any], condition: dict[str, Any]) -> bool:
    path = condition.get("path") or condition.get("key")
    if not isinstance(path, str) or path.strip() == "":
        return False

    expected = condition.get("equals", condition.get("value"))
    current = _get_by_path(payload, path)
    return current == expected

def apply_sdk_request_exclusions(
    args: list[Any],
    kwargs: dict[str, Any],
    exclusions: list[dict[str, Any]] | None,
) -> tuple[list[Any], dict[str, Any]]:
    """Apply schema-driven exclusion rules to SDK kwargs."""
    if not isinstance(kwargs, dict):
        return args, kwargs

    for rule in exclusions or []:
        if not isinstance(rule, dict):
            continue

        condition = rule.get("when") if isinstance(rule.get("when"), dict) else rule.get("if")
        if not isinstance(condition, dict) or not _condition_match(kwargs, condition):
            continue

        remove_spec = rule.get("remove")
        remove_paths: list[str] = []
        if isinstance(remove_spec, str):
            remove_paths = [remove_spec]
        elif isinstance(remove_spec, list):
            remove_paths = [item for item in remove_spec if isinstance(item, str)]

        for remove_path in remove_paths:
            _remove_by_path(kwargs, remove_path)

    return args, kwargs