from __future__ import annotations
from typing import Any, Callable

def _resolve_key(mapping: dict[str, Any], key: str, canonicalize_key: Callable[[str], str] | None = None) -> str | None:
    if key in mapping:
        return key
    if canonicalize_key is None:
        return None

    expected = canonicalize_key(key)
    for existing_key in mapping.keys():
        if canonicalize_key(str(existing_key)) == expected:
            return str(existing_key)
    return None


def _walk_dict_path(
    payload: dict[str, Any],
    parts: list[str],
    canonicalize_key: Callable[[str], str] | None = None,
) -> tuple[bool, Any]:
    current: Any = payload
    for part in parts:
        if not isinstance(current, dict):
            return False, None
        resolved = _resolve_key(current, part, canonicalize_key)
        if resolved is None:
            return False, None
        current = current.get(resolved)
    return True, current


def _get_by_path(
    payload: dict[str, Any],
    path: str,
    *,
    use_kwargs_fallback: bool,
    canonicalize_key: Callable[[str], str] | None = None,
) -> Any:
    parts = [part for part in str(path or "").split(".") if part]
    if len(parts) == 0:
        return None

    found, value = _walk_dict_path(payload, parts, canonicalize_key)
    if found:
        return value

    if not use_kwargs_fallback:
        return None

    # Support schema SDK-call shape where real fields are under "$kwargs".
    expanded_parts: list[str] = []
    for index, part in enumerate(parts):
        expanded_parts.append(part)
        if index < len(parts) - 1:
            expanded_parts.append("$kwargs")

    found, value = _walk_dict_path(payload, expanded_parts, canonicalize_key)
    return value if found else None


def _remove_by_path(
    payload: dict[str, Any],
    path: str,
    *,
    use_kwargs_fallback: bool,
    canonicalize_key: Callable[[str], str] | None = None,
) -> bool:
    parts = [part for part in str(path or "").split(".") if part]
    if len(parts) == 0:
        return False

    direct_parent_parts = parts[:-1]
    direct_last = parts[-1]

    found, parent = _walk_dict_path(payload, direct_parent_parts, canonicalize_key)
    if found and isinstance(parent, dict):
        resolved_last = _resolve_key(parent, direct_last, canonicalize_key)
        if resolved_last is not None:
            del parent[resolved_last]
            return True

    if not use_kwargs_fallback:
        return False

    expanded_parts: list[str] = []
    for index, part in enumerate(parts):
        expanded_parts.append(part)
        if index < len(parts) - 1:
            expanded_parts.append("$kwargs")

    found, parent = _walk_dict_path(payload, expanded_parts[:-1], canonicalize_key)
    if found and isinstance(parent, dict):
        resolved_last = _resolve_key(parent, expanded_parts[-1], canonicalize_key)
        if resolved_last is not None:
            del parent[resolved_last]
            return True

    return False


def _condition_match(
    payload: dict[str, Any],
    condition: dict[str, Any],
    *,
    use_kwargs_fallback: bool,
    canonicalize_key: Callable[[str], str] | None = None,
) -> bool:
    path = condition.get("path") or condition.get("key")
    if not isinstance(path, str) or path.strip() == "":
        return False

    expected = condition.get("equals", condition.get("value"))
    current = _get_by_path(
        payload,
        path,
        use_kwargs_fallback=use_kwargs_fallback,
        canonicalize_key=canonicalize_key,
    )
    return current == expected


def apply_exclusions_to_payload(
    payload: dict[str, Any],
    exclusions: list[dict[str, Any]] | None,
    *,
    use_kwargs_fallback: bool,
    canonicalize_key: Callable[[str], str] | None = None,
) -> dict[str, Any]:
    if not isinstance(payload, dict):
        return payload

    for rule in exclusions or []:
        if not isinstance(rule, dict):
            continue

        condition = rule.get("when") if isinstance(rule.get("when"), dict) else rule.get("if")
        if isinstance(condition, dict):
            if not _condition_match(
                payload,
                condition,
                use_kwargs_fallback=use_kwargs_fallback,
                canonicalize_key=canonicalize_key,
            ):
                continue

        remove_spec = rule.get("remove")
        remove_paths: list[str] = []
        if isinstance(remove_spec, str):
            remove_paths = [remove_spec]
        elif isinstance(remove_spec, list):
            remove_paths = [item for item in remove_spec if isinstance(item, str)]

        for remove_path in remove_paths:
            _remove_by_path(
                payload,
                remove_path,
                use_kwargs_fallback=use_kwargs_fallback,
                canonicalize_key=canonicalize_key,
            )

    return payload

def apply_sdk_request_exclusions(
    args: list[Any],
    kwargs: dict[str, Any],
    exclusions: list[dict[str, Any]] | None,
) -> tuple[list[Any], dict[str, Any]]:
    """Apply schema-driven exclusion rules to SDK kwargs."""
    if not isinstance(kwargs, dict):
        return args, kwargs

    filtered_kwargs = apply_exclusions_to_payload(
        kwargs,
        exclusions,
        use_kwargs_fallback=True,
        canonicalize_key=None,
    )
    return args, filtered_kwargs
