from __future__ import annotations

import ast
import dataclasses
import json
import re
from typing import Any


def _normalize_input(value: Any) -> Any:
    if isinstance(value, str):
        text_value = value.strip()
        if not text_value:
            return value
        try:
            return json.loads(text_value)
        except Exception:
            normalized = re.sub(r"\bUnset\s*\(\s*\)", "None", text_value)
            try:
                return ast.literal_eval(normalized)
            except Exception:
                return value
    return value


def _to_builtin(value: Any) -> Any:
    value = _normalize_input(value)

    if isinstance(value, dict):
        return {str(k): _to_builtin(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_to_builtin(item) for item in value]
    if isinstance(value, tuple):
        return [_to_builtin(item) for item in value]
    if dataclasses.is_dataclass(value):
        return _to_builtin(dataclasses.asdict(value))
    if hasattr(value, "__dict__") and not isinstance(value, (str, int, float, bool, type(None))):
        return {str(k): _to_builtin(v) for k, v in vars(value).items() if not str(k).startswith("_")}
    return value


def _extract_content(payload: Any) -> str | None:
    data = _to_builtin(payload)
    if not isinstance(data, dict):
        return None

    choices = data.get("choices")
    if not isinstance(choices, list) or len(choices) == 0:
        return None

    first_choice = choices[0] if isinstance(choices[0], dict) else None
    if not isinstance(first_choice, dict):
        return None

    message = first_choice.get("message")
    if not isinstance(message, dict):
        return None

    content = message.get("content")
    if isinstance(content, str) and content.strip():
        return content
    return None


def handle_response(api_result, schema=None, loaded_client=None, response_url=None, client=None, sdk_context=None):
    payload = _to_builtin(api_result)
    content = _extract_content(payload)

    if content is not None:
        return ["text_result", content.encode("utf-8")]

    fallback = {
        "error": "Mistral content not found in response. Saved full response instead.",
        "response": payload,
    }
    fallback_text = json.dumps(fallback, ensure_ascii=False, indent=2, default=str)
    return ["text_result", fallback_text.encode("utf-8")]
