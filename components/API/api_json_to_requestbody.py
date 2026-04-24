from __future__ import annotations
from typing import Any
import os
import ast
import mimetypes
import re
from pathlib import Path
from . import external_api_backend
from . import api_helper
from . import request_exceptions

class KlingRequestBuilder:
    MULTI_PROMPT_SEPARATOR_RE = re.compile(r"(?m)^\s*---\s*$")
    MULTI_INPUT_MODEL_TYPES = {"reference-to-video", "video-to-video/reference"}

    @staticmethod
    def is_kling_schema(schema: Any) -> bool:
        if not isinstance(schema, dict):
            return False
        schema_service = str(schema.get("service", "")).lower()
        schema_provider = str(schema.get("provider", "")).lower()
        request_data = schema.get("request") if isinstance(schema.get("request"), dict) else {}
        sdk_call = request_data.get("sdk_call") if isinstance(request_data.get("sdk_call"), dict) else {}
        sdk_args = sdk_call.get("args") if isinstance(sdk_call.get("args"), list) else []
        endpoint = str(sdk_args[0] if len(sdk_args) > 0 else "").lower()
        return "kling" in schema_service or "kling" in endpoint or (schema_provider == "fal" and "kling" in endpoint)

    @classmethod
    def supports_multi_inputs_for_model_type(cls, model_type: Any) -> bool:
        return str(model_type or "").strip().lower() in cls.MULTI_INPUT_MODEL_TYPES

    @classmethod
    def resolve_model_type(cls, selected_parameters: dict[str, Any], schema: dict[str, Any]) -> str:
        model_type_value = selected_parameters.get("model_type")
        if model_type_value not in (None, ""):
            return str(model_type_value)

        possible_parameters = schema.get("possible_parameters", {}) if isinstance(schema, dict) else {}
        model_type_options = possible_parameters.get("model_type") if isinstance(possible_parameters, dict) else None
        if isinstance(model_type_options, list) and len(model_type_options) > 0:
            return str(model_type_options[0])
        return ""

    @staticmethod
    def apply_prompt_logic(selected_parameters: dict[str, Any], prompt_value: Any) -> dict[str, Any]:
        updated = dict(selected_parameters)
        if not isinstance(prompt_value, str):
            existing_multi_prompt = updated.get("multi_prompt")
            if isinstance(existing_multi_prompt, list) and len(existing_multi_prompt) > 0:
                updated["prompt"] = ""
                updated["shot_type"] = "customize"
            else:
                updated.pop("shot_type", None)
            return updated

        normalized_prompt = prompt_value.replace("\r\n", "\n")
        prompt_blocks = [block.strip() for block in KlingRequestBuilder.MULTI_PROMPT_SEPARATOR_RE.split(normalized_prompt) if block.strip()]
        existing_multi_prompt = updated.get("multi_prompt")
        has_existing_multi_prompt = isinstance(existing_multi_prompt, list) and len(existing_multi_prompt) > 0
        is_multi_prompt = len(prompt_blocks) > 1 or has_existing_multi_prompt

        updated.pop("shot_type", None)

        if len(prompt_blocks) == 0:
            if has_existing_multi_prompt:
                updated["prompt"] = ""
                updated["shot_type"] = "customize"
            return updated

        if has_existing_multi_prompt and len(prompt_blocks) <= 1:
            updated["prompt"] = ""
            updated["shot_type"] = "customize"
            return updated

        if not is_multi_prompt:
            updated["prompt"] = prompt_blocks[0]
            updated.pop("multi_prompt", None)
            return updated

        updated["prompt"] = ""
        multi_prompt: list[dict[str, str]] = []
        for raw_block in prompt_blocks:
            prompt_part = raw_block.strip()
            duration_part = "Default"
            if "::" in raw_block:
                prompt_candidate, duration_candidate = raw_block.rsplit("::", 1)
                prompt_candidate = prompt_candidate.strip()
                duration_candidate = duration_candidate.strip()
                if prompt_candidate:
                    prompt_part = prompt_candidate
                duration_part = duration_candidate if duration_candidate else "Default"
            multi_prompt.append({"prompt": prompt_part, "duration": duration_part})

        if len(multi_prompt) > 0:
            updated["multi_prompt"] = multi_prompt
            updated["shot_type"] = "customize"
        return updated

    @staticmethod
    def normalize_path_input(path_input: Any) -> list[str]:
        if path_input in (None, ""):
            return []
        if isinstance(path_input, (list, tuple)):
            return [str(item) for item in path_input if item not in (None, "")]
        if isinstance(path_input, str):
            stripped = path_input.strip()
            if not stripped:
                return []
            if stripped.startswith("[") and stripped.endswith("]"):
                try:
                    parsed = ast.literal_eval(stripped)
                    if isinstance(parsed, (list, tuple)):
                        return [str(item) for item in parsed if item not in (None, "")]
                except Exception:
                    pass
            return [stripped]
        return []

    @staticmethod
    def _upload_local_reference(path_value: str, loaded_client_for_upload: Any) -> str:
        if hasattr(loaded_client_for_upload, "upload_file") and os.path.isfile(path_value):
            return str(loaded_client_for_upload.upload_file(path_value))
        return str(path_value)

    @classmethod
    def _detect_media_type(cls, source_path: str) -> str:
        guessed_mime, _ = mimetypes.guess_type(source_path)
        mime_low = str(guessed_mime or "").lower()
        suffix = Path(source_path).suffix.lower()

        if mime_low.startswith("video/") or suffix in {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}:
            return "video"
        if mime_low.startswith("image/") or suffix in {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}:
            return "image"
        return "other"

    @classmethod
    def _build_single_element(cls, path_input: Any, loaded_client_for_upload: Any) -> dict[str, Any] | None:
        normalized_paths = cls.normalize_path_input(path_input)
        if len(normalized_paths) == 0:
            return None

        image_urls: list[str] = []
        video_url: str | None = None

        for single_path in normalized_paths:
            source_path = str(single_path).strip()
            if not source_path:
                continue
            media_type = cls._detect_media_type(source_path)
            if media_type not in {"image", "video"}:
                continue

            uploaded_path = cls._upload_local_reference(source_path, loaded_client_for_upload)
            if media_type == "image":
                image_urls.append(uploaded_path)
            elif video_url is None:
                video_url = uploaded_path

        if len(image_urls) == 0:
            return None

        element_object: dict[str, Any] = {
            "frontal_image_url": image_urls[0],
            "reference_image_urls": image_urls[1:],
        }
        if video_url is not None:
            element_object["video_url"] = video_url
        return element_object

    @classmethod
    def build_elements(cls, path_inputs: list[Any], loaded_client_for_upload: Any) -> list[dict[str, Any]]:
        element_list: list[dict[str, Any]] = []
        for single_input in path_inputs:
            element_object = cls._build_single_element(single_input, loaded_client_for_upload)
            if isinstance(element_object, dict) and len(element_object) > 0:
                element_list.append(element_object)
        return element_list

    @staticmethod
    def summarize_multi_prompt_duration(multi_prompt_value: Any, default_duration: int = 5) -> int | None:
        if not isinstance(multi_prompt_value, list) or len(multi_prompt_value) == 0:
            return None

        total_duration = 0
        for entry in multi_prompt_value:
            duration_value = default_duration
            if isinstance(entry, dict):
                raw_duration = entry.get("duration")
                if isinstance(raw_duration, str):
                    stripped = raw_duration.strip()
                    if stripped and stripped.lower() != "default":
                        try:
                            duration_value = int(float(stripped))
                        except (TypeError, ValueError):
                            duration_value = default_duration
                elif isinstance(raw_duration, (int, float)):
                    duration_value = int(raw_duration)
            total_duration += max(0, int(duration_value))

        return total_duration

def _provider_api_key(spec: dict[str, Any]) -> Any:
    provider = str(spec.get("provider") or "").strip()
    if not provider:
        return None
    try:
        config = api_helper.get_api_config("apiconfig.json")
    except Exception:
        return None
    entry = config.get(provider, {}) if isinstance(config, dict) else {}
    if not isinstance(entry, dict):
        return None
    value = entry.get("APIKEY")
    return value if value not in (None, "") else None


def _secret_placeholder_default(key: str, spec: dict[str, Any]) -> Any:
    normalized = str(key or "").strip()
    low = normalized.lower()
    looks_like_secret = (
        low.endswith(("_api_key", "_key", "_token", "_secret", "_password"))
        or low in {"api_key", "apikey", "x_api_key", "provider_api_key", "authorization", "auth_token", "access_token", "bearer_token", "token"}
    )

    if normalized.upper() == normalized and "_" in normalized:
        if looks_like_secret:
            return normalized

    if looks_like_secret:
        direct_env = os.environ.get(normalized) or os.environ.get(normalized.upper())
        if direct_env not in (None, ""):
            return direct_env

    if looks_like_secret:
        return _provider_api_key(spec)

    return None

def _marker_default(marker: Any) -> Any:
    if not isinstance(marker, str):
        return None
    value = marker.strip().upper()
    if value == "INT":
        return 1
    if value == "FLOAT":
        return 1.0
    if value == "STRING":
        return ""
    if value == "BOOLEAN":
        return False
    return None

def _default_value(name: str) -> Any:
    k = name.lower()
    if k in {"prompt", "negative_prompt", "multi_prompt"}:
        return None
    if "response_modalities" in k:
        return "IMAGE"
    if "model" in k:
        return "valid-model-name"
    if "aspect_ratio" in k:
        return "1:1"
    if "resolution" in k:
        return "1K"
    if "number" in k or "count" in k or "width" in k or "height" in k:
        return 1
    return None

def _is_optional_image_input(name: str) -> bool:
    low = str(name or "").lower()
    if low in {"reference_images", "first_image", "last_image"}:
        return True
    if low == "input_image" or low.startswith("input_image_"):
        return True
    return False

def _build_values(spec: dict[str, Any], values: dict[str, Any] | None = None) -> dict[str, Any]:
    user_values = values or {}
    request = spec.get("request", {})
    possible_parameters = spec.get("possible_parameters", {}) if isinstance(spec.get("possible_parameters"), dict) else {}

    placeholders = external_api_backend.list_placeholders(
        {
            "endpoint": request.get("endpoint", ""),
            "method": request.get("method", "POST"),
            "headers": request.get("headers", {}),
            "query": request.get("query", {}),
            "body": request.get("body"),
            "sdk_call": request.get("sdk_call"),
        }
    )

    resolved: dict[str, Any] = {}
    for key in placeholders:
        canonical = external_api_backend.canonical_param_name(key)
        selected = None

        if key in user_values and user_values[key] not in (None, ""):
            selected = user_values[key]
        elif canonical in user_values and user_values[canonical] not in (None, ""):
            selected = user_values[canonical]
        elif key in possible_parameters and isinstance(possible_parameters[key], list) and len(possible_parameters[key]) > 0:
            selected = possible_parameters[key][0]
        elif canonical in possible_parameters and isinstance(possible_parameters[canonical], list) and len(possible_parameters[canonical]) > 0:
            selected = possible_parameters[canonical][0]
        else:
            if _is_optional_image_input(canonical) or _is_optional_image_input(key):
                resolved[key] = None
                continue
            secret_selected = _secret_placeholder_default(key, spec)
            if secret_selected is not None:
                selected = secret_selected
            else:
                marker_selected = _marker_default(possible_parameters.get(key)) if key in possible_parameters else None
                if marker_selected is None and canonical in possible_parameters:
                    marker_selected = _marker_default(possible_parameters.get(canonical))
                selected = marker_selected if marker_selected is not None else _default_value(canonical)

        if selected is not None:
            raw_marker = possible_parameters.get(key) if not isinstance(possible_parameters.get(key), list) else possible_parameters.get(canonical)
            if isinstance(raw_marker, str):
                t = raw_marker.strip().upper()
                if t == "INT":
                    try:
                        selected = int(selected)
                    except (ValueError, TypeError):
                        pass
                elif t == "FLOAT":
                    try:
                        selected = round(float(selected), 1)
                    except (ValueError, TypeError):
                        pass

        resolved[key] = selected

    return resolved

def _filter_used_values_from_template(
    spec: dict[str, Any],
    used_values: dict[str, Any],
    remove_paths: list[str],
) -> dict[str, Any]:
    request = spec.get("request", {}) if isinstance(spec, dict) else {}
    request_template = {
        "endpoint": request.get("endpoint", ""),
        "method": request.get("method", "POST"),
        "headers": request.get("headers", {}),
        "query": request.get("query", {}),
        "body": request.get("body"),
        "sdk_call": request.get("sdk_call"),
    }

    sdk_call_template = request_template.get("sdk_call")
    template_args, template_kwargs = external_api_backend.normalize_sdk_call(sdk_call_template if isinstance(sdk_call_template, dict) else None)
    template_kwargs_copy = dict(template_kwargs)
    request_exceptions.apply_remove_paths(
        template_kwargs_copy,
        remove_paths,
        use_kwargs_fallback=True,
    )
    request_template["sdk_call"] = {"args": list(template_args), "kwargs": template_kwargs_copy}

    remaining_placeholders = external_api_backend.list_placeholders(request_template)
    canonical_allowed = {external_api_backend.canonical_param_name(name) for name in remaining_placeholders}

    filtered: dict[str, Any] = {}
    for key, value in used_values.items():
        if external_api_backend.canonical_param_name(key) in canonical_allowed:
            filtered[key] = value
    return filtered

def _remove_none_values(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned: dict[str, Any] = {}
        for key, child in value.items():
            if child is None:
                continue
            result = _remove_none_values(child)
            if result == {} or result == []:
                continue
            cleaned[key] = result
        return cleaned
    if isinstance(value, list):
        cleaned_list = []
        for child in value:
            if child is None:
                continue
            result = _remove_none_values(child)
            if result == {} or result == []:
                continue
            cleaned_list.append(result)
        return cleaned_list
    if isinstance(value, tuple):
        cleaned_tuple = []
        for child in value:
            if child is None:
                continue
            result = _remove_none_values(child)
            if result == {} or result == []:
                continue
            cleaned_tuple.append(result)
        return tuple(cleaned_tuple)
    return value

def render_from_schema(spec: dict[str, Any], values: dict[str, Any] | None = None):
    used_values = _build_values(spec, values)
    rendered = external_api_backend.build_request(spec, used_values)

    if KlingRequestBuilder.is_kling_schema(spec):
        sdk_call_data = rendered.sdk_call if isinstance(rendered.sdk_call, dict) else {}
        sdk_kwargs = sdk_call_data.get("kwargs") if isinstance(sdk_call_data.get("kwargs"), dict) else {}
        arguments = sdk_kwargs.get("arguments") if isinstance(sdk_kwargs.get("arguments"), dict) else {}
        model_type_value = used_values.get("model_type")
        if KlingRequestBuilder.supports_multi_inputs_for_model_type(model_type_value):
            multi_prompt_value = arguments.get("multi_prompt")
            if isinstance(multi_prompt_value, list) and len(multi_prompt_value) > 0:
                arguments["shot_type"] = "customize"
                total_duration = KlingRequestBuilder.summarize_multi_prompt_duration(multi_prompt_value, default_duration=5)
                if total_duration is not None:
                    arguments["duration"] = total_duration
            else:
                arguments.pop("shot_type", None)
        else:
            arguments.pop("multi_prompt", None)
            arguments.pop("elements", None)
            arguments.pop("shot_type", None)

    request_exclusions = rendered.request_exclusions if isinstance(rendered.request_exclusions, list) else []
    sdk_call_data = rendered.sdk_call if isinstance(rendered.sdk_call, dict) else {}
    _, rendered_kwargs = external_api_backend.normalize_sdk_call(sdk_call_data)
    matched_remove_paths = request_exceptions.collect_matching_remove_paths(
        dict(rendered_kwargs),
        request_exclusions,
        use_kwargs_fallback=True,
        match_context=used_values,
    )

    filtered_used_values = _filter_used_values_from_template(spec, used_values, matched_remove_paths)
    filtered_used_values = {k: v for k, v in filtered_used_values.items() if v is not None}

    if matched_remove_paths and isinstance(rendered.sdk_call, dict):
        sdk_kwargs = rendered.sdk_call.get("kwargs")
        if isinstance(sdk_kwargs, dict):
            request_exceptions.apply_remove_paths(sdk_kwargs, matched_remove_paths, use_kwargs_fallback=True)

    rendered.headers = _remove_none_values(rendered.headers) if isinstance(rendered.headers, dict) else rendered.headers
    rendered.query = _remove_none_values(rendered.query) if isinstance(rendered.query, dict) else rendered.query
    rendered.body = _remove_none_values(rendered.body)
    rendered.sdk_call = _remove_none_values(rendered.sdk_call)

    return rendered, filtered_used_values