from __future__ import annotations

from ..components.tree import TREE_API
from ..components.tree import PRIMERE_ROOT
import os
from ..components import utility
from ..components.API import api_helper
import folder_paths

import random
import argparse
import json
import copy
from pathlib import Path
from typing import Any
import requests
import sys
from PIL import Image
from io import BytesIO
import numpy as np
import torch
import comfy.utils

from ..components.API import api_json_to_requestbody
from ..components.API import external_api_backend
from ..components.API import api_schema_registry

class PrimereApiProcessor:
    CATEGORY = TREE_API
    RETURN_TYPES = ("APICLIENT", "STRING", "TUPLE", "TUPLE", "TUPLE", "TUPLE", "IMAGE")
    RETURN_NAMES = ("CLIENT", "PROVIDER", "SCHEMA", "RENDERED", "API_SCHEMAS", "API_RESULT", "RESULT_IMAGE")
    FUNCTION = "process_uniapi"

    API_RESULT = api_helper.get_api_config("apiconfig.json")
    API_SCHEMAS_RAW = utility.json2tuple(os.path.join(PRIMERE_ROOT, 'json', 'api_schemas.json'))
    API_SCHEMA_REGISTRY = api_schema_registry.normalize_registry(API_SCHEMAS_RAW)
    NANOBANANA_ASPECT_RATIOS = ["1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"]
    VEO_ASPECT_RATIOS = ["9:16", "16:9"]

    @classmethod
    def _default_provider_service(cls):
        if isinstance(cls.API_SCHEMA_REGISTRY, dict) and len(cls.API_SCHEMA_REGISTRY) > 0:
            first_provider = next(iter(cls.API_SCHEMA_REGISTRY))
            provider_services = cls.API_SCHEMA_REGISTRY.get(first_provider, {})
            if isinstance(provider_services, dict) and len(provider_services) > 0:
                first_service = next(iter(provider_services))
                return first_provider, first_service
            return first_provider, "default"

        providers = list(cls.API_RESULT.keys()) if isinstance(cls.API_RESULT, dict) else []
        if len(providers) > 0:
            return providers[0], "default"

        return "custom", "default"

    @classmethod
    def _provider_list(cls):
        default_provider, _ = cls._default_provider_service()
        config_providers = []
        if isinstance(cls.API_RESULT, dict):
            config_providers = [str(provider) for provider in cls.API_RESULT.keys()]
        schema_provider_set = set()
        if isinstance(cls.API_SCHEMA_REGISTRY, dict):
            schema_provider_set = {str(provider) for provider in cls.API_SCHEMA_REGISTRY.keys()}

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

    @classmethod
    def _service_list(cls):
        default_provider, default_service = cls._default_provider_service()
        services = []
        if isinstance(cls.API_SCHEMA_REGISTRY, dict):
            provider_services = cls.API_SCHEMA_REGISTRY.get(default_provider, {})
            if isinstance(provider_services, dict):
                services = [str(service) for service in provider_services.keys()]
        ordered_services = [default_service]
        for service in services:
            if service not in ordered_services:
                ordered_services.append(service)

        return ordered_services

    @classmethod
    def _parameter_options(cls):
        default_provider, default_service = cls._default_provider_service()
        provider_services = cls.API_SCHEMA_REGISTRY.get(default_provider, {}) if isinstance(cls.API_SCHEMA_REGISTRY, dict) else {}
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

    @classmethod
    def INPUT_TYPES(cls):
        required_inputs = {
            "processor": ("BOOLEAN", {"default": True, "label_on": "ON", "label_off": "OFF"}),
            "api_provider": (cls._provider_list(),),
            "api_service": (cls._service_list(),),
            "prompt": ("STRING", {"forceInput": True}),
            "batch": ("INT", {"default": 1, "max": 10, "min": 1, "step": 1})
        }

        optional_inputs = {
            "negative_prompt": ("STRING", {"default": None, "forceInput": True}),
            "reference_images": ("IMAGE", {"default": None, "forceInput": True}),
            "first_image": ("IMAGE", {"default": None, "forceInput": True}),
            "last_image": ("IMAGE", {"default": None, "forceInput": True}),
            "width": ("INT", {"default": 1024, "max": 8192, "min": 64, "step": 64, "forceInput": True}),
            "height": ("INT", {"default": 1024, "max": 8192, "min": 64, "step": 64, "forceInput": True}),
            "aspect_ratio": ("STRING", {"forceInput": True, "default": "1:1"}),
            "seed": ("INT", {"default": 1, "min": 0, "max": (2 ** 32) - 1, "forceInput": True})
        }

        for key, values in cls._parameter_options().items():
            required_inputs[key] = (values,)

        return {"required": required_inputs, "optional": optional_inputs}

    @classmethod
    def _parse_ratio(cls, value):
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

    @classmethod
    def _closest_valid_ratio(cls, value, valid_ratios):
        if not isinstance(valid_ratios, list) or len(valid_ratios) == 0:
            return value

        normalized_valid = [str(ratio) for ratio in valid_ratios]
        candidate = str(value).strip() if value is not None else ""
        if candidate in normalized_valid:
            return candidate

        candidate_ratio = cls._parse_ratio(candidate)
        if candidate_ratio is None:
            return normalized_valid[0]

        best_value = normalized_valid[0]
        best_diff = float("inf")
        for ratio_text in normalized_valid:
            parsed_ratio = cls._parse_ratio(ratio_text)
            if parsed_ratio is None:
                continue
            diff = abs(parsed_ratio - candidate_ratio)
            if diff < best_diff:
                best_diff = diff
                best_value = ratio_text

        return best_value

    def process_uniapi(self, processor, api_provider, api_service, prompt, negative_prompt = None, batch = 1, reference_images = None, first_image = None, last_image = None, width = 1024, height = 1024, aspect_ratio = '1:1', seed = None, **kwargs):
        img_binary_api = []

        if reference_images is not None:
            if (type(reference_images).__name__ == "list" or type(reference_images).__name__ == "Tensor") and len(reference_images) > 0:
                source_images = []
                if type(reference_images).__name__ == "list":
                    source_images = reference_images
                else:
                    source_images.append(reference_images)

                for single_image in source_images:
                    r1 = random.randint(1000, 9999)
                    if single_image is not None and type(single_image).__name__ == "Tensor":
                        ref_image = (single_image[0].numpy() * 255).astype(np.uint8)
                        ref_file = Image.fromarray(ref_image)
                        TEMP_FILE_REF = os.path.join(folder_paths.temp_directory, f"{api_provider}_edit_{r1}.png")
                        ref_file.save(TEMP_FILE_REF, format="PNG")
                        if api_provider == "Gemini":
                            gemini_image_data = Image.open(TEMP_FILE_REF)
                            img_binary_api.append(gemini_image_data)
                        else:
                            img_binary_api.append(TEMP_FILE_REF)

        if not processor:
            return (None, api_provider, None, None, None, None, reference_images)

        config_json = self.API_RESULT
        client, api_provider = api_helper.create_api_client(api_provider, config_json)

        schema, selected_service = api_schema_registry.get_schema(self.API_SCHEMA_REGISTRY, api_provider, api_service)
        if schema is None:
            schema = {
                "provider": api_provider,
                "service": api_service,
                "request": {
                    "method": "SDK",
                    "endpoint": "",
                    "sdk_call": {"args": [], "kwargs": {}},
                },
            }
            selected_service = api_service

        schema["provider"] = api_provider
        schema["service"] = selected_service or api_service

        selected_parameters = {"prompt": prompt}
        if aspect_ratio not in (None, ""):
            selected_aspect_ratio = aspect_ratio
            if api_provider == "Gemini" and (selected_service or api_service) == "Nanobanana":
                selected_aspect_ratio = self._closest_valid_ratio(aspect_ratio, self.NANOBANANA_ASPECT_RATIOS)
            selected_parameters["aspect_ratio"] = selected_aspect_ratio

        if len(img_binary_api) > 0:
            selected_parameters["reference_images"] = img_binary_api

        possible_parameters = schema.get("possible_parameters", {}) if isinstance(schema, dict) else {}
        if isinstance(possible_parameters, dict):
            for key in possible_parameters.keys():
                if key in kwargs and kwargs[key] not in (None, ""):
                    selected_parameters[key] = kwargs[key]

        rendered, used_values = api_json_to_requestbody.render_from_schema(schema, selected_parameters)
        # rendered_payload = rendered.__dict__
        rendered_payload = copy.deepcopy(rendered.__dict__)
        if len(img_binary_api) > 0:
            def _redact_reference_images(node):
                if isinstance(node, dict):
                    sanitized = {}
                    for key, value in node.items():
                        if key == "reference_images":
                            sanitized[key] = "[reference_images omitted]"
                        else:
                            sanitized[key] = _redact_reference_images(value)
                    return sanitized
                if isinstance(node, list):
                    return [_redact_reference_images(value) for value in node]
                return node

            rendered_payload = _redact_reference_images(rendered_payload)
            sdk_call = rendered_payload.get("sdk_call")
            if isinstance(sdk_call, dict):
                sdk_kwargs = sdk_call.get("kwargs")
                if isinstance(sdk_kwargs, dict):
                    contents = sdk_kwargs.get("contents")
                    if isinstance(contents, list):
                        sanitized_contents = []
                        for content_item in contents:
                            if isinstance(content_item, list) and len(content_item) > 0 and all(isinstance(item, (Image.Image, str)) for item in content_item):
                                sanitized_contents.append("[reference_images omitted]")
                            else:
                                sanitized_contents.append(content_item)
                        sdk_kwargs["contents"] = sanitized_contents

        api_result = None
        api_error = None
        final_batch_img = []
        result_image = None
        image_list = []
        batch = max(1, int(batch))

        try:
            if rendered.method.upper() == "SDK":
                context = {"client": client}
                allowed_roots = {"client"}

                try:
                    from google.genai import types as genai_types
                    context["types"] = genai_types
                    allowed_roots.add("types")
                except Exception:
                    pass

                api_result = external_api_backend.execute_sdk_request(rendered, context, allowed_roots)
            else:
                import requests

                response = requests.request(
                    method=rendered.method,
                    url=rendered.endpoint,
                    headers=rendered.headers,
                    params=rendered.query,
                    json=rendered.body,
                    timeout=60,
                )
                api_result = {
                    "status_code": response.status_code,
                    "ok": response.ok,
                    "text": response.text,
                }
        except Exception as e:
            api_error = str(e)

        selected_parameters_output = {k: v for k, v in selected_parameters.items() if k != "reference_images"}

        api_schemas = (
            {
                "schema": schema,
                "selected_parameters": selected_parameters_output,
                "used_values": used_values,
                "selected_service": selected_service,
                # "rendered": rendered.__dict__,
                "rendered": rendered_payload,
                "api_result": api_result,
                "api_error": api_error,
            },
        )

        if api_error is not None:
            raise RuntimeError(f"API call failed for {api_provider}/{selected_service}: {api_error}")

        if api_error is None:
            if api_result.candidates[0].content is not None and api_result.candidates[0].content.parts is not None:
                image_parts = [
                    part.inline_data.data
                    for part in api_result.candidates[0].content.parts
                    if part.inline_data
                ]
                if image_parts:
                    result_image = Image.open(BytesIO(image_parts[0]))
                    if result_image is not None:
                        result_image = result_image.convert("RGB")
                        result_image = np.array(result_image).astype(np.float32) / 255.0
                        result_image = torch.from_numpy(result_image)[None,]
                        final_batch_img.append(result_image)

                if type(final_batch_img).__name__ == "list" and len(final_batch_img) > 1:
                    image_list = final_batch_img
                    single_image_start = final_batch_img[0]
                    batch_count = 0
                    s = None
                    for single_image in final_batch_img:
                        if (batch_count + 1) < len(final_batch_img):
                            current_single_image = final_batch_img[batch_count + 1]
                            if single_image_start.shape[1:] != current_single_image.shape[1:]:
                                current_single_image = comfy.utils.common_upscale(current_single_image.movedim(-1, 1), single_image_start.shape[2], single_image_start.shape[1], "bilinear", "center").movedim(1, -1)
                            batch_count = batch_count + 1
                            if s is not None:
                                single_image = s
                            s = torch.cat((current_single_image, single_image), dim=0)
                            result_image = s

        return (client, api_provider, schema, rendered_payload, api_schemas, api_result, result_image)