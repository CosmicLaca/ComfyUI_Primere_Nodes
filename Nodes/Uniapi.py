from __future__ import annotations

from ..components.tree import TREE_API
from ..components.tree import PRIMERE_ROOT
import os
from ..components import utility
from ..components.API import api_helper

import argparse
import json
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

    @classmethod
    def _provider_list(cls):
        providers = list(cls.API_RESULT.keys()) if isinstance(cls.API_RESULT, dict) else []
        if not providers:
            providers = api_schema_registry.list_providers(cls.API_SCHEMA_REGISTRY)
        return providers or ["custom"]

    @classmethod
    def _service_list(cls):
        all_services = sorted({
            service
            for provider in cls.API_SCHEMA_REGISTRY
            for service in cls.API_SCHEMA_REGISTRY.get(provider, {})
        })
        return all_services or ["default"]

    @classmethod
    def _parameter_options(cls):
        options: dict[str, list[str]] = {}
        for provider_services in cls.API_SCHEMA_REGISTRY.values():
            for schema in provider_services.values():
                possible = schema.get("possible_parameters", {}) if isinstance(schema, dict) else {}
                if not isinstance(possible, dict):
                    continue
                for key, values in possible.items():
                    key_name = str(key)
                    if key_name == "prompt":
                        continue
                    value_list = [str(v) for v in values] if isinstance(values, list) else []
                    if len(value_list) == 0:
                        value_list = [f"default_{key_name}"]
                    existing = options.get(key_name, [])
                    merged = sorted(set(existing + value_list))
                    options[key_name] = merged
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
            "reference_images": ("IMAGE", {"default": None, "forceInput": True}),
            "width": ("INT", {"default": 1024, "max": 8192, "min": 64, "step": 64, "forceInput": True}),
            "height": ("INT", {"default": 1024, "max": 8192, "min": 64, "step": 64, "forceInput": True}),
            "aspect_ratio": ("STRING", {"forceInput": True, "default": "1:1"}),
            "seed": ("INT", {"default": 1, "min": 0, "max": (2 ** 32) - 1, "forceInput": True})
        }

        for key, values in cls._parameter_options().items():
            required_inputs[key] = (values,)

        return {"required": required_inputs, "optional": optional_inputs}

    def process_uniapi(self, processor, api_provider, api_service, prompt, batch = 1, reference_images = None, width = 1024, height = 1024, aspect_ratio = '1:1', seed = None, **kwargs):
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
        possible_parameters = schema.get("possible_parameters", {}) if isinstance(schema, dict) else {}
        if isinstance(possible_parameters, dict):
            for key in possible_parameters.keys():
                if key in kwargs and kwargs[key] not in (None, ""):
                    selected_parameters[key] = kwargs[key]

        rendered, used_values = api_json_to_requestbody.render_from_schema(schema, selected_parameters)

        api_result = None
        api_error = None
        final_batch_img = []
        result_image = None
        image_list = []

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

        api_schemas = (
            {
                "schema": schema,
                "selected_parameters": selected_parameters,
                "used_values": used_values,
                "selected_service": selected_service,
                "rendered": rendered.__dict__,
                "api_result": api_result,
                "api_error": api_error,
            },
        )

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

        return (client, api_provider, schema, rendered, api_schemas, api_result, result_image)

