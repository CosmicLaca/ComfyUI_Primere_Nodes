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
import numpy as np

from ..components.API import api_json_to_requestbody
from ..components.API import external_api_backend
from ..components.API import api_schema_registry

class PrimereApiProcessor:
    CATEGORY = TREE_API
    RETURN_TYPES = ("APICLIENT", "STRING", "TUPLE", "TUPLE", "TUPLE", "TUPLE", "IMAGE")
    RETURN_NAMES = ("CLIENT", "PROVIDER", "SCHEMA", "RENDERED", "API_SCHEMAS", "API_RESULT", "RESULT_IMAGE")
    FUNCTION = "process_uniapi"

    API_RESULT = api_helper.get_api_config("apiconfig.json")
    API_SCHEMAS_RAW = utility.json2tuple(os.path.join(PRIMERE_ROOT, 'front_end', 'api_schemas.json'))
    API_SCHEMA_REGISTRY = api_schema_registry.normalize_registry(API_SCHEMAS_RAW)

    @classmethod
    def INPUT_TYPES(cls):
        cls.required_inputs = {
            "processor": ("BOOLEAN", {"default": True, "label_on": "ON", "label_off": "OFF"}),
            "debug_mode": ("BOOLEAN", {"default": False, "label_on": "DEBUG ONLY", "label_off": "PRODUCTION"}),
            "api_provider": (external_api_backend.provider_list(cls),),
            "api_service": (external_api_backend.service_list(cls),),
            "prompt": ("STRING", {"forceInput": True}),
            "batch": ("INT", {"default": 1, "max": 10, "min": 1, "step": 1})
        }

        cls.optional_inputs = {
            "negative_prompt": ("STRING", {"default": None, "forceInput": True}),
            "reference_images": ("IMAGE", {"default": None, "forceInput": True}),
            "first_image": ("IMAGE", {"default": None, "forceInput": True}),
            "last_image": ("IMAGE", {"default": None, "forceInput": True}),
            "width": ("INT", {"default": 1024, "max": 8192, "min": 64, "step": 64, "forceInput": True}),
            "height": ("INT", {"default": 1024, "max": 8192, "min": 64, "step": 64, "forceInput": True}),
            "aspect_ratio": ("STRING", {"forceInput": True, "default": "1:1"}),
            "seed": ("INT", {"default": 1, "min": 0, "max": (2 ** 32) - 1, "forceInput": True})
        }

        hidden_inputs = {
            "extra_pnginfo": "EXTRA_PNGINFO",
            "prompt_extra": "PROMPT"
        }

        return {"required": cls.required_inputs, "optional": cls.optional_inputs, "hidden": hidden_inputs}

    def process_uniapi(self, processor, api_provider, api_service, prompt, negative_prompt = None, batch = 1, reference_images = None, first_image = None, last_image = None, width = 1024, height = 1024, aspect_ratio = '1:1', seed = None, debug_mode = False, **kwargs):
        API_SCHEMAS_PATH = os.path.join(PRIMERE_ROOT, 'front_end', 'api_schemas.json')
        API_CONFIG_PATH = os.path.join(PRIMERE_ROOT, 'json', 'apiconfig.json')
        API_SCHEMA_REGISTRY = api_schema_registry.load_and_validate_api_schema_registry(API_SCHEMAS_PATH, API_CONFIG_PATH)

        img_binary_api = []

        WORKFLOWDATA = kwargs['extra_pnginfo']['workflow']['nodes']
        custom_values = utility.getInputsFromWorkflowByNode(WORKFLOWDATA, 'PrimereApiProcessor', kwargs['prompt_extra'])

        custom_user_inputs = {k: v for k, v in custom_values.items() if k not in self.required_inputs}
        custom_user_inputs = {k: v for k, v in custom_user_inputs.items() if k not in self.optional_inputs}
        # return (None, api_provider, None, custom_user_inputs, None, None, None)
        del kwargs['extra_pnginfo']
        del kwargs['prompt_extra']

        if not processor:
            return (None, api_provider, None, None, None, None, reference_images)

        config_json = self.API_RESULT
        client, api_provider = api_helper.create_api_client(api_provider, config_json)

        schema, selected_service = api_schema_registry.get_schema(API_SCHEMA_REGISTRY, api_provider, api_service)
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
        schema_import_modules = schema.get("import_modules", []) if isinstance(schema, dict) else []
        if not isinstance(schema_import_modules, list):
            raise RuntimeError("Schema key 'import_modules' must be a list of import statements.")
        schema["import_modules"] = schema_import_modules

        selected_parameters = {"prompt": prompt}
        selected_parameters = {"width": width}
        selected_parameters = {"height": height}

        local_inputs = locals()
        required_keys = set(self.required_inputs.keys()) if isinstance(getattr(self, "required_inputs", None), dict) else set()
        optional_keys = set(self.optional_inputs.keys()) if isinstance(getattr(self, "optional_inputs", None), dict) else set()
        reserved_keys = {"processor", "api_provider", "api_service"}

        for key in (required_keys | optional_keys):
            if key in reserved_keys:
                continue
            if key in local_inputs and local_inputs[key] not in (None, ""):
                selected_parameters[key] = local_inputs[key]

        if isinstance(custom_user_inputs, dict):
            for key, value in custom_user_inputs.items():
                if value not in (None, ""):
                    selected_parameters[key] = value

        if aspect_ratio not in (None, ""):
            selected_aspect_ratio = aspect_ratio
            schema_aspect_ratios = external_api_backend.schema_possible_values(self, api_provider, (selected_service or api_service), "aspect_ratio",)
            if len(schema_aspect_ratios) > 0:
                selected_aspect_ratio = external_api_backend.closest_valid_ratio(aspect_ratio, schema_aspect_ratios)
            selected_parameters["aspect_ratio"] = selected_aspect_ratio

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
                        if api_provider == 'FAL':
                            ref_url = client.upload_file(TEMP_FILE_REF)
                            img_binary_api.append(ref_url)
                        else:
                            img_binary_api.append(TEMP_FILE_REF)

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
            rendered_payload = external_api_backend.redact_reference_images(rendered_payload)
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
        result_image = None
        batch = max(1, int(batch))
        sdk_context = {}
        response_url = None
        loaded_client = client

        try:
            if rendered.method.upper() == "SDK":
                context = {"client": client}
                allowed_roots = {"client"}
                imported_context, imported_roots = external_api_backend.load_import_modules(schema_import_modules)
                context.update(imported_context)
                allowed_roots.update(imported_roots)
                sdk_context = dict(context)
                sdk_call_data = rendered.sdk_call if isinstance(rendered.sdk_call, dict) else {}
                sdk_args = sdk_call_data.get("args", []) if isinstance(sdk_call_data, dict) else []
                if isinstance(sdk_args, list) and len(sdk_args) > 0 and isinstance(sdk_args[0], str):
                    response_url = sdk_args[0]

                endpoint_root = rendered.endpoint.split(".", 1)[0] if isinstance(rendered.endpoint, str) and "." in rendered.endpoint else "client"
                loaded_client = context.get(endpoint_root, client)

                if debug_mode:
                    return (loaded_client, api_provider, schema, rendered_payload, None, api_result, None)
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
            # result_image = external_api_backend.apply_response_handler(schema, api_result, provider=api_provider, service=(selected_service or api_service))
            response_context = {"response_url": response_url, "call_url": response_url, "loaded_client": loaded_client, "client": client, "sdk_context": sdk_context}
            result_image = external_api_backend.apply_response_handler(schema, api_result, provider=api_provider, service=(selected_service or api_service), response_context=response_context)

        return (client, api_provider, schema, rendered_payload, api_schemas, api_result, result_image)