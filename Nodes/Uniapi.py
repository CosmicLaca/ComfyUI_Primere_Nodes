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
import datetime
from pathlib import Path
from typing import Any
import requests
import sys
from PIL import Image
from io import BytesIO
import numpy as np

from ..components.API import api_json_to_requestbody
from ..components.API import external_api_backend
from ..components.API import api_schema_registry
from ..components import file_output
from server import PromptServer

class PrimereApiProcessor:
    CATEGORY = TREE_API
    RETURN_TYPES = ("IMAGE", "APICLIENT", "STRING", "TUPLE", "TUPLE", "TUPLE", "TUPLE", "TUPLE", "TUPLE")
    RETURN_NAMES = ("RESULT", "CLIENT", "PROVIDER", "SCHEMA", "RENDERED", "RAW_PAYLOAD", "REQUEST_BODY", "API_SCHEMAS", "API_RESULT")
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
            "batch": ("INT", {"default": 1, "max": 10, "min": 1, "step": 1}),
            "auto_save_result": ("BOOLEAN", {"default": False, "label_on": "Save result", "label_off": "Don't save result"}),
            "output_path": ("STRING", {"default": '[time(%Y-%m-%d)]', "multiline": False}),
            "subpath": (["None", "Dev", "Test", "Serie", "Production", "Preview", "NewModel", "Project", "Portfolio", "Civitai", "Behance", "Facebook", "Instagram", "Character", "Style", "Product", "Fun", "SFW", "NSFW"], {"default": "Project"}),
            "add_provider_to_path": ("BOOLEAN", {"default": False}),
            "add_service_to_path": ("BOOLEAN", {"default": False}),
            "add_model_to_path": ("BOOLEAN", {"default": False}),
            "filename_prefix": ("STRING", {"default": "API"}),
            "filename_delimiter": ("STRING", {"default": "_"}),
            "add_date_to_filename": ("BOOLEAN", {"default": True}),
            "add_time_to_filename": ("BOOLEAN", {"default": True}),
            "filename_number_padding": ("INT", {"default": 2, "min": 1, "max": 9, "step": 1}),
            "filename_number_start": ("BOOLEAN", {"default": False}),
            "image_extension": ([ext.lstrip('.') for ext in file_output.ALLOWED_EXT], {"default": "jpg"}),
            "image_quality": ("INT", {"default": 95, "min": 1, "max": 100, "step": 1}),
            "save_data_to_json": ("BOOLEAN", {"default": False}),
            "save_data_to_txt": ("BOOLEAN", {"default": False}),
        }

        cls.optional_inputs = {
            "negative_prompt": ("STRING", {"default": None, "forceInput": True}),
            "reference_images": ("IMAGE", {"default": None, "forceInput": True}),
            "first_image": ("IMAGE", {"default": None, "forceInput": True}),
            "last_image": ("IMAGE", {"default": None, "forceInput": True}),
            "frontal_image": ("IMAGE", {"default": None, "forceInput": True}),
            "reference_video": ("VIDEO", {"default": None, "forceInput": True}),
            "width": ("INT", {"default": 1024, "max": 8192, "min": 64, "step": 64, "forceInput": True}),
            "height": ("INT", {"default": 1024, "max": 8192, "min": 64, "step": 64, "forceInput": True}),
            "aspect_ratio": ("STRING", {"forceInput": True, "default": "1:1"}),
            "seed": ("INT", {"default": 1, "min": 0, "max": (2 ** 32) - 1, "forceInput": True})
        }

        hidden_inputs = {
            "extra_pnginfo": "EXTRA_PNGINFO",
            "prompt_extra": "PROMPT",
            "unique_id": "UNIQUE_ID",
        }

        return {"required": cls.required_inputs, "optional": cls.optional_inputs, "hidden": hidden_inputs}

    def process_uniapi(self, processor, api_provider, api_service, prompt, negative_prompt = None, batch = 1, reference_images = None, first_image = None, last_image = None, frontal_image = None, reference_video = None, width = 1024, height = 1024, aspect_ratio = '1:1', seed = None, debug_mode = False, unique_id = None, **kwargs):
        API_SCHEMAS_PATH = os.path.join(PRIMERE_ROOT, 'front_end', 'api_schemas.json')
        API_CONFIG_PATH = os.path.join(PRIMERE_ROOT, 'json', 'apiconfig.json')
        API_SCHEMA_REGISTRY = api_schema_registry.load_and_validate_api_schema_registry(API_SCHEMAS_PATH, API_CONFIG_PATH)

        img_binary_api = None

        WORKFLOWDATA = kwargs['extra_pnginfo']['workflow']['nodes']
        custom_values = utility.getInputsFromWorkflowByNode(WORKFLOWDATA, 'PrimereApiProcessor', kwargs['prompt_extra'])

        custom_user_inputs = {k: v for k, v in custom_values.items() if k not in self.required_inputs}
        custom_user_inputs = {k: v for k, v in custom_user_inputs.items() if k not in self.optional_inputs}
        del kwargs['extra_pnginfo']
        del kwargs['prompt_extra']

        if not processor:
            return (None, None, api_provider, None, None, None, None, None, None)

        config_json = self.API_RESULT
        _requested_provider = api_provider
        client, api_provider = api_helper.create_api_client(api_provider, config_json)
        if client is None:
            raise RuntimeError(f"Unknown or unconfigured API provider '{_requested_provider}'. Check apiconfig.json.")

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

        imported_context, imported_roots = external_api_backend.load_import_modules(schema_import_modules)
        endpoint_value = (((schema.get("request") or {}).get("endpoint")) if isinstance(schema, dict) else "") or ""
        endpoint_root = endpoint_value.split(".", 1)[0] if isinstance(endpoint_value, str) and "." in endpoint_value else "client"
        loaded_client_for_upload = imported_context.get(endpoint_root, client)

        ref_type_name = type(reference_images).__name__ if reference_images is not None else ""
        if ref_type_name in {"list", "Tensor"} and len(reference_images) > 0:
            source_images = reference_images if ref_type_name == "list" else [reference_images]
            img_binary_api = []

            for single_image in source_images:
                r1 = random.randint(1000, 9999)
                if single_image is not None and type(single_image).__name__ == "Tensor":
                    ref_image = (single_image[0].numpy() * 255).astype(np.uint8)
                    ref_file = Image.fromarray(ref_image)
                    TEMP_FILE_REF = os.path.join(folder_paths.temp_directory, f"{api_provider}_edit_{r1}.png")
                    ref_file.save(TEMP_FILE_REF, format="PNG")
                    handler_context = {
                        "img_binary_api": img_binary_api,
                        "single_image": single_image,
                        "source_images": source_images,
                        "temp_file_ref": TEMP_FILE_REF,
                        "loaded_client_for_upload": loaded_client_for_upload,
                    }
                    if not debug_mode:
                        img_binary_api = external_api_backend.apply_reference_images_handler(schema, api_provider, handler_context)
                        if not isinstance(img_binary_api, list):
                            break
                    else:
                        img_binary_api = f'Debug mode, data or view of [{len(reference_images)}] reference images ignored.'
        else:
            if debug_mode:
                img_binary_api = 'Reference images off. Please check the source.'

        selected_parameters = {"prompt": prompt, "width": width, "height": height}

        local_inputs = locals()
        required_keys = set(self.required_inputs.keys()) if isinstance(getattr(self, "required_inputs", None), dict) else set()
        optional_keys = set(self.optional_inputs.keys()) if isinstance(getattr(self, "optional_inputs", None), dict) else set()
        reserved_keys = {"processor", "api_provider", "api_service", "reference_images"}

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

        if isinstance(img_binary_api, dict) and len(img_binary_api) > 0:
            selected_parameters.update(img_binary_api)
        elif isinstance(img_binary_api, list) and len(img_binary_api) > 0:
            selected_parameters["reference_images"] = img_binary_api

        first_image_source = first_image if first_image is not None else (reference_images[0] if isinstance(reference_images, list) else reference_images) if reference_images is not None else None

        single_ref_inputs = [
            ("first_image", first_image_source),
            ("last_image", last_image),
            ("frontal_image", frontal_image),
            ("reference_video", reference_video),
        ]
        if not debug_mode:
            for ref_key, ref_source in single_ref_inputs:
                if ref_source is None:
                    continue
                ref_result = external_api_backend.apply_reference_images_handler(
                    schema, api_provider,
                    {"img_binary_api": ref_source, "loaded_client_for_upload": loaded_client_for_upload, "target_key": ref_key}
                )
                if isinstance(ref_result, dict) and ref_result:
                    selected_parameters.update(ref_result)

        possible_parameters = schema.get("possible_parameters", {}) if isinstance(schema, dict) else {}
        if isinstance(possible_parameters, dict):
            for key in possible_parameters.keys():
                if key in kwargs and kwargs[key] not in (None, ""):
                    selected_parameters[key] = kwargs[key]

        selected_parameters = external_api_backend.apply_parameter_constraints(selected_parameters, schema)
        rendered, used_values = api_json_to_requestbody.render_from_schema(schema, selected_parameters)
        rendered_payload = copy.deepcopy(rendered.__dict__)
        if img_binary_api not in (None, "") and (not isinstance(img_binary_api, list) or len(img_binary_api) > 0):
            rendered_payload = external_api_backend.sanitize_api_debug_payload(rendered_payload)
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

        used_values_output = external_api_backend.sanitize_api_debug_payload(used_values)

        if isinstance(rendered.sdk_call, dict):
            raw_payload = external_api_backend.sanitize_api_debug_payload(copy.deepcopy(rendered.sdk_call.get("kwargs", {})))
        elif rendered.body is not None:
            raw_payload = external_api_backend.sanitize_api_debug_payload(copy.deepcopy(rendered.body))
        else:
            raw_payload = {}

        api_result = None
        api_error = None
        result_image = None
        save_bytes = None
        batch = max(1, int(batch))
        sdk_context = {}
        response_url = None
        loaded_client = client

        try:
            if rendered.method.upper() == "SDK":
                provider_config = config_json.get(api_provider, {}) if isinstance(config_json, dict) else {}
                provider_api_key = provider_config.get("APIKEY") if isinstance(provider_config, dict) else None
                context = {"client": client, "provider_api_key": provider_api_key}
                allowed_roots = {"client"}
                imported_context, imported_roots = external_api_backend.load_import_modules(schema_import_modules)
                context.update(imported_context)
                allowed_roots.update(imported_roots)

                auto_context, auto_roots = external_api_backend.build_sdk_context(rendered, client)
                for root_name in auto_roots:
                    if root_name not in context and root_name in auto_context:
                        context[root_name] = auto_context[root_name]
                        allowed_roots.add(root_name)

                sdk_context = dict(context)
                sdk_call_data = rendered.sdk_call if isinstance(rendered.sdk_call, dict) else {}
                sdk_args = sdk_call_data.get("args", []) if isinstance(sdk_call_data, dict) else []
                if isinstance(sdk_args, list) and len(sdk_args) > 0 and isinstance(sdk_args[0], str):
                    response_url = sdk_args[0]

                endpoint_root = rendered.endpoint.split(".", 1)[0] if isinstance(rendered.endpoint, str) and "." in rendered.endpoint else "client"
                loaded_client = context.get(endpoint_root, client)

                if debug_mode:
                    return (reference_images, loaded_client, api_provider, schema, rendered_payload, raw_payload, used_values_output, api_result, None)
                api_result = external_api_backend.execute_sdk_request(rendered, context, allowed_roots, match_context=used_values)
            else:
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
        selected_parameters_output = external_api_backend.sanitize_api_debug_payload(selected_parameters_output)

        api_result_debug = external_api_backend.sanitize_debug_value(api_result)
        api_schemas = (
            {
                "schema": schema,
                "selected_parameters": selected_parameters_output,
                "used_values": used_values_output,
                "selected_service": selected_service,
                "rendered": rendered_payload,
                "api_result": api_result_debug,
                "api_error": api_error,
            },
        )

        if api_error is not None:
            raise RuntimeError(f"API call failed for {api_provider}/{selected_service}: {api_error}")

        if api_error is None:
            response_context = {"response_url": response_url, "call_url": response_url, "loaded_client": loaded_client, "client": client, "sdk_context": sdk_context}
            handler_result = external_api_backend.apply_response_handler(schema, api_result, provider=api_provider, service=(selected_service or api_service), response_context=response_context)
            if isinstance(handler_result, list) and len(handler_result) == 2:
                result_image, save_bytes = handler_result
            else:
                result_image = handler_result

        # --- File naming and output path resolution ---
        auto_save_result = kwargs.get('auto_save_result', False)
        if auto_save_result and result_image is not None:
            if type(result_image).__name__ == "str":
                SAVED_IMAGE_PATH = os.path.join(PRIMERE_ROOT, 'front_end', 'images')
                SAVED_IMAGE = os.path.join(SAVED_IMAGE_PATH, "file_saved.jpg")
                result_image = utility.ImageLoaderFromPath(SAVED_IMAGE)

            output_path_input = kwargs.get('output_path', '[time(%Y-%m-%d)]')
            subpath = kwargs.get('subpath', 'None')
            add_provider_to_path = kwargs.get('add_provider_to_path', False)
            add_service_to_path = kwargs.get('add_service_to_path', False)
            filename_prefix = kwargs.get('filename_prefix', 'API')
            filename_delimiter = kwargs.get('filename_delimiter', '_')
            add_date_to_filename = kwargs.get('add_date_to_filename', True)
            add_time_to_filename = kwargs.get('add_time_to_filename', True)
            filename_number_padding = kwargs.get('filename_number_padding', 2)
            filename_number_start = kwargs.get('filename_number_start', False)
            image_extension = kwargs.get('image_extension', 'jpg')
            image_quality = kwargs.get('image_quality', 95)
            save_data_to_json = kwargs.get('save_data_to_json', False)
            save_data_to_txt = kwargs.get('save_data_to_txt', False)
            add_model_to_path = kwargs.get('add_model_to_path', False)

            model_subdir = None
            if add_model_to_path:
                model_subdir = next((custom_values[k] for k in ('model', 'model_name', 'version') if custom_values.get(k)), None)
            if model_subdir:
                model_subdir = file_output.sanitize_path_part(model_subdir)

            if not os.path.isabs(output_path_input):
                output_path_input = file_output.sanitize_path_part(output_path_input)
            filename_prefix = file_output.sanitize_path_part(filename_prefix)

            subdirs = []
            if add_provider_to_path and api_provider:
                subdirs.append(file_output.sanitize_path_part(api_provider))
            if add_service_to_path and (selected_service or api_service):
                subdirs.append(file_output.sanitize_path_part(selected_service or api_service))
            if model_subdir:
                subdirs.append(model_subdir)
            if subpath and subpath != 'None' and subpath.strip():
                subdirs.append(file_output.sanitize_path_part(subpath))

            output_file, json_file, txt_file = file_output.resolve_output_file(
                output_path_input, folder_paths.output_directory, subdirs,
                filename_prefix, filename_delimiter,
                add_date_to_filename, add_time_to_filename,
                filename_number_padding, filename_number_start, image_extension,
            )

            Path(folder_paths.temp_directory).mkdir(parents=True, exist_ok=True)
            try:
                saved_path = file_output.save_bytes_to_file(save_bytes, output_file, image_extension, image_quality, folder_paths.temp_directory)

                save_data = {
                    "provider": api_provider,
                    "service": selected_service or api_service,
                    "selected_parameters": selected_parameters_output,
                    "used_values": used_values_output,
                    # "rendered": rendered_payload,
                    "raw_payload": raw_payload,
                    # "api_result": api_result_debug,
                }
                file_output.save_metadata(save_data, json_file, txt_file, save_data_to_json, save_data_to_txt, used_values_output)

                PromptServer.instance.send_sync("primere.save_result", {
                    "node_id": unique_id,
                    "status": "success",
                    "path": saved_path,
                })
            except Exception as save_error:
                PromptServer.instance.send_sync("primere.save_result", {
                    "node_id": unique_id,
                    "status": "failed",
                    "error": str(save_error),
                })

        return (result_image, client, api_provider, schema, rendered_payload, raw_payload, used_values_output, api_schemas, api_result_debug)