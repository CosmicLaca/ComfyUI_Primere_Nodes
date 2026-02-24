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

class PrimereApiProcessor:
    CATEGORY = TREE_API
    RETURN_TYPES = ("APICLIENT", "STRING", "TUPLE", "TUPLE", "TUPLE", "TUPLE", "IMAGE")
    RETURN_NAMES = ("CLIENT", "PRIVIDER", "SCHEMA", "RENDERED", "API_SCHEMAS", "API_RESULT", "RESULT_IMAGE")
    FUNCTION = "process_uniapi"

    API_RESULT = api_helper.get_api_config("apiconfig.json")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "api_provider": (list(cls.API_RESULT.keys()),),
            }
        }

    def process_uniapi(self, api_provider):
        config_json = self.API_RESULT
        client, api_provider = api_helper.create_api_client(api_provider, config_json)

        path = os.path.join(PRIMERE_ROOT, 'json')
        fp = os.path.join(path, 'api_schemas.json')
        schema = utility.json2tuple(fp)

        rendered, used_values = api_json_to_requestbody.render_from_schema(schema)

        # return (client, api_provider, schema, rendered, used_values)

        api_result = None
        api_error = None
        final_batch_img = []
        result_image = None
        image_list = []

        try:
            if rendered.method.upper() == "SDK":
                print('Rendering SDK ====================================')
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
                print('Rendering NOT SDK ================================')
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
                "used_values": used_values,
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

