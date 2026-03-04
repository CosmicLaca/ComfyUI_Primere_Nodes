from __future__ import annotations

import time
from urllib.parse import urlparse
import requests
import json
from . import response_helper

def handle_response(api_result, schema=None, loaded_client=None, response_url=None):
    status_accepted = ['Ready', 'Pending', 'Task not found']
    parsed_url = urlparse(response_url)
    blackforest_api_region = parsed_url.netloc or None
    path_parts = [part for part in parsed_url.path.split("/") if part]
    blackforest_api_version = path_parts[0] if len(path_parts) > 0 else None

    try:
        json_object = json.loads(api_result.text)
    except ValueError as e:
        raise RuntimeError(f"Input object failed: {api_result}")

    if 'polling_url' in json_object:
        resp = requests.get(json_object['polling_url'])
        resp_json_object = json.loads(resp.text)

        status = 'Start'
        request_tryout = 0
        error_tryout = 0

        while error_tryout <= 1:
            while status != 'Ready' or request_tryout <= 20:
                url_res = f"https://{blackforest_api_region}/{blackforest_api_version}/get_result"
                querystring = {"id": resp_json_object['id']}
                response = requests.request("GET", url_res, params=querystring)
                resp_json_object = json.loads(response.text)
                status = resp_json_object['status']
                if status not in status_accepted:
                    resp_error = requests.get(json_object['polling_url'])
                    resp_error_json_object = json.loads(resp_error.text)
                    error_status = resp_error_json_object['status']
                    raise RuntimeError(f"Response status: {status}, error status: {error_status}")
                if status == 'Ready':
                    break
                time.sleep(2)
                request_tryout = request_tryout + 1
            time.sleep(1)
            if status == 'Ready':
                break
            error_tryout = error_tryout + 1

        if status != "Ready":
            raise RuntimeError("No result image...")

        image_url = resp_json_object.get("result", {}).get("sample")
        if not image_url:
            raise RuntimeError("No result image...")

        return response_helper.url_to_tensor(image_url)