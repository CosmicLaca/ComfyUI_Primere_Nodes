from __future__ import annotations

import json

from . import response_helper


def handle_response(api_result, schema=None):
    try:
        json_object = json.loads(json.dumps(api_result))
    except (ValueError, TypeError) as exc:
        raise RuntimeError(f"Invalid JSON response received: {api_result}") from exc

    video = json_object.get("video") if isinstance(json_object, dict) else None
    if not isinstance(video, dict):
        raise RuntimeError(f"No 'video' key in Kling response: {json_object}")

    video_url = video.get("url")
    if not video_url:
        raise RuntimeError(f"No URL in Kling video response: {video}")

    video_bytes = response_helper.fetch_url_bytes(video_url)
    return ["video_result", video_bytes]
