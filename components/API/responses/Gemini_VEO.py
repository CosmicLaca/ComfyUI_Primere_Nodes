from __future__ import annotations

import time
import os
import random
import folder_paths


def handle_response(api_result, schema=None, loaded_client=None, response_url=None, client=None, sdk_context=None):
    if api_result is None:
        return [None, None]

    operation = api_result
    while not operation.done:
        time.sleep(10)
        operation = client.operations.get(operation)

    generated_videos = getattr(getattr(operation, 'response', None), 'generated_videos', None)
    if not generated_videos:
        return [None, None]

    generated_video = generated_videos[0]
    tmp_path = os.path.join(folder_paths.temp_directory, f"veo_{random.randint(10000, 99999)}.mp4")
    client.files.download(file=generated_video.video)
    generated_video.video.save(tmp_path)

    with open(tmp_path, 'rb') as f:
        video_bytes = f.read()

    return ["video_result", video_bytes]
