result = fal_client.subscribe("fal-ai/kling-video/o3/pro/reference-to-video",
    arguments={
        "prompt": prompt,
        "multi_prompt": null,
        "start_image_url": reference_image,
        "duration": "8",
        "aspect_ratio": "16:9",
        "with_logs": with_logs
    }
)