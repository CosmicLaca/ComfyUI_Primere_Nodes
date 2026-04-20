result = fal_client.subscribe("bytedance/seedance-2.0/fast/reference-to-video",
    arguments={
        "prompt": prompt,
        "image_urls": reference_image,
        "video_urls": video_urls,
        "audio_urls": audio_urls,
        "resolution": resolution,
        "seed": seed,
        "duration": "auto",
        "aspect_ratio": "auto",
        "with_logs": with_logs
    }
)