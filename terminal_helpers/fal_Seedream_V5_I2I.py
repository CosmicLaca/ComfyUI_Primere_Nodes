handler = lient.submit("fal-ai/bytedance/seedream/v5/lite/edit",
    arguments={
        "prompt": prompt,
        "img_urls": [images],
        "image_size": {
          "width": 1280,
          "height": 720
        },
        "num_images": batch,
        "max_images": 1,
        "enable_safety_checker": False,
        "enhance_prompt_mode": "standard",
        "seed": seed
    },
    with_logs=False
)