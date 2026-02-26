handler = client.submit("fal-ai/bytedance/seedream/v5/lite/text-to-image",
    arguments={
        "prompt": prompt,
        "image_size": {
          "width": 1280,
          "height": 720
        },
        "num_images": 1,
        "max_images": 1,
        "enable_safety_checker": true,
        "enhance_prompt_mode": "standard",
        "seed": seed
    }
)