response = requests.post("https://api.bfl.ai/v1/flux-pro-1.0-fill",
    payload={
        "output_format": "png",
        "image": reference_images,
        "mask": mask_images,
        "safety_tolerance": safety_tolerance,
        "prompt": prompt,
        "seed": seed,
        "aspect_ratio": aspect_ratio,
        "prompt_upsampling": "STRING",
        "guidance": "FLOAT",
        "steps": "INT"
    }
)