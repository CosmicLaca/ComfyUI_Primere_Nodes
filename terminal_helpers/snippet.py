response = client.models.generate_content(
    model="gemini-3-pro-image-preview",
    contents=[prompt, reference_images],
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        image_config=types.ImageConfig(
            aspect_ratio="16:9",
            image_size="1K",
        )
    )
)