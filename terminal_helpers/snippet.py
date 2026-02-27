response = client.models.generate_content(
    model="gemini-3.1-flash-image-preview",
    contents=[prompt, reference_images],
    config=types.GenerateContentConfig(
        response_modalities=["IMAGE"],
        thinking_config=types.ThinkingConfig(
            aspect_ratio=aspect_ratio,
            image_size="1K",
            thinking_level="High",
            include_thoughts=False
        ),
    )
)