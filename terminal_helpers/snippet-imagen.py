response = client.models.generate_images(
    model='imagen-4.0-generate-001',
    prompt=prompt,
    config=types.GenerateImagesConfig(
        number_of_images=4,
        aspect_ratio="16:9",
        image_size="1K",
        person_generation="allow_all",
    )
)