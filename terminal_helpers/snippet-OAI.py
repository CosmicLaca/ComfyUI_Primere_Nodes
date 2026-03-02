response = client.images.generate(
    model=model,
    prompt=prompt,
    n=batch,
    moderation = moderation,
    background = background,
    output_format = output_format,
    output_compression = "INT",
    size=size,
    quality = quality,
    response_format = response_format
)
