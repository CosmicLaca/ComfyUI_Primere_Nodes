response = client.models.generate_videos(
    model="veo-3.1-generate-preview",
    prompt=prompt,
    image=first_image,
    config=types.GenerateVideosConfig(
      aspectRatio="9:16",
      resolution="720p",
      durationSeconds="8s",
      personGeneration="allow_all",
      reference_images=reference_images,
      last_frame=last_image
    ),
)