result = fal_client.subscribe(
    "fal-ai/kling-video/v3/pro/image-to-video",
    arguments={
        "prompt": "This is the main prompt for the video",
        "start_image_url": "https://v3b.fal.media/files/b/0a92706d/h2V27DeUiMH1Pa6lgQ5F4_Frtd4z3L.png",
        "duration": "10",
        "aspect_ratio": "16:9",
        "cfg_scale": 0.5,
        "negative_prompt": "Negative prompt for the full video like blur, distort, low quality, shaky camera, cartoon, anime, text, watermark, deformed face, extra limbs",
        "generate_audio": True,
        "multi_prompt": [{
            "prompt": "Multiprompt 1 example: Slow cinematic push-in through the empty ancient temple. Fog drifts lazily through the valley below. Golden light catches dust particles floating between stone pillars. Wind sways hanging moss and vines on crumbling archways. A flock of birds takes flight in the distance. Atmospheric, still, haunting. No people.",
            "duration": "5"
        }, {
            "prompt": "Prompt for element 1, part of multiprompt dict: @Element1 walks slowly into frame from the left, stepping onto the stone path toward the cliff edge. His cloak billows in the wind. Camera follows him from behind, then he stops at the edge and slowly turns, revealing his scarred face. Golden hour light hits his features. Cinematic, dramatic, anamorphic lens flare.",
            "duration": "5"
        }, {
            "prompt": "Prompt for element 2, part of multiprompt dict: @Element2 custom prompt for element2",
            "duration": "3"
        }, {
            "prompt": "Prompt for element 2, part of multiprompt dict: @Element3 custom prompt for element3",
            "duration": "7"
        }],
        "shot_type": "customize",
        "elements": [{
            "reference_image_urls": ["https://v3b.fal.media/files/b/0a927085/-N8Hiq-2WXSAxnCoN3pAm_PPmLM9y0.png", "blob:https://fal.ai/29772014-3aa9-4561-9ab0-ac0235710bfb", "blob:https://fal.ai/d60ec0cf-2274-459e-b696-45046e88a5c3"],
            "frontal_image_url": "blob:https://fal.ai/ec348dbb-a134-4b7e-9e41-eac9b68a6e39",
            "video_url": "blob:https://fal.ai/041037b2-a43b-4f0c-a974-adab12f401f8"
        }, {
            "reference_image_urls": ["blob:https://fal.ai/97f919d2-756a-4b5d-bf8a-67e404bd97c9", "blob:https://fal.ai/22870181-8cbc-43c2-9373-985622eb4827"],
            "frontal_image_url": "blob:https://fal.ai/b6d773b9-0b15-4e87-9668-5264f1b5b568"
        }, {
            "reference_image_urls": ["blob:https://fal.ai/b68bdc5d-f414-438a-afb5-13118598efc4"],
            "frontal_image_url": "blob:https://fal.ai/ff017f7e-e693-4bc5-aebf-efea9553e435"
        }]
    }
)