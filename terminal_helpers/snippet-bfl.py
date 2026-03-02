response = requests.post("https://{{regions}}/{{version}}/{{model_name}}",
     headers={
         "accept": "application/json",
         "x-key": os.environ.get("BFL_API_KEY"),
         "Content-Type": "application/json",
     },
     json={
         "output_format": "png",
         "image_prompt": reference_images,
         "safety_tolerance": safety_tolerance,
         "prompt": prompt,
         "seed": seed,
         "width": width,
         "height": height,
         "prompt_upsampling": "BOOLEAN",
         "guidance": "FLOAT",
         "steps": "INT",
     },
 )
