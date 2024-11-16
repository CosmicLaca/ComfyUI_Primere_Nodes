# <ins>Prompt Enhancement Node with Local LLMs:</ins>

<img src="llm_enhancer.jpg" width="300px">

This specialized ComfyUI node utilizes local LLM models to enhance, refine, or repair image generation prompts. It's particularly valuable for optimizing prompts for modern DiT (Diffusion Transformers) models like Flux and Cascade, which require T5-XXL compatible prompt structures.

<hr>

### <ins>Comparison test:</ins>

#### Recommended Models:
- **Flux-Prompt-Enhance**: 850MB ~4 sec :: [link](https://huggingface.co/gokaygokay/Flux-Prompt-Enhance)
- **Llama-3.2-3B-Instruct**: 5.9GB ~38 sec :: [link](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct)
- **Llama-3.2-3B-Instruct-PromptEnhancing**: 17.5MB, but Llama-3.2-3B-Instruct required ~8 sec :: [link](https://huggingface.co/groloch/Llama-3.2-3B-Instruct-PromptEnhancing)
- **granite-3.0-2b-instruct**: 4.9GB ~10 sec :: [link](https://huggingface.co/ibm-granite/granite-3.0-2b-instruct)
- **Qwen2.5-3B-Instruct**: 5.7GB ~30 sec :: [link](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- **Qwen2.5-3B-Instruct-PromptEnhancing**: 13.7MB, but Qwen2.5-3B-Instruct required ~9 sec :: [link](https://huggingface.co/groloch/Qwen2.5-3B-Instruct-PromptEnhancing)
- **SmoLLM2-1.7B-Instruct**: 3.1GB ~17 sec :: [link](https://huggingface.co/HuggingFaceTB/SmolLM2-1.7B-Instruct)
- **SmoLLM2-Prompt-Enhance**: 517MB ~10 sec :: [link](https://huggingface.co/gokaygokay/SmolLM2-Prompt-Enhance)
- **TinyLlama-1.1B-32k-Instruct**: 4GB ~11 sec :: [link](https://huggingface.co/Doctor-Shotgun/TinyLlama-1.1B-32k-Instruct)

Test workflow: `[your_comfy_dir]\custom_nodes\ComfyUI_Primere_Nodes\Workflow\civitai-llm-enhancer.json`

All models can be downloaded from Huggingface repositories, the node compatible with lot of another models and adapters.
The model path is: `[your_comfy_dir]\custom_nodes\ComfyUI_Primere_Nodes\Nodes\Downloads\LLM\`, models have to save in separated subfolders, basically same name as the model.

**The best model is Qwen2.5-3B-Instruct-PromptEnhancing but because this is just adapter, Qwen2.5-3B-Instruct required.**

<hr>

## Test run of recommenden nodes: (GPU: RTX 3060/12, seed: `564992973323879`, precision: `FP32`)

#### <ins>Flux-Prompt-Enhance:</ins>
- **Input prompt:** `beautiful 35 years old woman, model, professional photographer, workshop, glamour magazine, lights`
  - **Configurator:** `balanced`
    - **Result:** `A detailed oil painting of a womans face and upper body in profile her hair is styled to fit the models 135. The model has long straight brown locks that are held together by their sides with one hand resting on an open space where they meet at least 50 feet away from each other as she looks down or look like it was created for many years.\"LAMBORGHINIA\" appears written prominently across all four pages while labonnai says\"gabrielel\". Intricate lighting effects include various lights including headband bands around his neck some light sources reflect off different areas creating shadow effect behind him (photorealistic)`
    - **Running time:** 4.5 sec
  - **Configurator:** `photorealistic`
    - **Result:** `An ethereal detailed digital illustration of a woman adorned in a black dress sits at the forefront of the image. The text is a masterpiece in itself capturing a moment of beauty and reflection with a professional photographer adjusting the models hair. The model seated at a wooden table holds a candle in her right hand adding a warm glow to the scene. The piece is set against a dark background highlighting the intricate details of the model and the workshop behind her`
    - **Running time:** 4 sec
  - **Configurator:** `consistent`
    - **Result:** `a closeup photograph of text in two columns the left column is mostly black with gold accents and has detailed facial features including long brown hair styled up to look like an older woman or model wearing traditional attire such as dresses that are also aged but have visible skin tone patterns on their face. The center row starts by looking down at her profile while she faces away towards another page which appears slightly blurred because itll be so well lit for all three items`
    - **Running time:** 4 sec
  - **Configurator:** `refinement`
    - **Result:** `An indoor closeup view of a texttoimage image of a woman and a professional photographer the woman is positioned in the center of the image and is looking directly at the camera she has long dark hair and is wearing a longsleeved top she is also wearing a highwaisted top the model is positioned on the right side of the image she is looking directly into the camera and is looking down at the model. The model is placed on top of a table that is made up of wooden boards that are positioned horizontally and placed side by side. Behind the table is a large number of lights that are illuminating the area`
    - **Running time:** 6 sec
  - **Configurator:** `default`
    - **Result:** `A detailed digital illustration of a beautiful woman with long brown hair her face illuminated by the warm light from above. The text is written in an elegant serif font that reads \"1 high quality 1\" and below it says\"the model.\"An indoor professional photographer stands behind me to direct your attention into this captivating scene his hands are placed on their lap as she looks down at something unseen beyond him\`
    - **Running time:** 5.3 sec

<hr>

#### <ins>Llama-3.2-3B-Instruct:</ins>
- **Input prompt:** `beautiful 35 years old woman, model, professional photographer, workshop, glamour magazine, lights`
  - **Configurator:** `balanced`
    - **Result:** `Capture the essence of a stunning 35yearold woman who embodies both elegance and sophistication. As a successful model and professional photographer she exudes confidence and poise in front of the camera. Set against a glamorous backdrop with soft golden lighting her beauty shines through as she poses effortlessly in a luxurious setting. The focus is on highlighting her features from her striking eyes to her flawless skin all while conveying an air of refinement and class that would make her cover a highend fashion magazine like Glamour`
    - **Running time:** 40 sec
  - **Configurator:** `photorealistic`
    - **Result:** `Model A closeup shot highlighting facial features (smile) along side fullbody pose`
    - **Running time:** 43 sec
  - **Configurator:** `consistent`
    - **Result:** `Create an image of a stunning 35yearold woman posing as a model with a professional photographers expertise. Incorporate glamorous lighting and a sophisticated backdrop reminiscent of the pages of a highend fashion magazine`
    - **Running time:** 42 sec
  - **Configurator:** `refinement`
    - **Result:** `Photograph a stunning 35yearold female model in a glamorous setting with soft natural light`
    - **Running time:** 43 sec
  - **Configurator:** `default`
    - **Result:** `Capture the essence of a stunning 35yearold model who embodies elegance and sophistication. Set against a glamorous backdrop with soft natural light pouring in from above she poses confidently in front of a polished wooden table adorned with luxurious props such as designer handbags fine jewelry and highend cameras. The atmosphere is one of refinement and poise reflecting her status as a successful model and professional photographer`
    - **Running time:** 47 sec

<hr>

#### <ins>Llama-3.2-3B-Instruct-PromptEnhancing:</ins>
- **Input prompt:** `beautiful 35 years old woman, model, professional photographer, workshop, glamour magazine, lights`
  - **Configurator:** `balanced`
    - **Result:** `A beautiful young attractive female with short hair and bright eyes. She is a 2 5 year old wearing an elegant outfit in the studio of a professional photographer doing photoshoot for photography magazine glamorous lighting very detailed face highly detailed face well lit digital painting concept art by Brian Froud Artgerm Ruan Jia and Brent Crump and Donato Giancola and William Adolph Haag and Charles Westbrook trending on DeviantArt`
    - **Running time:** 8.6 sec
  - **Configurator:** `photorealistic`
    - **Result:** `A beautiful young female with long hair and a pretty face. She is wearing glasses. The background of the photo will be a studio photography work shop for photographers taking pictures of models in glamorous fashion magazines. A large lens camera on tripod stands at right foreground left center. There are many light sources from all directions lighting up this scene by natural daylight or artificial light. Artstation trending Photorealism high quality octane render cinematic lighting ultra detailed unreal engine`
    - **Running time:** 8.2 sec
  - **Configurator:** `consistent`
    - **Result:** `a beautiful young woman with long hair and a professional photographer in the background taking pictures of her self full body shot studio lighting studio lighting setup studio lighting studio lighting studio lighting studio lighting studio lighting studio lighting studio lighting studio lighting studio lighting studio lighting studio lighting studio lighting studio lighting studio lighting studio lighting studio lighting studio lighting studio lighting studio lighting studio lighting studio lighting studio lighting`
    - **Running time:** 8 sec
  - **Configurator:** `refinement`
    - **Result:** `A beautiful young woman with long dark hair and blue eyes. She is a 35 year old female model in her studio wearing an elegant outfit by designer brand. Her face has high definition features she has full lips perfect nose very pretty eyebrows smooth skin. The background of the photo is white. A light source from above shines down on her head. This is a highly detailed digital painting concept art created by Greg Rutkowski. It is trending on Artstation`
    - **Running time:** 8.3 sec
  - **Configurator:** `default`
    - **Result:** `A beautiful young woman with long hair and a big smile. She is wearing a white tank top and black shorts in the studio of an art photography class for women by a famous photographer. The lighting is perfect there are no shadows she has perfectly defined facial features her eyes sparkle she looks like a supermodel glamorous fashion shoot very detailed face realistic pose natural skin tone 35mm film grain digital retouched ultra wide angle shot highly`
    - **Running time:** 8.2 sec

<hr>

#### <ins>granite-3.0-2b-instruct:</ins>
- **Input prompt:** `beautiful 35 years old woman, model, professional photographer, workshop, glamour magazine, lights`
  - **Configurator:** `balanced`
    - **Result:** `Capture the essence of elegance and charm with a captivating portrait. A stunningly beautiful women in her prime stands tall at an age that exudes confidence yet retains youthful allure`
    - **Running time:** 10 sec
  - **Configurator:** `photorealistic`
    - **Result:** `A captivating 30yearold woman resembling a professional model poses elegantly in a workshop setting. The room is adorned with highend photography equipment and equipment reflecting her profession as a photographer. The atmosphere is filled with natural soft lighting reminiscent of a glam`
    - **Running time:** 9.2 sec
  - **Configurator:** `consistent`
    - **Result:** `A captivating threedecade young lady with an elegant and alluring appearance is seeking your expertise in a photoshoot for the prestigious Glamor Magazines Workshop`
    - **Running time:** 11 sec
  - **Configurator:** `refinement`
    - **Result:** `Imagine a stunning 35yearold woman a professional model and photographer setting up a workshop in a glamorous magazine setting. The room is welllit highlighting her beauty and professionalism`
    - **Running time:** 10 sec
  - **Configurator:** `default`
    - **Result:** `A captivating threedimensional representation of an attractive lady aged thirteen`
    - **Running time:** 10 sec

<hr>

#### <ins>Qwen2.5-3B-Instruct:</ins>
- **Input prompt:** `beautiful 35 years old woman, model, professional photographer, workshop, glamour magazine, lights`
  - **Configurator:** `balanced`
    - **Result:** `Can you create an image of a beautiful 35yearold woman who is both a model and a professional photographer in the middle of a glamorous photography workshop? She should be captured with elegant lighting that enhances her features`
    - **Running time:** 36 sec
  - **Configurator:** `photorealistic`
    - **Result:** `Heres a detailed and descriptive photo request\"A stunning photograph of an elegant 35yearold woman who exudes the confidence and grace typical of both a successful model and dedicated professional photographer. She is captured in her element at one of her photography workshops where shes sharing tips with aspiring photographers or discussing creative techniques that have made her career as glamorous as it has been.The lighting setup for this shot should be perfectsoft natural light streaming through large windows behind our subject casting flattering shadows on soft furnishings around them like plush cushions and vintage wooden bookshelves adorned with photographs from various projects theyve worked on over their illustrious career. The background features subtle textures such as woven tapestries hanging off walls creating depth while keeping focus squarely on our beautiful protagonist.\"Feel free to adjust any details based on your vision further if needed`
    - **Running time:** 39 sec
  - **Configurator:** `consistent`
    - **Result:** `Can you create an image of a beautiful 35yearold woman who is a model and professional photographer attending a glamorous workshop for a cover story in a glamour magazine? The scene should be illuminated with flattering studio lights`
    - **Running time:** 34 sec
  - **Configurator:** `refinement`
    - **Result:** `A stunning and glamorous photograph of a beautiful 35yearold woman who embodies the essence of professionalism as both an accomplished model and talented professional photographer. The scene should be set in or around her photography studioworkshop environment with soft lighting that highlights her features and captures the elegance and sophistication expected from a cover shot for a highend glamour magazine. Ensure she is centered frontally facing towards the viewer surrounded by props such as cameras lenses and perhaps some samples of her work emphasizing her expertise in capturing beauty through images`
    - **Running time:** 36 sec
  - **Configurator:** `default`
    - **Result:** `Heres a detailed and creative  for generating an image\"An exquisite portrait of a beautiful 35yearold woman who is both a model and a professional photographer. She stands confidently in the center of a studio setup that perfectly captures her elegance and professionalism. The lighting design should be glamorous with soft shadows underlining her features while highlighting her striking beauty.The background behind her could feature elegant textures like fine fabric or delicate patterns reminiscent of luxury magazines such as Glamour. Her pose suggests shes leading into action  perhaps about to capture another stunning moment on camera.In the foreground there might be props related to photography equipmentlike cameras resting gently beside her feetand possibly some accessories from her worka vintage lens cap next to one hand maybe even a small photo book open nearby showing off her portfolio.Overall this scene aims to convey not just her physical attractiveness but also her professionality through thoughtful details and composition.\"Feel free to adjust any elements based on your specific vision or preferences`
    - **Running time:** 40 sec

<hr>

#### <ins>Qwen2.5-3B-Instruct-PromptEnhancing:</ins>
- **Input prompt:** `beautiful 35 years old woman, model, professional photographer, workshop, glamour magazine, lights`
  - **Configurator:** `balanced`
    - **Result:** `A beautiful female in her early thirties is a fashion and photography professional. She has long flowing hair perfect skin with a flawless face natural beauty. A highly detailed digital painting by susanmccue from the internet. Trending on Artstation.com. Professional portrait of an attractive young adult girl. Studio lighting. Fashion magazine cover photo. Magazine spread. Glamorous shot. Soft focus effect. Beautiful background. By greg rutkowski. High resolution image`
    - **Running time:** 10 sec
  - **Configurator:** `photorealistic`
    - **Result:** `a beautiful young women in her early thirties with long blonde hair and a perfect face. she is wearing an elegant dress as the cover of a glamorous fashion magizine photo shoot by greg rutkowski on a white background at night under soft studio lighting from behind. portrait shot taken through glass window looking out to sea surrounded by flowers. highly detailed realistic illustration concept art cinematic scene trending on pixiv fanbox hd wallpaper octane render unreal engine material shading ai generated high quality`
    - **Running time:** 9 sec
  - **Configurator:** `consistent`
    - **Result:** `a beautiful 35 year old woman with long blonde hair and blue eyes a professional photographer in a workshop in a glamour magazine in the style of a magazine cover by greg rutkowski trending on artstation high quality sharp focus realistic face photorealistic hyper detailed intricate elegant cinematic lighting octane render cgsociety pixiv unreal engine hd wallpaper 8k resolution art by g`
    - **Running time:** 9.3 sec
  - **Configurator:** `refinement`
    - **Result:** `a beautiful young women in her early thirties with long blonde hair and a perfect face. she is wearing an elegant outfit as a professional photographer at work on a photo shoot of flowers. the background has soft lighting from behind that makes it look like a magazine cover page. high quality photograph by greg rutkowski. trending on artstation. this image was taken by yoshitaka amano. highly detailed shot. realistic painting style. hyperrealistic digital illustration concept art`
    - **Running time:** 9 sec
  - **Configurator:** `default`
    - **Result:** `a beautiful young women in her early thirties with long blonde hair and a perfect face. she is wearing an elegant dress as a professional photographer at work on the floor of a photography studio while taking pictures. she has dark brown eyes and blue lips. she looks glamorous like a cover shot from a fashion magazine. trending on artstation. award winning photo by greg rutkowski. cinematic lighting. digital painting. hyper detailed artwork. sharp focus. unreal engine rendering.`
    - **Running time:** 10 sec

<hr>

#### <ins>SmoLLM2-1.7B-Instruct:</ins>
- **Input prompt:** `beautiful 35 years old woman, model, professional photographer, workshop, glamour magazine, lights`
  - **Configurator:** `balanced`
    - **Result:** `Model  She can be considered both physically attractive and also someone who works professionally with models or photography equipment`
    - **Running time:** 17.4 sec
  - **Configurator:** `photorealistic`
    - **Result:** `Photographers Workshop for the Glamorous Model of a Lifetime`
    - **Running time:** 17.4 sec
  - **Configurator:** `consistent`
    - **Result:** `Beautiful 35yearold woman model professional photographer and instructor at a glamour magazines workshop. She specializes in capturing stunning images that showcase her elegance and grace under the soft glow of natural or artificial lighting`
    - **Running time:** 17 sec
  - **Configurator:** `refinement`
    - **Result:** `Im looking for a beautiful and experienced female model in her midthirties who is proficient at modeling sessions. She should have an excellent portfolio of work that showcases both natural beauty as well as artistic skill through the use of various lighting techniques commonly used by professionals such as photographers`
    - **Running time:** 17 sec
  - **Configurator:** `default`
    - **Result:** `What is the name of this stunning young lady who has made a career out of capturing beauty through her lens? Is she also an accomplished fashion designer or perhaps involved in any other creative pursuits that showcase her artistic talents beyond photography?`
    - **Running time:** 20 sec

<hr>

#### <ins>SmoLLM2-Prompt-Enhance:</ins>
- **Input prompt:** `beautiful 35 years old woman, model, professional photographer, workshop, glamour magazine, lights`
  - **Configurator:** `balanced`
    - **Result:** `beautiful 35 years old woman model professional photographer workshop glamour magazine lights A black and white image of the same person as seen from above. The man is in his mid to late thirties with long dark brown hair that has been styled into an oval shaped headdress he looks up at her while shes standing behind him looking out over their hands touching each others shoulders. His left hand rests on hers holding them close together so they can be pulled apart if you feel any tension or pressure between your fingers being pressed against theirs.) He also wears glasses which look like silver ones but have no light shine through it because there isnt sunlight shining down onto this face.\"I am doing my business\" (the caption reads) \"The world will never know what Ive done.\" (\"My work\")\"Well always remember our love...\"(\"Love\"))\"Our friendship...(a smile)\") She holds one arm extended straight forward extending its length toward those who need help.\" A large glass mirror sits atop two small wooden pedestals hanging vertically along both sides.nnthe bottom right corner shows three smaller mirrors placed horizontally across these sculpturesndepicturing other people wearing similar clothing..\" An open book stands beside another framed photograph showing children playing outdoors under tree branches. Another photo frame hangs slightly off center positioned near middle ground`
    - **Running time:** 12 sec
  - **Configurator:** `photorealistic`
    - **Result:** `beautiful 35 years old woman model professional photographer workshop glamour magazine lights A 1st image of a 30 year old woman wearing a white dress and a black skirt her hair is styled in a ponytail her eyes are bright blue and her smile is bright. She is standing in front of a workshop she is looking to the right and there is a professional photographer in the background behind her there is another woman with long blonde hair and behind that is a woman with short brown hair and a bright smile. The model is wearing a black jacket and she is holding a camera in her right hand. The photograph is taken from the top left corner and the bottom right corner`
    - **Running time:** 5.6 sec
  - **Configurator:** `consistent`
    - **Result:** `beautiful 35 years old woman model professional photographer workshop glamour magazine lights A single image is created with the use of black ink on white paper. The background has light blue lines that run horizontally and vertically across it from bottom left to top right corner in an upward direction as if looking up at something or another persons face but not seeing anything else going through their eyes while they look down towards them there isnt any other objects visible within this area so no shadows can be seen because all surfaces have been completely blanked out except these two small areas where some letters were written above what looks like someone was writing (the letter \"a\" appears slightly larger than its size). A large red circle sits directly behind her nose which also casts shadow onto one side over hers head when she moves forward toward those who will see more details about how much time weve had since our last conversation.\"Im sorry I didnt think youd come back here\" said my husband after he finished his coffee before heading off into town.\"\"nThe photo shows three people standing around wearing dark clothing facing each others faces.\"\"   picture left right   40 inches tall metal frame placed atop wooden planks  wood grain runs diagonally between both frames . img src\`
    - **Running time:** 10 sec
  - **Configurator:** `refinement`
    - **Result:** `beautiful 35 years old woman model professional photographer workshop glamour magazine lights A detailed 3D render of a 35yearold woman a professional photographer and a workshop model. The woman is dressed in a white dress with her hair styled in a flowing ponytail and her eyes are filled with a deep shade of blue. She is wearing a black jacket and a black hoodie and her hair is styled in loose waves. She is holding a camera in her right hand and her left hand is holding a phone. She is standing in front of a glass wall and she is looking at the camera. Behind her there is a workshop model and behind that there is another model. The model is wearing a white shirt and it is holding a smartphone in its right hand. There is a window in front of the model and the window is open. The window is on the left side of the image and there is a light fixture on the right side of the window. The light fixture is reflecting off of the window and it casts a shadow on the model`
    - **Running time:** 7 sec
  - **Configurator:** `default`
    - **Result:** `beautiful 35 years old woman model professional photographer workshop glamour magazine lights An image of the same person as seen from above. The photo is taken at night with light shining down on her face and looking up to see what she looks like in detail theres no background or any other objects that can be visible but it has been turned into an exact replica so they will look exactly alike (the only difference between them being their age). There isnt much sunlight coming through this door which makes everything appear darker than its actual size because youre not seeing anything else outside your viewport while taking these photos  just one small window out over top left corner showing something inside another smaller glass pane behind each individual picture frame placed directly across front right side where all four frames have come together forming two large vertical lines running vertically along both sides going horizontally towards bottom center line below) She also appears slightly more slender compared having less weight since nothing was dropped onto hers except possibly some dust particles floating around within those three framed pictures about five inches away off opposite corners respectively.) Her eyes shine bright blue when viewed against dark gray backgrounds near upper middle portion toward lower half central section (text0 rightarrow left(fracpi6). A black metal rod extends diagonally upward extending far enough forward beyond most surrounding buildings windows`
    - **Running time:** 11 sec


<hr>

#### <ins>TinyLlama-1.1B-32k-Instruct:</ins>
- **Input prompt:** `beautiful 35 years old woman, model, professional photographer, workshop, glamour magazine, lights`
  - **Configurator:** `balanced`
    - **Result:** `You have provided a very detailed and rich description of a beautiful 35 yearold woman with whom you had a great experience. Its hard to imagine her being any more gorgeous than she is in this image`
    - **Running time:** 11 sec
  - **Configurator:** `photorealistic`
    - **Result:** `I present to you a stunning portrait of the beautiful and talented model. She is not only an incredible artist but also one who has dedicated her life to helping others achieve their dreams through artistry. Her name? womanname  shell be sure to make your day brighter than ever with this gorgeous image`
    - **Running time:** 13.5 sec
  - **Configurator:** `consistent`
    - **Result:** `No useful result`
    - **Running time:** - sec
  - **Configurator:** `refinement`
    - **Result:** `No useful result`
    - **Running time:** - sec
  - **Configurator:** `default`
    - **Result:** `I present to you a stunning portrait of the beautiful and talented 35 year old woman named Model. She is not only an accomplished photographer but also a successful businesswoman who has worked with many renowned magazines. Her name is Woman and she has been featured in several prestigious publications such as Glamour Magazine`
    - **Running time:** 11 sec