import json
from pathlib import Path
from ..components import utility
import os
from PIL import Image
from server import PromptServer
from aiohttp import web
import folder_paths
from ..components.tree import PRIMERE_ROOT

'''
# http://127.0.0.1:8188/primere/getdata/ez az adat
@PromptServer.instance.routes.get("/primere/getdata/{data}")
async def primere_getdata(request):
    print('ez itt a getdata szerver')
    print(request.match_info['data'])
    return web.json_response({
        "input_data": request.match_info['data']
    })

# http://127.0.0.1:8188/primere/getquery?sss=ddd&www=rrr&cccc=4444
@PromptServer.instance.routes.get("/primere/getquery")
async def primere_getquery(request):
    print('ez itt a getquery szerver')
    print(request.rel_url.query)
    return web.json_response({
        "input_data": "megvolt"
    })
'''

routes = PromptServer.instance.routes
@routes.post('/primere_preview_post')
async def primere_preview_post(request):
    post = await request.post()
    PreviewSaveResponse = None
    SAVE_MODE = 'Create'

    PREVIEW_DATA = json.loads(post.get('previewdata')) # {'PreviewTarget': 'Checkpoint', 'PreviewTargetOriginal': 'Sci-fi\\colorful_v30.safetensors', 'extension': 'jpg', 'ImageName': 'ComfyUI_temp_pmzjp_00092_.png', 'ImagePath': 'H:\\ComfyUI\\output', 'SaveImageName': 'colorful_v30', 'maxWidth': 220, 'maxHeight': 220}
    IMG_SOURCE = os.path.join(PREVIEW_DATA['ImagePath'], PREVIEW_DATA['ImageName']) # H:\ComfyUI\output\ComfyUI_temp_pmzjp_00092_.png
    PRW_TYPE = PREVIEW_DATA['PreviewTarget'] # Checkpoint
    CONVERSION = utility.PREVIEW_PATH_BY_TYPE # {'Checkpoint': 'H:\\ComfyUI\\web\\extensions\\Primere\\images\\checkpoints', 'CSV Prompt': 'H:\\ComfyUI\\web\\extensions\\Primere\\images\\styles', 'Lora': 'H:\\ComfyUI\\web\\extensions\\Primere\\images\\loras', 'Lycoris': 'H:\\ComfyUI\\web\\extensions\\Primere\\images\\lycoris', 'Hypernetwork': 'H:\\ComfyUI\\web\\extensions\\Primere\\images\\hypernetworks', 'Embedding': 'H:\\ComfyUI\\web\\extensions\\Primere\\images\\embeddings'}
    TARGET_DIR = CONVERSION[PRW_TYPE]
    if os.path.isfile(IMG_SOURCE) and os.path.exists(IMG_SOURCE): # H:\ComfyUI\output\ComfyUI_temp_pmzjp_00092_.png
        path, filename = os.path.split(PREVIEW_DATA['PreviewTargetOriginal']) # Sci-fi
        FULL_TARGET_PATH = os.path.join(TARGET_DIR, path) # H:\ComfyUI\web\extensions\Primere\images\checkpoints\Sci-fi

        TARGET_FILE = os.path.join(TARGET_DIR, path, PREVIEW_DATA['SaveImageName'] + '.' + PREVIEW_DATA['extension']) # H:\ComfyUI\web\extensions\Primere\images\checkpoints\Sci-fi\colorful_v30_000_test.jpg
        if os.path.isfile(TARGET_FILE):
            SAVE_MODE = PREVIEW_DATA['PrwSaveMode']

        PreviewSaveResponse = "Preview file for " + PREVIEW_DATA['PreviewTargetOriginal'] + " used [" + SAVE_MODE + "] mode and saved for " + PREVIEW_DATA['PreviewTarget'] + "."

        if not os.path.isdir(FULL_TARGET_PATH): # H:\ComfyUI\web\extensions\Primere\images\checkpoints\Sci-fi
            Path(str(FULL_TARGET_PATH)).mkdir(parents = True, exist_ok = True)

        if os.path.isfile(IMG_SOURCE) and os.path.isdir(FULL_TARGET_PATH):
            try:
                prw_img = Image.open(IMG_SOURCE).convert("RGB")
                newsize = (PREVIEW_DATA['maxWidth'], PREVIEW_DATA['maxHeight'])
                prw_img_resized = prw_img.resize(newsize)

                if os.path.isfile(TARGET_FILE):
                    match SAVE_MODE:
                        case "Overwrite":
                            prw_img_resized.save(TARGET_FILE, quality=50, optimize=True)
                        case "Keep":
                            PreviewSaveResponse = "Preview file not saved for [" + filename + "] because image already exist and selected [" + SAVE_MODE + "] mode."
                        case "Join horizontal":
                            prw_img_exist = Image.open(TARGET_FILE)
                            joined_img = utility.ImageConcat(prw_img_exist, prw_img, 1)
                            joined_img.save(TARGET_FILE, quality = 50, optimize = True)

                        case "Join vertical":
                            prw_img_exist = Image.open(TARGET_FILE)
                            joined_img = utility.ImageConcat(prw_img_exist, prw_img, 0)
                            if joined_img.size[1] > 250:
                                heigth_ratio = joined_img.size[1] / 220
                                new_width = round(joined_img.size[0] / heigth_ratio)
                                joined_img = joined_img.resize([new_width, 220])
                            joined_img.save(TARGET_FILE, quality = 50, optimize = True)
                else:
                    prw_img_resized.save(TARGET_FILE, quality = 50, optimize = True)

            except Exception:
                PreviewSaveResponse = 'ERROR: Cannot save target image to: ' + str(FULL_TARGET_PATH) + ' for ' + PREVIEW_DATA['PreviewTarget'] + "."
        else:
            PreviewSaveResponse = 'ERROR: Cannot save target image to: ' + str(FULL_TARGET_PATH) + ' for ' + PREVIEW_DATA['PreviewTarget'] + "."
    else:
        PreviewSaveResponse = 'ERROR: Source file: ' + str(IMG_SOURCE) + ' does not exist. Cannot save preview for ' + ' for ' + PREVIEW_DATA['PreviewTarget'] + "."

    if PreviewSaveResponse is not None:
        PromptServer.instance.send_sync("PreviewSaveResponse", PreviewSaveResponse)

    return web.json_response({})


routes2 = PromptServer.instance.routes
@routes2.post('/primere_keyword_parser')
async def primere_keyword_parser(request):
    post = await request.post()
    model_name = post.get('modelName')
    if model_name is not None:
        keyword_list = ['None']
        print('model_name')
        print(model_name)
        ckpt_path = folder_paths.get_full_path("checkpoints", model_name)
        print('ckpt_path')
        print(ckpt_path)
        if os.path.isfile(ckpt_path):
            ModelKvHash = utility.get_model_hash(ckpt_path)
            if ModelKvHash is not None:
                KEYWORD_PATH = os.path.join(PRIMERE_ROOT, 'front_end', 'keywords', 'model-keyword.txt')
                keywords = utility.get_model_keywords(KEYWORD_PATH, ModelKvHash, model_name)
                if keywords is not None and isinstance(keywords, str) == True:
                    if keywords.find('|') > 1:
                        keyword_list = ['None', "Select in order", "Random select"] + keywords.split("|")
                    else:
                        keyword_list = ['None', "Select in order", "Random select"] + [keywords]

            PromptServer.instance.send_sync("ModelKeywordResponse", keyword_list)

    return web.json_response({})
