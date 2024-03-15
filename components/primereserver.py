from server import PromptServer
from aiohttp import web
import json
from pathlib import Path
from ..components import utility
import os
from PIL import Image

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

    PREVIEW_DATA = json.loads(post.get('previewdata')) # {'PreviewTarget': 'Checkpoint', 'PreviewTargetOriginal': 'Sci-fi\\colorful_v30.safetensors', 'extension': 'jpg', 'ImageName': 'ComfyUI_temp_pmzjp_00092_.png', 'ImagePath': 'H:\\ComfyUI\\output', 'SaveImageName': 'colorful_v30', 'maxWidth': 220, 'maxHeight': 220}
    IMG_SOURCE = os.path.join(PREVIEW_DATA['ImagePath'], PREVIEW_DATA['ImageName']) # H:\ComfyUI\output\ComfyUI_temp_pmzjp_00092_.png
    PRW_TYPE = PREVIEW_DATA['PreviewTarget'] # Checkpoint
    CONVERSION = utility.PREVIEW_PATH_BY_TYPE # {'Checkpoint': 'H:\\ComfyUI\\web\\extensions\\Primere\\images\\checkpoints', 'CSV Prompt': 'H:\\ComfyUI\\web\\extensions\\Primere\\images\\styles', 'Lora': 'H:\\ComfyUI\\web\\extensions\\Primere\\images\\loras', 'Lycoris': 'H:\\ComfyUI\\web\\extensions\\Primere\\images\\lycoris', 'Hypernetwork': 'H:\\ComfyUI\\web\\extensions\\Primere\\images\\hypernetworks', 'Embedding': 'H:\\ComfyUI\\web\\extensions\\Primere\\images\\embeddings'}
    TARGET_DIR = CONVERSION[PRW_TYPE]
    if os.path.isfile(IMG_SOURCE) and os.path.exists(IMG_SOURCE): # H:\ComfyUI\output\ComfyUI_temp_pmzjp_00092_.png
        path, filename = os.path.split(PREVIEW_DATA['PreviewTargetOriginal']) # Sci-fi
        FULL_TARGET_PATH = os.path.join(TARGET_DIR, path) # H:\ComfyUI\web\extensions\Primere\images\checkpoints\Sci-fi

        TARGET_FILE = os.path.join(TARGET_DIR, path, PREVIEW_DATA['SaveImageName'] + '_000_test.' + PREVIEW_DATA['extension']) # H:\ComfyUI\web\extensions\Primere\images\checkpoints\Sci-fi\colorful_v30_000_test.jpg
        if os.path.isfile(TARGET_FILE):
            PreviewSaveResponse = "Preview file for " + PREVIEW_DATA['PreviewTargetOriginal'] + " replaced with current preview for " + PREVIEW_DATA['PreviewTarget'] + "."
        else:
            PreviewSaveResponse = "Preview file for " + PREVIEW_DATA['PreviewTargetOriginal'] + " created and saved for " + PREVIEW_DATA['PreviewTarget'] + "."

        if not os.path.isdir(FULL_TARGET_PATH): # H:\ComfyUI\web\extensions\Primere\images\checkpoints\Sci-fi
            Path(str(FULL_TARGET_PATH)).mkdir(parents = True, exist_ok = True)

        if os.path.isfile(IMG_SOURCE) and os.path.isdir(FULL_TARGET_PATH):
            try:
                prw_img = Image.open(IMG_SOURCE).convert("RGB")
                newsize = (PREVIEW_DATA['maxWidth'], PREVIEW_DATA['maxHeight'])
                prw_img_resized = prw_img.resize(newsize)
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
