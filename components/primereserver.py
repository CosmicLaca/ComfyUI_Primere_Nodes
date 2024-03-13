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
    # print(json.loads(post.get('previewdata')))
    # {'PreviewTarget': 'CSV Prompt', 'extension': 'jpg', 'ImageName': 'ComfyUI_temp_pmzjp_00078_.png', 'ImagePath': 'H:\\ComfyUI\\temp', 'SaveImageName': 'Beast_in_the_beauty_saloon', 'maxWidth': 350, 'maxHeight': 250}
    print('==============================')

    PREVIEW_DATA = json.loads(post.get('previewdata'))
    print(PREVIEW_DATA)
    IMG_SOURCE = os.path.join(PREVIEW_DATA['ImagePath'], PREVIEW_DATA['ImageName'])
    print(IMG_SOURCE)
    PRW_TYPE = PREVIEW_DATA['PreviewTarget']
    print(PRW_TYPE)
    CONVERSION = utility.PREVIEW_PATH_BY_TYPE
    print(CONVERSION)
    TARGET_DIR = CONVERSION[PRW_TYPE]
    TARGET_FILE = os.path.join(CONVERSION[PRW_TYPE], PREVIEW_DATA['SaveImageName'] + '_000test.' + PREVIEW_DATA['extension'])
    print(TARGET_FILE)
    print('------------------------------')
    print(IMG_SOURCE)
    if os.path.isfile(IMG_SOURCE):
        print('létezik a source')
    print('------------------------------')
    print(CONVERSION[PRW_TYPE])
    if os.path.isdir(CONVERSION[PRW_TYPE]):
        print('létezik a target dir')
    print('------------------------------')
    print(TARGET_FILE)
    if os.path.isfile(TARGET_FILE):
        print('létezik a target file')
    print('------------------------------')
    print(TARGET_DIR)
    if os.path.isfile(TARGET_DIR):
        print('létezik a target DIR')

    print('=================================')

    if os.path.isfile(IMG_SOURCE) and os.path.isdir(CONVERSION[PRW_TYPE]):
        prw_img = Image.open(IMG_SOURCE).convert("RGB")
        print('======= **** ============= **** =============')
        width, height = prw_img.size
        print(width)
        print(height)
        newsize = (PREVIEW_DATA['maxWidth'], PREVIEW_DATA['maxHeight'])
        print('------------------------------')
        print(newsize)
        prw_img_resized = prw_img.resize(newsize)
        width, height = prw_img_resized.size
        print('------------------------------')
        print(width)
        print(height)
        prw_img_resized.save(TARGET_FILE, quality = 60, optimize = True)


    return web.json_response({})
