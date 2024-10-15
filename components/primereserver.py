import json
from pathlib import Path
from ..components import utility
import os
from PIL import Image
from server import PromptServer
from aiohttp import web
import folder_paths
from ..components.tree import PRIMERE_ROOT
import base64
import imagesize
from io import BytesIO
from ..Nodes.Inputs import PrimereStyleLoader


'''
************ TEST *******************
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

# ************ IMG SAVER *******************
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
    else:
        PromptServer.instance.send_sync("PreviewSaveResponse", 'Error on serverside process.')

    return web.json_response(PreviewSaveResponse)

# ************ KEYWORDS *******************

routes2 = PromptServer.instance.routes
@routes2.post('/primere_keyword_parser') # sendPOSTModelName()
async def primere_keyword_parser(request):
    post = await request.post()
    model_name = post.get('modelName')
    if model_name is not None:
        keyword_list = ['None']
        ckpt_path = folder_paths.get_full_path("checkpoints", model_name)
        if ckpt_path is not None:
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

# ************ VISUALS *******************

routes3 = PromptServer.instance.routes
@routes3.post('/primere_category_handler') # categoryHandler()
async def primere_category_handler(request):
    post = await request.post()
    setupValue = post.get('setupValue')
    setupMethod = post.get('setupMethod')
    setupKey = post.get('setupKey')

    lastVisual = {}
    if setupMethod == 'read':
        lastVisual['value'] = utility.get_value_from_cache('setup', setupKey)
        lastVisual['key'] = setupKey
        PromptServer.instance.send_sync("LastCategoryResponse", lastVisual)
    elif setupMethod == 'add':
        addResult = utility.add_value_to_cache('setup', setupKey, setupValue)
        PromptServer.instance.send_sync("LastCategoryResponse", addResult)

    return web.json_response({})

routes4 = PromptServer.instance.routes
@routes4.post('/primere_supported_models') # getSupportedModels
async def primere_supported_models(request):
    post = await request.post()
    PromptServer.instance.send_sync("SupportedModelsResponse", utility.SUPPORTED_MODELS)
    return web.json_response({})

routes5 = PromptServer.instance.routes
@routes5.post('/primere_modelpaths') # getAllPath
async def primere_modelpaths(request):
    post = await request.post()
    subdirType = post.get('sourceType')
    models_by_path = []

    if subdirType != 'styles':
        allSource = folder_paths.get_filename_list(subdirType)
        for Source in allSource:
            if "\\" in Source:
                modelSubdir = Source[:Source.index("\\")]
                if modelSubdir not in models_by_path:
                    models_by_path.append(modelSubdir)
    else:
        STYLE_DIR = os.path.join(PRIMERE_ROOT, 'stylecsv')
        STYLE_FILE = os.path.join(STYLE_DIR, "styles.csv")
        STYLE_FILE_EXAMPLE = os.path.join(STYLE_DIR, "styles.example.csv")

        if Path(STYLE_FILE).is_file() == True:
            STYLE_SOURCE = STYLE_FILE
        else:
            STYLE_SOURCE = STYLE_FILE_EXAMPLE
        styles_csv = PrimereStyleLoader.load_styles_csv(STYLE_SOURCE)
        subpathList = styles_csv['preferred_subpath']
        models_by_path = [x for x in list(set(subpathList)) if str(x) != 'nan']

    # models_by_path.sort()
    models_by_path = sorted(models_by_path, key=lambda x: x.casefold()[match_after_x(x, '.', 1)])
    PromptServer.instance.send_sync("AllPathResponse", models_by_path)
    return web.json_response({})

routes6 = PromptServer.instance.routes
@routes6.post('/primere_get_category') # getModelData()
async def primere_get_category(request):
    post = await request.post()
    categoryKey = post.get('cache_key')
    categories = utility.get_category_from_cache(categoryKey)
    categories_by_type = {}
    if categories is not None and len(categories) > 0:
        for cat_key in categories:
            category_val = categories[cat_key]
            if category_val in categories_by_type:
                categories_by_type[category_val].append(cat_key)
            else:
                categories_by_type[category_val] = [cat_key]

    PromptServer.instance.send_sync("CategoryListResponse", categories_by_type)
    return web.json_response({})

routes7 = PromptServer.instance.routes
@routes7.post('/primere_get_subdir') # getModelDatabyPath()
async def primere_get_subdir(request):
    post = await request.post()
    subdirKey = post.get('subdir')
    subdirType = post.get('type')

    if subdirKey != 'styles':
        allSource = folder_paths.get_filename_list(subdirKey)
        models_by_path = list(filter(lambda x: x.startswith(subdirType), allSource))
        if len(models_by_path) == 0:
            models_by_path = allSource
    else:
        STYLE_DIR = os.path.join(PRIMERE_ROOT, 'stylecsv')
        STYLE_FILE = os.path.join(STYLE_DIR, "styles.csv")
        STYLE_FILE_EXAMPLE = os.path.join(STYLE_DIR, "styles.example.csv")

        if Path(STYLE_FILE).is_file() == True:
            STYLE_SOURCE = STYLE_FILE
        else:
            STYLE_SOURCE = STYLE_FILE_EXAMPLE
        styles_csv = PrimereStyleLoader.load_styles_csv(STYLE_SOURCE)
        subpathList = list(styles_csv['preferred_subpath'])
        nameList = list(styles_csv['name'])
        models_by_path = []
        for stylename in nameList:
            nameIndex = nameList.index(stylename)
            pathValue = subpathList[nameIndex]
            if pathValue is not None and pathValue != "" and str(pathValue) != 'nan':
                styleSubString = pathValue + '\\' + stylename
            else:
                styleSubString = 'Root\\' + stylename

            if subdirType == 'All':
                models_by_path.append(styleSubString)
            elif subdirType == pathValue:
                models_by_path.append(styleSubString)
            elif (pathValue is None or pathValue == "" or str(pathValue) == 'nan') and subdirType == 'Root':
                models_by_path.append(styleSubString)

    models_by_path = sorted(models_by_path, key=lambda x: x.casefold()[match_after_x(x, '\\', 1)])
    PromptServer.instance.send_sync("SourceListResponse", models_by_path)
    return web.json_response({})

def match_after_x(filename, match, aftervalue = 0):
    nextindex = filename.find(match) + aftervalue
    return nextindex

routes8 = PromptServer.instance.routes
@routes8.post('/primere_get_version') # getModelDatabyVersion()
async def primere_get_version(request):
    post = await request.post()
    subdirKey = post.get('subdir')
    versionName = post.get('version')
    cachekey = post.get('cachekey')

    allSource = folder_paths.get_filename_list(subdirKey)
    categories = utility.get_category_from_cache(cachekey)
    categories_by_type = {}
    category_modelnames = []
    if categories is not None and len(categories) > 0:
        for cat_key in categories:
            category_val = categories[cat_key]
            if category_val in categories_by_type:
                categories_by_type[category_val].append(cat_key)
            else:
                categories_by_type[category_val] = [cat_key]

    if len(categories_by_type) > 0 and versionName in categories_by_type:
        if len(categories_by_type[versionName]) > 0:
            typefilter = categories_by_type[versionName]
            category_modelnames = [s for s in allSource if any(x + '.' in s for x in typefilter)]

    if len(category_modelnames) == 0:
        category_modelnames = allSource

    PromptServer.instance.send_sync("VersionListResponse", category_modelnames)
    return web.json_response({})

routes9 = PromptServer.instance.routes
@routes9.post('/primere_get_cache') # cacheReadByKey()
async def primere_get_cache(request):
    post = await request.post()
    categoryKey = post.get('chachekey')
    categories = utility.get_category_from_cache(categoryKey)
    PromptServer.instance.send_sync("CacheByKey", categories)
    return web.json_response({})

routes10 = PromptServer.instance.routes
@routes10.post('/primere_get_ascores') # ReadAScores()
async def primere_get_ascores(request):
    post = await request.post()
    asc_type = post.get('type') + '_ascores'
    scoredata = utility.get_category_from_cache(asc_type)
    PromptServer.instance.send_sync("AscoreData", scoredata)
    return web.json_response({})

routes11 = PromptServer.instance.routes
@routes11.post('/primere_get_images') # modelImageData()
async def primere_get_images(request):
    post = await request.post()
    SubdirName = post.get('SubdirName')
    PreviewPath = post.get('PreviewPath')

    if PreviewPath == "false":
        subdir = os.path.join(utility.comfy_dir, 'models', str(SubdirName))
        rootSubdir = Path(subdir).parent
        folder_paths.add_model_folder_path("previewpics_models" + SubdirName, subdir)
        allfiles = folder_paths.get_filename_list("previewpics_models" + SubdirName)
    else:
        subdir = os.path.join(utility.comfy_dir, 'web', 'extensions', 'PrimerePreviews', 'images', str(SubdirName))
        rootSubdir = Path(subdir).parent
        folder_paths.add_model_folder_path("previewpics_legacy" + SubdirName, subdir)
        allfiles = folder_paths.get_filename_list("previewpics_legacy" + SubdirName)

    imagefiles = folder_paths.filter_files_extensions(allfiles, ['.jpg', '.png', '.jpeg', '.preview.jpg', '.preview.jpeg', '.preview.png'])
    imgbase_tuple = {}
    for imagefile in imagefiles:
        if PreviewPath == "false":
            image_path = os.path.abspath(folder_paths.get_full_path('previewpics_models' + SubdirName, imagefile))
        else:
            image_path = os.path.abspath(folder_paths.get_full_path('previewpics_legacy' + SubdirName, imagefile))
        filename = Path(image_path).stem.replace('.preview', '')
        if PreviewPath == "false":
            width, height = imagesize.get(image_path)
            img = Image.open(image_path).convert("RGB")
            if height > 220:
                resizerate = height / 200
                newwidth = width / resizerate
                newsize = (int(newwidth), int(200))
                img = img.resize(newsize, Image.LANCZOS)

            buffered = BytesIO()
            img.save(buffered, format="JPEG", quality=50, optimize=True)
            data = base64.b64encode(buffered.getvalue())
            imgbase_tuple[filename] = data.decode('utf-8')
        else:
            relative_path = image_path.replace(str(rootSubdir), '')
            imgbase_tuple[filename] = relative_path

    PromptServer.instance.send_sync("CollectedImageData", imgbase_tuple)
    return web.json_response({})

routes12 = PromptServer.instance.routes
@routes12.post('/primere_get_filedates') # ReadFileDate()
async def primere_get_filedates(request):
    post = await request.post()
    subdirKey = post.get('type')
    allSource = folder_paths.get_filename_list(subdirKey)
    filedates = {}

    for filename in allSource:
        singleFile = folder_paths.get_full_path(subdirKey, filename)
        filenameonly = Path(singleFile).stem
        singlefiledate = os.path.getctime(singleFile)
        filedates[filenameonly] = singlefiledate

    PromptServer.instance.send_sync("FileDateData", filedates)
    return web.json_response({})