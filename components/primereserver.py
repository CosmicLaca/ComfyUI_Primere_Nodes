import json
from pathlib import Path
import glob
from ..components import utility
from ..components import stylehandler
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
from ..Nodes.Rasterix import PrimereRasterix
import csv
import shutil
from ..utils import here
from ..components.images import histogram as histogram
from ..components.API import api_schema_registry
from ..components.images import image_similarity
import importlib
import inspect

_SECTION_TITLES_CACHE = None

def _collect_section_titles():
    global _SECTION_TITLES_CACHE
    if _SECTION_TITLES_CACHE is not None:
        return _SECTION_TITLES_CACHE

    section_map = {}
    nodes_dir = Path(PRIMERE_ROOT) / "Nodes"
    base_pkg = __package__.split(".components")[0]

    for py_file in nodes_dir.glob("*.py"):
        stem = py_file.stem
        if stem.startswith("_"):
            continue
        try:
            mod = importlib.import_module(f"{base_pkg}.Nodes.{stem}")
        except Exception:
            continue

        for _, obj in vars(mod).items():
            if inspect.isclass(obj) and hasattr(obj, "SECTION_TITLES"):
                titles = getattr(obj, "SECTION_TITLES", None)
                if isinstance(titles, list):
                    section_map[obj.__name__] = titles

    _SECTION_TITLES_CACHE = section_map
    return _SECTION_TITLES_CACHE


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
@routes.post('/primere_select_path')
async def primere_select_path(request):
    post = await request.post()
    node_id_raw = post.get('node_id')
    node_id = str(node_id_raw) if node_id_raw is not None else ""

    if str(post.get('clear', "")).lower() in ("1", "true", "yes"):
        utility.set_node_path(node_id, "")
        return web.json_response({"success": True, "path": ""})

    select_file_param = str(post.get('select_file', '1')).lower()
    select_file = select_file_param not in ("0", "false", "directory", "dir")
    path, error =  utility.open_path_dialog(select_file)

    if error is None:
        utility.set_node_path(node_id, path or "")

    status = 200 if error is None else 503
    response = {"success": error is None, "path": path or "", "error": error}
    return web.json_response(response, status=status)

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
                is_link = os.path.islink(str(ckpt_path))
                ModelKvHash = None
                if is_link == False:
                    ModelKvHash = utility.get_model_hash(ckpt_path)
                if ModelKvHash is not None:
                    KEYWORD_PATH = os.path.join(PRIMERE_ROOT, 'front_end', 'keywords', 'model-keyword.txt')
                    keywords = utility.get_model_keywords(KEYWORD_PATH, ModelKvHash, model_name)
                    if keywords is not None and isinstance(keywords, str) == True:
                        if keywords.find('|') > 1:
                            keyword_list = ['None', "Select in order", "Random select"] + keywords.split("|")
                        else:
                            keyword_list = ['None', "Select in order", "Random select"] + [keywords]

            utility.KEYWORD_SELECTOR_VALUES.clear()
            utility.KEYWORD_SELECTOR_VALUES.extend(keyword_list)
            PromptServer.instance.send_sync("ModelKeywordResponse", keyword_list)

    return web.json_response({})

routes_styles = PromptServer.instance.routes
@routes_styles.post('/primere_unistyle_data')
async def primere_unistyle_data(request):
    post = await request.post()
    selected_file = post.get('style_file')

    style_dir = os.path.join(PRIMERE_ROOT, 'Toml', 'Styles')
    source_files = []
    if os.path.isdir(style_dir):
        source_files = sorted([f for f in os.listdir(style_dir) if f.lower().endswith('.toml')], key=str.casefold)

    if len(source_files) == 0:
        return web.json_response({"files": [], "selected_file": None, "styles": []})

    if selected_file is None or selected_file not in source_files:
        selected_file = source_files[0]

    style_path = os.path.join(style_dir, os.path.basename(selected_file))
    style_result = stylehandler.toml2node(style_path)
    input_dict = style_result[0]

    styles = []
    for input_key, input_value in input_dict.items():
        if input_key.endswith('_strength'):
            continue

        values = ['None']
        if isinstance(input_value, tuple) and len(input_value) > 0 and isinstance(input_value[0], list):
            values = input_value[0]
        elif isinstance(input_value, list):
            values = input_value

        strength_key = input_key + "_strength"
        strength_cfg = {"default": 1, "min": 0.0, "max": 10.0, "step": 0.01}
        if strength_key in input_dict:
            raw_strength = input_dict[strength_key]
            if isinstance(raw_strength, tuple) and len(raw_strength) > 1 and isinstance(raw_strength[1], dict):
                strength_cfg = raw_strength[1]

        styles.append({"key": input_key, "values": values, "strength": strength_cfg})

    return web.json_response({"files": source_files, "selected_file": selected_file, "styles": styles})

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
        excluded = ['.locks', 'Bjornulf_civitAI', 'depthfm', 'models--xiaozaa--cat-tryoff-flux', 'SUPIR', 'Refiners']
        allSource = folder_paths.get_filename_list(subdirType)
        for Source in allSource:
            if "\\" in Source:
                modelSubdir = Source[:Source.index("\\")]
                if modelSubdir not in models_by_path and modelSubdir not in excluded:
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
    supportedImages = ['.jpg', '.png', '.jpeg', '.preview.jpg', '.preview.jpeg', '.preview.png']
    frontend_source = os.path.join(here, 'front_end')

    if PreviewPath == "false":
        subName = str(folder_paths.folder_names_and_paths[SubdirName][0][0])
        modelHomes = [f.path for f in os.scandir(subName) if f.is_dir()]
        imagefiles = []
        for modelHome in modelHomes:
            dirName = os.path.basename(os.path.normpath(modelHome))
            allFiles = [os.path.join(dirName, os.path.basename(x)) for x in glob.glob(modelHome + '/**/*', recursive=True)]
            imgFiles = folder_paths.filter_files_extensions(allFiles, supportedImages)
            imagefiles.extend(imgFiles)
    else:
        # subdir = os.path.join(utility.comfy_dir, 'web', 'extensions', 'PrimerePreviews', 'images', str(SubdirName))
        subdir = os.path.join(frontend_source, 'images', str(SubdirName))
        rootSubdir = Path(subdir).parent
        folder_paths.add_model_folder_path("previewpics_legacy" + SubdirName, subdir)
        allfiles = folder_paths.get_filename_list("previewpics_legacy" + SubdirName)
        imagefiles = folder_paths.filter_files_extensions(allfiles, supportedImages)

    imgbase_tuple = {}
    for imagefile in imagefiles:
        if PreviewPath == "false":
            image_path = os.path.abspath(folder_paths.get_full_path(SubdirName, imagefile))
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

routes11b = PromptServer.instance.routes
@routes11b.post('/primere_get_similarity') # ReadSimilarity()
async def primere_get_similarity(request):
    post = await request.post()
    subdir_name = post.get('SubdirName')
    preview_path = post.get('PreviewPath')
    reference_name = post.get('SelectedModel')
    supported_images = ['.jpg', '.png', '.jpeg', '.preview.jpg', '.preview.jpeg', '.preview.png']
    preview_source = os.path.join(here, 'front_end')

    imagefiles = []
    if preview_path == "false":
        sub_name = str(folder_paths.folder_names_and_paths[subdir_name][0][0])
        model_homes = [f.path for f in os.scandir(sub_name) if f.is_dir()]
        for model_home in model_homes:
            dir_name = os.path.basename(os.path.normpath(model_home))
            all_files = [os.path.join(dir_name, os.path.basename(x)) for x in glob.glob(model_home + '/**/*', recursive=True)]
            img_files = folder_paths.filter_files_extensions(all_files, supported_images)
            imagefiles.extend(img_files)
    else:
        subdir = os.path.join(preview_source, 'images', str(subdir_name))
        folder_paths.add_model_folder_path("previewpics_similarity" + subdir_name, subdir)
        allfiles = folder_paths.get_filename_list("previewpics_similarity" + subdir_name)
        imagefiles = folder_paths.filter_files_extensions(allfiles, supported_images)

    image_lookup = {}
    for imagefile in imagefiles:
        if preview_path == "false":
            image_path = os.path.abspath(folder_paths.get_full_path(subdir_name, imagefile))
        else:
            image_path = os.path.abspath(folder_paths.get_full_path('previewpics_similarity' + subdir_name, imagefile))
        filename = Path(image_path).stem.replace('.preview', '')
        if os.path.isfile(image_path):
            image_lookup[filename] = image_path

    similarity_data = {}
    if len(image_lookup) > 1:
        reference_clean = Path(str(reference_name)).name
        reference_stem = Path(reference_clean).stem
        if reference_stem in image_lookup:
            ordered_images = [image_lookup[reference_stem]]
            ordered_images.extend([v for k, v in image_lookup.items() if k != reference_stem])
        else:
            ordered_images = list(image_lookup.values())

        similarity_data = image_similarity.img_similarity(ordered_images)

    PromptServer.instance.send_sync("SimilarityData", similarity_data)
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
        is_link = os.path.islink(str(singleFile))
        if is_link == False and singleFile is not None and type(singleFile).__name__ != "NoneType" and type(singleFile).__name__ == "str" and os.path.isfile(singleFile) == True:
            filenameonly = Path(singleFile).stem
            singlefiledate = os.path.getctime(singleFile)
            filedates[filenameonly] = singlefiledate
        else:
            singleFile_link = Path(str(singleFile)).resolve()
            if os.path.isfile(singleFile_link) == True:
                filenameonly = Path(singleFile_link).stem
                singlefiledate = os.path.getctime(singleFile_link)
                filedates[filenameonly] = singlefiledate

    PromptServer.instance.send_sync("FileDateData", filedates)
    return web.json_response({})

routes13 = PromptServer.instance.routes
@routes13.post('/primere_get_filelinks') # ReadFileSymlink()
async def primere_get_filelinks(request):
    post = await request.post()
    subdirKey = post.get('type')
    allSource = folder_paths.get_filename_list(subdirKey)
    filelinktypes = {}

    for filename in allSource:
        singleFile = folder_paths.get_full_path(subdirKey, filename)
        is_link = os.path.islink(str(singleFile))
        if is_link == True:
            singleFile_link = Path(str(singleFile)).resolve()
            if os.path.isfile(singleFile_link) == True:
                filenameonly = Path(singleFile).stem
                # comfyModelDir = os.path.join(utility.comfy_dir, 'models')
                comfyModelDir = str(Path(folder_paths.folder_names_and_paths[subdirKey][0][0]).parent)
                modelType = str(singleFile_link)[len(comfyModelDir) + 1:str(singleFile_link).find('\\', len(comfyModelDir) + 1)]
                linkName_U = str(folder_paths.folder_names_and_paths["diffusion_models"][0][0])
                linkName_D = str(folder_paths.folder_names_and_paths["diffusion_models"][0][1])
                if str(Path(linkName_U).stem + '\\') in str(singleFile_link):
                    modelType = str(Path(linkName_U).stem)
                if str(Path(linkName_D).stem + '\\') in str(singleFile_link):
                    modelType = str(Path(linkName_D).stem)
                filelinktypes[filenameonly] = modelType

    PromptServer.instance.send_sync("FileLinkData", filelinktypes)
    return web.json_response({})

routes14 = PromptServer.instance.routes
@routes14.post('/primere_get_stime') # ReadSTimes()
async def primere_get_ascores(request):
    post = await request.post()
    data_type = post.get('type') + '_samplingtime'
    stimedata = utility.get_category_from_cache(data_type)
    PromptServer.instance.send_sync("STimeData", stimedata)
    return web.json_response({})

routes15 = PromptServer.instance.routes
@routes15.post('/primere_prompt_data') # getPromptData()
async def primere_prompt_data(request):
    post = await request.post()
    folder = post.get('folder')
    name = post.get('name')
    filetype = post.get('type')
    keys = post.get('keys').split(',')
    dresults = {}

    STYLE_DIR = os.path.join(PRIMERE_ROOT, folder)
    STYLE_SOURCE = os.path.join(STYLE_DIR, f'{name}.{filetype}')
    STYLE_DEV_SOURCE = os.path.join(STYLE_DIR, f'{name}.example.{filetype}')
    if os.path.isfile(STYLE_SOURCE) == False and os.path.isfile(STYLE_DEV_SOURCE) == True:
        shutil.copy(STYLE_DEV_SOURCE, STYLE_SOURCE)

    if os.path.isfile(STYLE_SOURCE):
        styles_csv = PrimereStyleLoader.load_styles_csv(str(STYLE_SOURCE))
        for req_key in keys:
            if req_key in styles_csv:
                req_list = list(set(styles_csv[req_key].values))
                cleaned_List = [x for x in req_list if str(x) != 'nan']
                dresults[req_key] = sorted(cleaned_List, key = str.lower)

    PromptServer.instance.send_sync("PromptDataResponse", dresults)
    return web.json_response({})

routes16 = PromptServer.instance.routes
@routes16.post('/primere_prompt_saver') # savePromptData()
async def primere_prompt_saver(request):
    post = await request.post()
    folder = post.get('folder')
    name = post.get('name')
    filetype = post.get('type')
    prompt_data = json.loads(post.get('promptdata'))
    myCsvRow = ''

    STYLE_DIR = os.path.join(PRIMERE_ROOT, folder)
    STYLE_SOURCE = os.path.join(STYLE_DIR, f'{name}.{filetype}')
    if os.path.isfile(STYLE_SOURCE):
        with open(STYLE_SOURCE, "r") as f:
            reader = csv.reader(f)
            for header in reader:
                break

            is_replace = prompt_data['replace']
            del prompt_data['replace']

            def dictsort(element):
                if element in header:
                    return header.index(element)
                else:
                    return len(header)
            prompt_data = dict(sorted(prompt_data.items(), key=lambda pair: dictsort(pair[0])))

            for prompt_key in prompt_data.keys():
                if prompt_data[prompt_key] == 'None':
                    myCsvRow = myCsvRow + '"",'
                else:
                    if prompt_key != 'name':
                        if prompt_key == 'preferred_model':
                            myCsvRow = myCsvRow + '"' + Path(prompt_data[prompt_key]).stem + '",'
                        else:
                            myCsvRow = myCsvRow + '"' + prompt_data[prompt_key].replace('"', '\'') + '",'
                    else:
                        myCsvRow = myCsvRow + prompt_data[prompt_key] + ','
            myCsvRow = "\n" + myCsvRow.rstrip(',"') + '"'

            if len(myCsvRow) > 5:
                if is_replace == 0:
                    try:
                        with open(STYLE_SOURCE, 'a') as fd:
                            fd.write(myCsvRow)
                        PromptServer.instance.send_sync("PromptDataSaveResponse", True)
                    except Exception:
                        PromptServer.instance.send_sync("PromptDataSaveResponse", False)
                elif is_replace == 1:
                    try:
                        file_encoding = utility.get_file_encoding(STYLE_SOURCE)
                        file_content = utility.open_file_by_chardet(STYLE_SOURCE)
                        if file_content is not None and file_encoding is not None and len(file_content) > 0:
                            parsed_row = 0
                            for file_row in file_content:
                                if file_row.startswith(prompt_data['name']):
                                    file_content[parsed_row] = myCsvRow.lstrip() + "\n"
                                parsed_row = parsed_row + 1
                            with open(STYLE_SOURCE, 'w', newline='', encoding=file_encoding) as target_file:
                                target_file.writelines(file_content)
                            PromptServer.instance.send_sync("PromptDataSaveResponse", True)
                        else:
                            PromptServer.instance.send_sync("PromptDataSaveResponse", False)
                    except Exception:
                        PromptServer.instance.send_sync("PromptDataSaveResponse", False)
                else:
                    PromptServer.instance.send_sync("PromptDataSaveResponse", False)
            else:
                PromptServer.instance.send_sync("PromptDataSaveResponse", False)
    else:
        PromptServer.instance.send_sync("PromptDataSaveResponse", False)
    return web.json_response({})

routes17 = PromptServer.instance.routes
@routes17.get('/primere_apiconfig_check')
async def primere_apiconfig_check(request):
    config_path = os.path.join(PRIMERE_ROOT, 'json', 'apiconfig.json')
    return web.json_response({"exists": os.path.isfile(config_path)})

routes18 = PromptServer.instance.routes
@routes18.post('/primere_model_concept_save')
async def primere_model_concept_save(request):
    post = await request.json()
    concept = post.get('concept')
    data = post.get('data')
    if not concept or data is None:
        return web.json_response({"success": False, "error": "Missing concept or data"}, status=400)
    json_path = os.path.join(PRIMERE_ROOT, 'front_end', 'model_concept.json')
    existing = utility.json2tuple(json_path) or {}
    existing[concept] = data
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2)
    return web.json_response({"success": True})

routes19 = PromptServer.instance.routes
@routes19.get('/primere_rasterix_read')
async def primere_rasterix_read(request):
    json_path = os.path.join(PRIMERE_ROOT, 'front_end', 'rasterix.json')
    data = utility.json2tuple(json_path) or {}
    return web.json_response(data)

routes19b = PromptServer.instance.routes
@routes19b.get('/primere_titles')
async def primere_titles(request):
    node_name = request.rel_url.query.get("node_name", "PrimereRasterix")
    section_map = _collect_section_titles()
    return web.json_response({"success": True, "sections": section_map.get(node_name, [])})

routes20 = PromptServer.instance.routes
@routes20.post('/primere_rasterix_save')
async def primere_rasterix_save(request):
    post = await request.json()
    section = post.get('section')
    data = post.get('data')
    if not section or data is None:
        return web.json_response({"success": False}, status=400)
    json_path = os.path.join(PRIMERE_ROOT, 'front_end', 'rasterix.json')
    existing = utility.json2tuple(json_path) or {}
    existing[section] = data
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2)
    return web.json_response({"success": True})

routes21 = PromptServer.instance.routes
@routes21.post('/primere_rasterix_histogram_generate')
async def primere_rasterix_histogram_generate(request):
    post = await request.json()
    node_id = post.get('node_id')
    histogram_source = bool(post.get('histogram_source', False))
    histogram_channel = post.get('histogram_channel', 'RGB')
    histogram_style = post.get('histogram_style', 'bars')
    precision = bool(post.get('precision', False))
    force = bool(post.get('force', False))

    _, input_cache, output_cache = histogram.rasterix_hist_cache_paths(node_id=node_id)

    source_path = input_cache if histogram_source else output_cache
    if not os.path.isfile(source_path):
        return web.json_response({"success": False, "error": "Histogram cache is missing"}, status=404)

    source_img = Image.open(source_path).convert("RGB")
    target_file = histogram.rasterix_hist_render_path(node_id, histogram_source, histogram_channel, histogram_style)
    if force or not os.path.isfile(target_file):
        rendered = histogram.rasterix_histogram_render(source_img, histogram_channel, histogram_style, precision)
        rendered.save(target_file, compress_level=1)
    return web.json_response({"success": True, "filename": os.path.basename(target_file)})

routes21b = PromptServer.instance.routes
@routes21b.get('/primere_rasterix_histogram_image')
async def primere_rasterix_histogram_image(request):
    node_id = request.rel_url.query.get("node_id")
    histogram_source = request.rel_url.query.get("histogram_source", "false").lower() == "true"
    histogram_channel = request.rel_url.query.get("histogram_channel", "RGB")
    histogram_style = request.rel_url.query.get("histogram_style", "bars")
    target_file = histogram.rasterix_hist_render_path(node_id, histogram_source, histogram_channel, histogram_style)
    if not os.path.isfile(target_file):
        return web.json_response({"success": False, "error": "Histogram image missing"}, status=404)
    return web.FileResponse(target_file)

routes22 = PromptServer.instance.routes
@routes22.get('/primere_rasterix_setting_read')
async def primere_rasterix_setting_read(request):
    json_path = os.path.join(PRIMERE_ROOT, 'front_end', 'rasterix_settings.json')
    data = utility.json2tuple(json_path) or {}
    return web.json_response(data)

routes23 = PromptServer.instance.routes
@routes23.post('/primere_rasterix_setting_save')
async def primere_rasterix_setting_save(request):
    post = await request.json()
    concept = post.get('concept')
    data = post.get('data')
    if not concept or data is None:
        return web.json_response({"success": False, "error": "Missing concept or data"}, status=400)
    json_path = os.path.join(PRIMERE_ROOT, 'front_end', 'rasterix_settings.json')
    existing = utility.json2tuple(json_path) or {}
    existing[concept] = data
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(existing, f, indent=2)
    return web.json_response({"success": True})

routes24 = PromptServer.instance.routes
@routes24.get('/primere_uniapi_schema_read')
async def primere_uniapi_schema_read(request):
    schema_path = os.path.join(PRIMERE_ROOT, 'front_end', 'api_schemas.json')
    schema_example_path = os.path.join(PRIMERE_ROOT, 'front_end', 'api_schemas.example.json')
    apiconfig_path = os.path.join(PRIMERE_ROOT, 'json', 'apiconfig.json')

    source_path = schema_path if os.path.isfile(schema_path) else schema_example_path
    source_name = os.path.basename(source_path)
    if not os.path.isfile(source_path):
        return web.json_response({"success": False, "error": "Schema file is missing", "source": source_name, "registry": {},}, status=404)

    try:
        source_data = Path(source_path).read_text(encoding='utf-8-sig')
    except UnicodeDecodeError:
        file_encoding = utility.get_file_encoding(source_path) or 'utf-8'
        source_data = Path(source_path).read_text(encoding=file_encoding)
    except Exception as e:
        return web.json_response({"success": False, "error": f"Cannot read schema file: {e}", "source": source_name, "registry": {},}, status=500)

    try:
        raw_schema = json.loads(source_data)
    except json.JSONDecodeError as e:
        return web.json_response({"success": False, "error": f"Invalid JSON syntax at line {e.lineno}, column {e.colno}: {e.msg}", "source": source_name, "registry": {},}, status=400)

    if not isinstance(raw_schema, dict):
        return web.json_response({"success": False, "error": "Schema root must be a JSON object.", "source": source_name, "registry": {},}, status=400)

    warning = None
    if os.path.isfile(apiconfig_path):
        try:
            registry = api_schema_registry.load_and_validate_api_schema_registry(source_path, apiconfig_path)
        except Exception as e:
            warning = str(e)
            registry = api_schema_registry.normalize_registry(raw_schema)
    else:
        warning = "API config file missing, loaded schema without provider validation."
        registry = api_schema_registry.normalize_registry(raw_schema)

    return web.json_response({"success": True, "source": source_name, "warning": warning, "registry": registry,})
