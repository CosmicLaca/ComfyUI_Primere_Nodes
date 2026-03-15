from pathlib import Path
import sys
import os

here = Path(__file__).parent.parent.absolute()
comfy_dir = str(here.parent.parent)
sys.path.append(comfy_dir)

import folder_paths

primere_root = Path(__file__).parent.parent.absolute()
components_path = os.path.join(primere_root, 'components')
sys.path.append(components_path)

import utility

LYCO_DIR = os.path.join(folder_paths.models_dir, 'lycoris')
folder_paths.add_model_folder_path("lycoris", LYCO_DIR)
LyCORIS = folder_paths.get_filename_list("lycoris")
ModelsList = folder_paths.filter_files_extensions(LyCORIS, ['.ckpt', '.safetensors'])

def match_supported(name):
    name_lower = name.lower()
    for supported in utility.SUPPORTED_MODELS:
        if supported.lower() == name_lower:
            return supported
    return None

def get_type_from_dir(oneModel):
    parts = Path(oneModel).parts
    if len(parts) > 1:
        return match_supported(parts[0])
    return None

print('------------------- START -------------------------')
print(str(len(ModelsList)) + ' lycoris in system')
print('--------------- CACHED LYCORIS INFO ---------------------')

if len(ModelsList) > 0:
    model_counter = 1
    for oneModel in ModelsList:
        modelname_only = Path(oneModel).stem
        prefix = f"Lyco [{model_counter}] / {len(ModelsList)}"

        model_version = utility.get_value_from_cache('lycoris_version', modelname_only)
        if model_version is not None and model_version in utility.SUPPORTED_MODELS:
            print(f"{prefix} already cached: {modelname_only} -> {model_version}")
            model_counter += 1
            continue

        model_version = utility.getModelType(oneModel, 'lycoris')
        if model_version and model_version not in (None, False, 'NoneType') and model_version in utility.SUPPORTED_MODELS:
            utility.add_value_to_cache('lycoris_version', modelname_only, model_version)
            print(f"{prefix} cached from metadata: {modelname_only} -> {model_version}")
            model_counter += 1
            continue

        dir_version = get_type_from_dir(oneModel)
        if dir_version:
            utility.add_value_to_cache('lycoris_version', modelname_only, dir_version)
            print(f"{prefix} cached from directory: {modelname_only} -> {dir_version}")
            model_counter += 1
            continue

        cache_msg = f"UNKNOWN | path: models/lycoris/{oneModel}"
        utility.add_value_to_cache('lycoris_version', modelname_only, cache_msg)
        print(f"{prefix} {cache_msg}")

        model_counter += 1
else:
    print('No lycoris in your system....')
