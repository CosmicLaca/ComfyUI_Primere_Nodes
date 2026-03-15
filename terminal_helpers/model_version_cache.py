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

EXCLUDED_SUBDIRS = {'.locks', 'Bjornulf_civitAI', 'depthfm', 'models--xiaozaa--cat-tryoff-flux'}

def match_supported(name):
    name_lower = name.lower()
    for supported in utility.SUPPORTED_MODELS:
        if supported.lower() == name_lower:
            return supported
    return None

def get_type_from_dirs(oneModel, resolved_path=None):
    parts = Path(oneModel).parts
    if len(parts) > 1:
        matched = match_supported(parts[0])
        if matched:
            return matched
    if resolved_path is not None:
        parent_dir = Path(resolved_path).parent.name
        matched = match_supported(parent_dir)
        if matched:
            return matched
    return None

ModelsList = folder_paths.get_filename_list("checkpoints")

print('------------------- START -------------------------')
print(str(len(ModelsList)) + ' models in system')
print('--------------- CACHED MODELS INFO ---------------------')

if len(ModelsList) > 0:
    model_counter = 1
    for oneModel in ModelsList:
        parts = Path(oneModel).parts
        first_part = parts[0] if len(parts) > 1 else ''
        if first_part in EXCLUDED_SUBDIRS:
            model_counter += 1
            continue

        model_path = folder_paths.get_full_path("checkpoints", oneModel)
        is_link = os.path.islink(str(model_path))
        modelname_only = Path(oneModel).stem
        resolved_path = Path(str(model_path)).resolve() if is_link else None
        prefix = f"Model [{model_counter}] / {len(ModelsList)}"

        model_version = utility.get_value_from_cache('model_version', modelname_only)
        if model_version is not None and model_version in utility.SUPPORTED_MODELS:
            print(f"{prefix} already cached: {modelname_only} -> {model_version}")
            model_counter += 1
            continue

        model_version = utility.getModelType(oneModel, 'checkpoints')
        if model_version and model_version not in (None, False, 'NoneType') and model_version in utility.SUPPORTED_MODELS:
            utility.add_value_to_cache('model_version', modelname_only, model_version)
            src = f" (symlink from: {resolved_path})" if is_link else ""
            print(f"{prefix} cached from metadata: {modelname_only} -> {model_version}{src}")
            model_counter += 1
            continue

        dir_version = get_type_from_dirs(oneModel, resolved_path)
        if dir_version:
            utility.add_value_to_cache('model_version', modelname_only, dir_version)
            src = f" (symlink from: {resolved_path})" if is_link else ""
            print(f"{prefix} cached from directory: {modelname_only} -> {dir_version}{src}")
            model_counter += 1
            continue

        if is_link:
            cache_msg = f"UNKNOWN | checkpoint: models/checkpoints/{oneModel} | original: {resolved_path}"
        else:
            cache_msg = f"UNKNOWN | path: models/checkpoints/{oneModel}"
        utility.add_value_to_cache('model_version', modelname_only, cache_msg)
        print(f"{prefix} {cache_msg}")

        model_counter += 1
else:
    print('No models in your system....')
