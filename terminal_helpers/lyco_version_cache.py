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

import utility as utility

LYCO_DIR = os.path.join(folder_paths.models_dir, 'lycoris')
folder_paths.add_model_folder_path("lycoris", LYCO_DIR)
LyCORIS = folder_paths.get_filename_list("lycoris")
ModelsList = folder_paths.filter_files_extensions(LyCORIS, ['.ckpt', '.safetensors'])

print('------------------- START -------------------------')
print(str(len(ModelsList)) + ' lycoris in system')

print('--------------- CACHED LYCORIS INFO ---------------------')
if len(ModelsList) > 0:
    model_counter = 1
    for oneModel in ModelsList:
        model_path = folder_paths.get_full_path("lycoris", oneModel)
        modelaname_only = Path(oneModel).stem
        model_version = utility.get_value_from_cache('lycoris_version', modelaname_only)
        if model_version is None or model_version not in utility.SUPPORTED_MODELS:
            model_version = utility.getModelType(oneModel, 'lycoris')
            if model_version and model_version is not None and model_version != 'NoneType':
                utility.add_value_to_cache('lycoris_version', modelaname_only, str(model_version))
                print('Lyco [' + str(model_counter) + '] / ' + str(len(ModelsList)) + ' cached: ' + modelaname_only + ' -> ' + str(model_version))
            else:
                utility.add_value_to_cache('lycoris_version', modelaname_only, 'unknown')
                print('Lyco [' + str(model_counter) + '] / ' + str(len(ModelsList)) + ' cached: ' + modelaname_only + ' -> ' + 'unknown')
        else:
            print('Lyco [' + str(model_counter) + '] / ' + str(len(ModelsList)) + ' already cached: ' + modelaname_only + ' -> ' + str(model_version))
        model_counter = model_counter + 1
else:
    print('No lycoris in your system....')