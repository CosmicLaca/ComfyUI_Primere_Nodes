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

ModelsList = folder_paths.get_filename_list("checkpoints")

print('------------------- START -------------------------')
print(str(len(ModelsList)) + ' models in system')

print('--------------- CACHED MODELS INFO ---------------------')
if len(ModelsList) > 0:
    model_counter = 1
    for oneModel in ModelsList:
        model_path = folder_paths.get_full_path("checkpoints", oneModel)
        is_link = os.path.islink(str(model_path))
        modelaname_only = Path(oneModel).stem
        if is_link == False:
            model_version = utility.get_value_from_cache('model_version', modelaname_only)
            if model_version is None or model_version not in utility.SUPPORTED_MODELS:
                model_version = utility.getModelType(oneModel, 'checkpoints')
                if model_version and model_version is not None and model_version != 'NoneType':
                    utility.add_value_to_cache('model_version', modelaname_only, str(model_version))
                    print('Model [' + str(model_counter) + '] / ' + str(len(ModelsList)) + ' cached: ' + modelaname_only + ' -> ' + str(model_version))
                else:
                    utility.add_value_to_cache('model_version', modelaname_only, 'unknown')
                    print('Model [' + str(model_counter) + '] / ' + str(len(ModelsList)) + ' cached: ' + modelaname_only + ' -> ' + 'unknown')
            else:
                print('Model [' + str(model_counter) + '] / ' + str(len(ModelsList)) + ' already cached: ' + modelaname_only + ' -> ' + str(model_version))
        else:
            model_version = utility.get_value_from_cache('model_version', modelaname_only)
            if model_version is None or model_version not in utility.SUPPORTED_MODELS:
                File_link = Path(str(model_path)).resolve()
                comfyModelDir = str(Path(folder_paths.folder_names_and_paths['checkpoints'][0][0]).parent)
                # modelType = str(File_link)[(len(comfyModelDir) + 1):(str(File_link).find('\\', len(comfyModelDir) + 1))]
                try:
                    modelType = str(File_link)[str(File_link).index(os.sep + Path(comfyModelDir).stem) + len(Path(comfyModelDir).stem) + 2:(str(File_link).find(os.sep, len(comfyModelDir) + 1))]
                    linkName_U = str(folder_paths.folder_names_and_paths["diffusion_models"][0][0])
                    linkName_D = str(folder_paths.folder_names_and_paths["diffusion_models"][0][1])
                    if str(Path(linkName_U).stem + '\\') in str(File_link):
                        modelType = str(Path(linkName_U).stem)
                    if str(Path(linkName_D).stem + '\\') in str(File_link):
                        modelType = str(Path(linkName_D).stem)

                    utility.add_value_to_cache('model_version', modelaname_only, f"{modelType}_symlink")
                    print(f"Model [{model_counter}] / {len(ModelsList)} symlinked file: {modelaname_only} -> from: {modelType}")
                except Exception:
                    utility.add_value_to_cache('model_version', modelaname_only, f"unknown_symlink")
                    print(f"Model [{model_counter}] / {len(ModelsList)} unknown by meta error: {modelaname_only}")
            else:
                print('Model [' + str(model_counter) + '] / ' + str(len(ModelsList)) + ' already cached: ' + modelaname_only + ' -> ' + str(model_version))
        model_counter = model_counter + 1
else:
    print('No models in your system....')