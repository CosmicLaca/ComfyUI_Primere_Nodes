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

EmbeddingList = folder_paths.get_filename_list("embeddings")

print('------------------- START -------------------------')
print(str(len(EmbeddingList)) + ' embeddings in system')

print('--------------- CACHED EMBEDDING INFO ---------------------')
if len(EmbeddingList) > 0:
    model_counter = 1
    for onelora in EmbeddingList:
        name_only = Path(onelora).stem
        model_version = utility.get_value_from_cache('embedding_version', name_only)
        if model_version is None or model_version not in utility.SUPPORTED_MODELS:
            model_version = 'SD15'
            utility.add_value_to_cache('embedding_version', name_only, model_version)
            print('Embedding cached: ' + name_only + ' -> ' + str(model_version))
        else:
            print('Embedding already cached: ' + name_only + ' -> ' + str(model_version))
        model_counter = model_counter + 1
else:
    print('No embedding in your system....')