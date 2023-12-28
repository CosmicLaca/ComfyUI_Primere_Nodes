from pathlib import Path
import sys

here = Path(__file__).parent.parent.absolute()
comfy_dir = str(here.parent.parent)
sys.path.append(comfy_dir)

import folder_paths
import custom_nodes.ComfyUI_Primere_Nodes.components.utility as utility

EmbeddingList = folder_paths.get_filename_list("embeddings")

print('--------------- CACHED EMBEDDING INFO ---------------------')
if len(EmbeddingList) > 0:
    for onelora in EmbeddingList:
        loraname_only = Path(onelora).stem
        model_lora_version = utility.get_value_from_cache('embedding_version', loraname_only)
        if model_lora_version is None:
            loraVER = 'SD'
            utility.add_value_to_cache('embedding_version', loraname_only, loraVER)
            print('cached: ' + loraname_only + ' -> ' + str(loraVER))
        else:
            print('Already cached: ' + loraname_only + ' -> ' + str(model_lora_version))
else:
    print('No embedding in your system....')