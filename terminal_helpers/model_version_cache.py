from pathlib import Path
import sys

here = Path(__file__).parent.parent.absolute()
comfy_dir = str(here.parent.parent)
sys.path.append(comfy_dir)

import folder_paths
import custom_nodes.ComfyUI_Primere_Nodes.components.utility as utility
import nodes

ModelsList = folder_paths.get_filename_list("checkpoints")
chkp_loader = nodes.CheckpointLoaderSimple()

print('--------------- CACHED MODELS INFO ---------------------')
if len(ModelsList) > 0:
    for oneModel in ModelsList:
        lora_path = folder_paths.get_full_path("checkpoints", oneModel)
        loraname_only = Path(oneModel).stem
        model_version = utility.get_value_from_cache('model_version', loraname_only)
        if model_version is None:
            LOADED_CHECKPOINT = chkp_loader.load_checkpoint(oneModel)
            model_version = utility.getCheckpointVersion(LOADED_CHECKPOINT[0])
            if model_version is not None:
                utility.add_value_to_cache('model_version', loraname_only, model_version)
                print('cached: ' + loraname_only + ' -> ' + model_version)
        else:
            print('Already cached: ' + loraname_only + ' -> ' + str(model_version))
else:
    print('No models in your system....')