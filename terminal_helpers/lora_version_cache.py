from pathlib import Path
import sys

here = Path(__file__).parent.parent.absolute()
comfy_dir = str(here.parent.parent)
sys.path.append(comfy_dir)

import folder_paths
import custom_nodes.ComfyUI_Primere_Nodes.components.utility as utility
import comfy

LoraList = folder_paths.get_filename_list("loras")

print('--------------- CACHED LORAS INFO ---------------------')
if len(LoraList) > 0:
    for onelora in LoraList:
        lora_path = folder_paths.get_full_path("loras", onelora)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        loraname_only = Path(onelora).stem
        model_lora_version = utility.get_value_from_cache('lora_version', loraname_only)
        if model_lora_version is None:
            loraVER = utility.getLoraVersion(lora)
            if loraVER is not None:
                utility.add_value_to_cache('lora_version', loraname_only, loraVER)
                print('cached: ' + loraname_only + ' -> ' + str(loraVER))
        else:
            print('Already cached: ' + loraname_only + ' -> ' + str(model_lora_version))
else:
    print('No loras in your system....')