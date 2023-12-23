from pathlib import Path
import sys

here = Path(__file__).parent.parent.absolute()
comfy_dir = str(here.parent.parent)

sys.path.append(comfy_dir)
import os
import folder_paths
import custom_nodes.ComfyUI_Primere_Nodes.components.utility as utility
import comfy

here = Path(__file__).parent.parent.absolute()
comfy_dir = here.parent.parent

LYCO_DIR = os.path.join(comfy_dir, 'models', 'lycoris')
folder_paths.add_model_folder_path("lycoris", LYCO_DIR)
LyCORIS = folder_paths.get_filename_list("lycoris")
LyCORISList = folder_paths.filter_files_extensions(LyCORIS, ['.ckpt', '.safetensors'])

print('--------------- CACHED LYCORIS INFO ---------------------')
if len(LyCORISList) > 0:
    for oneLyco in LyCORISList:
        lora_path = folder_paths.get_full_path("lycoris", oneLyco)
        lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
        loraname_only = Path(oneLyco).stem
        model_lora_version = utility.get_value_from_cache('lycoris_version', loraname_only)
        if model_lora_version is None:
            loraVER = utility.getLoraVersion(lora)
            if loraVER is not None:
                utility.add_value_to_cache('lycoris_version', loraname_only, loraVER)
                print('cached: ' + loraname_only + ' -> ' + str(loraVER))
        else:
            print('Already cached: ' + loraname_only + ' -> ' + str(model_lora_version))
else:
    print('No lycos in your system....')