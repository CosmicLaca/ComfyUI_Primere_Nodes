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

EmbeddingList = folder_paths.get_filename_list("embeddings")

def match_supported(name):
    name_lower = name.lower()
    for supported in utility.SUPPORTED_MODELS:
        if supported.lower() == name_lower:
            return supported
    return None

def get_type_from_dir(oneEmbedding):
    parts = Path(oneEmbedding).parts
    if len(parts) > 1:
        return match_supported(parts[0])
    return None

print('------------------- START -------------------------')
print(str(len(EmbeddingList)) + ' embeddings in system')
print('--------------- CACHED EMBEDDING INFO ---------------------')

if len(EmbeddingList) > 0:
    model_counter = 1
    for oneEmbedding in EmbeddingList:
        name_only = Path(oneEmbedding).stem
        prefix = f"Embedding [{model_counter}] / {len(EmbeddingList)}"

        model_version = utility.get_value_from_cache('embedding_version', name_only)
        if model_version is not None and model_version in utility.SUPPORTED_MODELS:
            print(f"{prefix} already cached: {name_only} -> {model_version}")
            model_counter += 1
            continue

        dir_version = get_type_from_dir(oneEmbedding)
        if dir_version:
            utility.add_value_to_cache('embedding_version', name_only, dir_version)
            print(f"{prefix} cached from directory: {name_only} -> {dir_version}")
            model_counter += 1
            continue

        cache_msg = f"UNKNOWN | path: models/embeddings/{oneEmbedding}"
        utility.add_value_to_cache('embedding_version', name_only, cache_msg)
        print(f"{prefix} {cache_msg}")

        model_counter += 1
else:
    print('No embeddings in your system....')
