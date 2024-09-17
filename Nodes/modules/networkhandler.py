import folder_paths
import comfy.sd
import comfy.utils
from ...components import utility
import os
from ...components.tree import PRIMERE_ROOT
import random
from pathlib import Path
from ...components import hypernetwork

def LoraHandler(self, kwargs, model, clip, model_keyword, use_only_model_weight, lora_keywords_num, use_lora_keyword, lora_keyword_selection, lora_keyword_weight, lora_keyword_placement):
    loras = [kwargs.get(f"lora_{i}") for i in range(1, self.LORASCOUNT + 1)]
    model_weight = [kwargs.get(f"lora_{i}_model_weight") for i in range(1, self.LORASCOUNT + 1)]
    if use_only_model_weight == True:
        clip_weight = [kwargs.get(f"lora_{i}_model_weight") for i in range(1, self.LORASCOUNT + 1)]
    else:
        clip_weight = [kwargs.get(f"lora_{i}_clip_weight") for i in range(1, self.LORASCOUNT + 1)]

    uses = [kwargs.get(f"use_lora_{i}") for i in range(1, self.LORASCOUNT + 1)]
    lora_stack = [(lora_name, lora_model_weight, lora_clip_weight) for lora_name, lora_model_weight, lora_clip_weight, lora_uses in  zip(loras, model_weight, clip_weight, uses) if lora_uses == True]

    lora_params = list()
    if lora_stack and len(lora_stack) > 0:
        lora_params.extend(lora_stack)
    else:
        return (model, clip, lora_stack, model_keyword)

    model_lora = model
    clip_lora = clip
    list_of_keyword_items = []
    lora_keywords_num_set = lora_keywords_num

    for tup in lora_params:
        lora_name, strength_model, strength_clip = tup

        lora_path = folder_paths.get_full_path("loras", lora_name)
        if lora_path:
            lora = comfy.utils.load_torch_file(lora_path, safe_load=True)
            model_lora, clip_lora = comfy.sd.load_lora_for_models(model_lora, clip_lora, lora, strength_model, strength_clip)

        loraname_only = Path(lora_name).stem
        model_lora_version = utility.get_value_from_cache('lora_version', loraname_only)
        if model_lora_version is None:
            loraVER = utility.getLoraVersion(lora)
            if loraVER is not None:
                utility.add_value_to_cache('lora_version', loraname_only, loraVER)

        if use_lora_keyword == True:
            ModelKvHash = utility.get_model_hash(lora_path)
            if ModelKvHash is not None:
                KEYWORD_PATH = os.path.join(PRIMERE_ROOT, 'front_end', 'keywords', 'lora-keyword.txt')
                keywords = utility.get_model_keywords(KEYWORD_PATH, ModelKvHash, lora_name)
                if keywords is not None and keywords != "" and isinstance(keywords, str) == True:
                    if keywords.find('|') > 1:
                        keyword_list = [word.strip() for word in keywords.split('|')]
                        keyword_list = list(filter(None, keyword_list))
                        if (len(keyword_list) > 0):
                            lora_keywords_num = lora_keywords_num_set
                            keyword_qty = len(keyword_list)
                            if (lora_keywords_num > keyword_qty):
                                lora_keywords_num = keyword_qty
                            if lora_keyword_selection == 'Select in order':
                                list_of_keyword_items.extend(keyword_list[:lora_keywords_num])
                            else:
                                list_of_keyword_items.extend(random.sample(keyword_list, lora_keywords_num))
                    else:
                        list_of_keyword_items.append(keywords)

    if len(list_of_keyword_items) > 0:
        if lora_keyword_selection != 'Select in order':
            random.shuffle(list_of_keyword_items)

        list_of_keyword_items = list(set(list_of_keyword_items))
        keywords = ", ".join(list_of_keyword_items)

        if (lora_keyword_weight != 1):
            keywords = '(' + keywords + ':' + str(round(lora_keyword_weight, 1)) + ')'

        model_keyword = [keywords, lora_keyword_placement]

    return (model_lora, clip_lora, lora_stack, model_keyword)

def EmbeddingHandler(self, kwargs, embedding_placement_pos, embedding_placement_neg):
    embedding_pos_list = []
    embedding_neg_list = []

    embeddings = [kwargs.get(f"embedding_{i}") for i in range(1, self.EMBCOUNT + 1)]
    use_embeddings = [kwargs.get(f"use_embedding_{i}") for i in range(1, self.EMBCOUNT + 1)]
    embedding_weight = [kwargs.get(f"embedding_{i}_weight") for i in range(1, self.EMBCOUNT + 1)]
    neg_embedding = [kwargs.get(f"is_negative_{i}") for i in range(1, self.EMBCOUNT + 1)]

    embedding_stack = [(emb_name, emb_weight, is_emb_neg) for emb_name, emb_weight, is_emb_neg, emb_uses in zip(embeddings, embedding_weight, neg_embedding, use_embeddings) if emb_uses == True]
    if embedding_stack is not None and len(embedding_stack) > 0:
        for embedding_tuple in embedding_stack:
            embedd_name_path = embedding_tuple[0]
            embedd_weight = embedding_tuple[1]
            embedd_neg = embedding_tuple[2]
            embedd_name = Path(embedd_name_path).stem

            if (embedd_weight != 1):
                embedding_sting = '(embedding:' + embedd_name + ':' + str(embedd_weight) + ')'
            else:
                embedding_sting = 'embedding:' + embedd_name

            if embedd_neg == False:
                embedding_pos_list.append(embedding_sting)
            else:
                embedding_neg_list.append(embedding_sting)

    else:
        return ([None, None], [None, None], [])

    embedding_pos_list = list(set(embedding_pos_list))
    embedding_neg_list = list(set(embedding_neg_list))

    if len(embedding_pos_list) > 0:
        embedding_pos = ", ".join(embedding_pos_list)
    else:
        embedding_pos = None
        embedding_placement_pos = None

    if len(embedding_neg_list) > 0:
        embedding_neg = ", ".join(embedding_neg_list)
    else:
        embedding_neg = None
        embedding_placement_neg = None

    return ([embedding_pos, embedding_placement_pos], [embedding_neg, embedding_placement_neg], embedding_stack)

def HypernetworkHandler(self, kwargs, model, safe_load):
    model_hypernetwork = model
    hnetworks = [kwargs.get(f"hypernetwork_{i}") for i in range(1, self.HNCOUNT + 1)]
    use_hnetworks = [kwargs.get(f"use_hypernetwork_{i}") for i in range(1, self.HNCOUNT + 1)]
    hnetworks_weight = [kwargs.get(f"hypernetwork_{i}_weight") for i in range(1, self.HNCOUNT + 1)]

    hnetwork_stack = [(hn_name, hn_weight) for hn_name, hn_weight, hn_uses in zip(hnetworks, hnetworks_weight, use_hnetworks) if hn_uses == True]
    if hnetwork_stack is not None and len(hnetwork_stack) > 0:
        cloned_model = model
        for hn_tuple in hnetwork_stack:
            hypernetwork_path = folder_paths.get_full_path("hypernetworks", hn_tuple[0])
            if hypernetwork_path:
                model_hypernetwork = cloned_model.clone()
                try:
                    patch = hypernetwork.load_hypernetwork_patch(hypernetwork_path, hn_tuple[1], safe_load)
                except Exception:
                    patch = None
                if patch is not None:
                    model_hypernetwork.set_model_attn1_patch(patch)
                    model_hypernetwork.set_model_attn2_patch(patch)
                    cloned_model = model_hypernetwork
    else:
        return (model, [],)

    return (model_hypernetwork, hnetwork_stack,)

def LycorisHandler(self, kwargs, model, clip, model_keyword, use_only_model_weight, lycoris_keywords_num, use_lycoris_keyword, lycoris_keyword_selection, lycoris_keyword_weight, lycoris_keyword_placement):
    lycoris = [kwargs.get(f"lycoris_{i}") for i in range(1, self.LYCOSCOUNT + 1)]
    model_weight = [kwargs.get(f"lycoris_{i}_model_weight") for i in range(1, self.LYCOSCOUNT + 1)]
    if use_only_model_weight == True:
        clip_weight = [kwargs.get(f"lycoris_{i}_model_weight") for i in range(1, self.LYCOSCOUNT + 1)]
    else:
        clip_weight = [kwargs.get(f"lycoris_{i}_clip_weight") for i in range(1, self.LYCOSCOUNT + 1)]

    uses = [kwargs.get(f"use_lycoris_{i}") for i in range(1, self.LYCOSCOUNT + 1)]
    lycoris_stack = [(lycoris_name, lycoris_model_weight, lycoris_clip_weight) for lycoris_name, lycoris_model_weight, lycoris_clip_weight, lycoris_uses in zip(lycoris, model_weight, clip_weight, uses) if lycoris_uses == True]

    lycoris_params = list()
    if lycoris_stack and len(lycoris_stack) > 0:
        lycoris_params.extend(lycoris_stack)
    else:
        return (model, clip, lycoris_stack, model_keyword)

    model_lyco = model
    clip_lyco = clip
    list_of_keyword_items = []
    lycoris_keywords_num_set = lycoris_keywords_num

    for tup in lycoris_params:
        lycoris_name, strength_model, strength_clip = tup

        lycoris_path = folder_paths.get_full_path("lycoris", lycoris_name)
        if lycoris_path:
            lyco = comfy.utils.load_torch_file(lycoris_path, safe_load=True)
            model_lyco, clip_lyco = comfy.sd.load_lora_for_models(model_lyco, clip_lyco, lyco, strength_model, strength_clip)

            lyconame_only = Path(lycoris_name).stem
            model_lyco_version = utility.get_value_from_cache('lycoris_version', lyconame_only)
            if model_lyco_version is None:
                lycoVER = utility.getLoraVersion(lyco)
                if lycoVER is not None:
                    utility.add_value_to_cache('lycoris_version', lyconame_only, lycoVER)

            if use_lycoris_keyword == True:
                ModelKvHash = utility.get_model_hash(lycoris_path)
                if ModelKvHash is not None:
                    KEYWORD_PATH = os.path.join(PRIMERE_ROOT, 'front_end', 'keywords', 'lora-keyword.txt')
                    keywords = utility.get_model_keywords(KEYWORD_PATH, ModelKvHash, lycoris_name)
                    if keywords is not None and keywords != "" and isinstance(keywords, str) == True:
                        if keywords.find('|') > 1:
                            keyword_list = [word.strip() for word in keywords.split('|')]
                            keyword_list = list(filter(None, keyword_list))
                            if (len(keyword_list) > 0):
                                lycoris_keywords_num = lycoris_keywords_num_set
                                keyword_qty = len(keyword_list)
                                if (lycoris_keywords_num > keyword_qty):
                                    lycoris_keywords_num = keyword_qty
                                if lycoris_keyword_selection == 'Select in order':
                                    list_of_keyword_items.extend(keyword_list[:lycoris_keywords_num])
                                else:
                                    list_of_keyword_items.extend(random.sample(keyword_list, lycoris_keywords_num))
                        else:
                            list_of_keyword_items.append(keywords)

    if len(list_of_keyword_items) > 0:
        if lycoris_keyword_selection != 'Select in order':
            random.shuffle(list_of_keyword_items)

        list_of_keyword_items = list(set(list_of_keyword_items))
        keywords = ", ".join(list_of_keyword_items)

        if (lycoris_keyword_weight != 1):
            keywords = '(' + keywords + ':' + str(round(lycoris_keyword_weight, 1)) + ')'

        model_keyword = [keywords, lycoris_keyword_placement]

    return (model_lyco, clip_lyco, lycoris_stack, model_keyword)

def getNetworkLoader(workflow_tuple, network_key, network_cunt, clip_weigth, stack_version):
    KeyList = list(workflow_tuple['network_data'].keys())
    loader = {}
    network_count = 0

    if clip_weigth:
        empty_network = [False, f"no_{network_key}", 0, 0]
        weigth_key = 'model_'
    else:
        empty_network = [False, f"no_{network_key}", 0, False]
        weigth_key = ''

    loaded_network = list()
    for network_key_check in KeyList:
        if network_key_check.startswith(f"{network_key}_"):
            if network_key_check in workflow_tuple['network_data'] and len(workflow_tuple['network_data'][network_key_check]) > 0:
                for network_data in workflow_tuple['network_data'][network_key_check]:
                    if network_data[0] not in loaded_network:
                        # if folder_paths.get_full_path("loras", network_data[0]):
                        network_count = network_count + 1
                        loader[f"use_{network_key}_{network_count}"] = True
                        loader[f"{network_key}_{network_count}"] = network_data[0]
                        loader[f"{network_key}_{network_count}_{weigth_key}weight"] = network_data[1]
                        if clip_weigth:
                            loader[f"{network_key}_{network_count}_clip_weight"] = network_data[2]
                        if network_key == 'embedding':
                            loader[f"is_negative_{network_count}"] = network_data[2]
                        loaded_network.append(network_data[0])

    network_stack_diff = network_cunt - network_count
    if network_stack_diff > 0:
        for i in range(network_count + 1, network_cunt + 1):
            loader[f"use_{network_key}_{i}"] = empty_network[0]
            loader[f"{network_key}_{i}"] = empty_network[1]
            loader[f"{network_key}_{i}_{weigth_key}weight"] = empty_network[2]
            if clip_weigth:
                loader[f"{network_key}_{i}_clip_weight"] = empty_network[3]
            if network_key == 'embedding':
                loader[f"is_negative_{i}"] = empty_network[3]

    return loader