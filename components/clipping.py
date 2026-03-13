import torch
import gc
from .long_clip_model import longclip
from comfy.sd1_clip import load_embed, ClipTokenWeightEncoder
from comfy.sd1_clip import token_weights, escape_important, unescape_important
from comfy import model_management
import comfy
import comfy_extras.nodes_sd3 as nodes_sd3
import comfy_extras.nodes_flux as nodes_flux
from . import utility
import nodes
from ..Nodes.modules import long_clip as long_clip_module
from .sana.diffusion.model.utils import prepare_prompt_ar
from .sana.diffusion.data.datasets.utils import ASPECT_RATIO_1024_TEST

class SDLongClipModel(torch.nn.Module, ClipTokenWeightEncoder):
    LAYERS = [
        "last",
        "pooled",
        "hidden"
    ]

    def __init__(self, version="openai/clip-vit-large-patch14", device="cpu", max_length=77, freeze=True, layer="last", layer_idx=None, dtype=None, special_tokens={"start": 49406, "end": 49407, "pad": 49407}, layer_norm_hidden_state=True, enable_attention_masks=False, return_projected_pooled=True):
        super().__init__()
        assert layer in self.LAYERS
        self.transformer, _ = longclip.load(version, device=device)
        self.num_layers = self.transformer.transformer_layers
        self.max_length = max_length
        if freeze:
            self.freeze()
        self.layer = layer
        self.layer_idx = None
        self.special_tokens = special_tokens
        self.text_projection = torch.nn.Parameter(torch.eye(self.transformer.get_input_embeddings().weight.shape[1]))
        self.logit_scale = torch.nn.Parameter(torch.tensor(4.6055))
        self.enable_attention_masks = enable_attention_masks
        self.layer_norm_hidden_state = layer_norm_hidden_state
        self.return_projected_pooled = return_projected_pooled
        if layer == "hidden":
            assert layer_idx is not None
            assert abs(layer_idx) < self.num_layers
            self.clip_layer(layer_idx)
        self.layer_default = (self.layer, self.layer_idx)
        self.options_default = (self.layer, self.layer_idx, self.return_projected_pooled)

    def freeze(self):
        self.transformer = self.transformer.eval()
        for param in self.parameters():
            param.requires_grad = False

    def clip_layer(self, layer_idx):
        if abs(layer_idx) > self.num_layers:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

    def reset_clip_layer(self):
        self.layer = self.layer_default[0]
        self.layer_idx = self.layer_default[1]

    def set_clip_options(self, options):
        layer_idx = options.get("layer", self.layer_idx)
        self.return_projected_pooled = options.get("projected_pooled", self.return_projected_pooled)
        if layer_idx is None or abs(layer_idx) > self.num_layers:
            self.layer = "last"
        else:
            self.layer = "hidden"
            self.layer_idx = layer_idx

    def reset_clip_options(self):
        self.layer = self.options_default[0]
        self.layer_idx = self.options_default[1]
        self.return_projected_pooled = self.options_default[2]

    def set_up_textual_embeddings(self, tokens, current_embeds):
        out_tokens = []
        next_new_token = token_dict_size = current_embeds.weight.shape[0] - 1
        embedding_weights = []

        for x in tokens:
            tokens_temp = []
            for y in x:
                if isinstance(y, int):
                    if y == token_dict_size:
                        y = -1
                    tokens_temp += [y]
                else:
                    if y.shape[0] == current_embeds.weight.shape[1]:
                        embedding_weights += [y]
                        tokens_temp += [next_new_token]
                        next_new_token += 1
                    else:
                        print("WARNING: shape mismatch when trying to apply embedding, embedding will be ignored", y.shape[0], current_embeds.weight.shape[1])
            while len(tokens_temp) < len(x):
                tokens_temp += [self.special_tokens["pad"]]
            out_tokens += [tokens_temp]

        n = token_dict_size
        if len(embedding_weights) > 0:
            new_embedding = torch.nn.Embedding(next_new_token + 1, current_embeds.weight.shape[1], device=current_embeds.weight.device, dtype=current_embeds.weight.dtype)
            new_embedding.weight[:token_dict_size] = current_embeds.weight[:-1]
            for x in embedding_weights:
                new_embedding.weight[n] = x
                n += 1
            new_embedding.weight[n] = current_embeds.weight[-1]
            self.transformer.set_input_embeddings(new_embedding)

        processed_tokens = []
        for x in out_tokens:
            processed_tokens += [list(map(lambda a: n if a == -1 else a, x))]
        return processed_tokens

    def forward(self, tokens):
        backup_embeds = self.transformer.get_input_embeddings()
        device = backup_embeds.weight.device
        tokens = self.set_up_textual_embeddings(tokens, backup_embeds)
        tokens = torch.LongTensor(tokens).to(device)

        attention_mask = None
        if self.enable_attention_masks:
            attention_mask = torch.zeros_like(tokens)
            max_token = self.transformer.get_input_embeddings().weight.shape[0] - 1
            for x in range(attention_mask.shape[0]):
                for y in range(attention_mask.shape[1]):
                    attention_mask[x, y] = 1
                    if tokens[x, y] == max_token:
                        break

        outputs = self.transformer(tokens, attention_mask, intermediate_output=self.layer_idx, final_layer_norm_intermediate=self.layer_norm_hidden_state)
        self.transformer.set_input_embeddings(backup_embeds)

        if self.layer == "last":
            z = outputs[0]
        else:
            z = outputs[1]

        pooled_output = None
        if len(outputs) >= 3:
            if not self.return_projected_pooled and len(outputs) >= 4 and outputs[3] is not None:
                pooled_output = outputs[3].float()
            elif outputs[2] is not None:
                pooled_output = outputs[2].float()

        return z.float(), pooled_output

    def encode(self, tokens):
        return self(tokens)

    def load_sd(self, sd):
        if "text_projection" in sd:
            self.text_projection[:] = sd.pop("text_projection")
        if "text_projection.weight" in sd:
            self.text_projection[:] = sd.pop("text_projection.weight").transpose(0, 1)
        return self.transformer.load_state_dict(sd, strict=False)

class SDLongTokenizer:
    def __init__(self, max_length=248, pad_with_end=True, embedding_directory=None, embedding_size=768, embedding_key='clip_l',  has_start_token=True, pad_to_max_length=True):
        self.tokenizer = longclip.only_tokenize
        self.max_length = max_length
        empty = self.tokenizer('')[0]
        if has_start_token:
            self.tokens_start = 1
            self.start_token = empty[0]
            self.end_token = empty[1]
        else:
            self.tokens_start = 0
            self.start_token = None
            self.end_token = empty[0]
        self.pad_with_end = pad_with_end
        self.pad_to_max_length = pad_to_max_length
        self.embedding_directory = embedding_directory
        self.max_word_length = 8
        self.embedding_identifier = "embedding:"
        self.embedding_size = embedding_size
        self.embedding_key = embedding_key

    def _try_get_embedding(self, embedding_name:str):
        embed = load_embed(embedding_name, self.embedding_directory, self.embedding_size, self.embedding_key)
        if embed is None:
            stripped = embedding_name.strip(',')
            if len(stripped) < len(embedding_name):
                embed = load_embed(stripped, self.embedding_directory, self.embedding_size, self.embedding_key)
                return (embed, embedding_name[len(stripped):])
        return (embed, "")

    def tokenize_with_weights(self, text:str, return_word_ids=False):
        if self.pad_with_end:
            pad_token = self.end_token
        else:
            pad_token = 0

        text = escape_important(text)
        parsed_weights = token_weights(text, 1.0)

        tokens = []
        for weighted_segment, weight in parsed_weights:
            to_tokenize = unescape_important(weighted_segment).replace("\n", " ").split(' ')
            to_tokenize = [x for x in to_tokenize if x != ""]
            for word in to_tokenize:
                if word.startswith(self.embedding_identifier) and self.embedding_directory is not None:
                    embedding_name = word[len(self.embedding_identifier):].strip('\n')
                    embed, leftover = self._try_get_embedding(embedding_name)
                    if embed is None:
                        print(f"warning, embedding:{embedding_name} does not exist, ignoring")
                    else:
                        if len(embed.shape) == 1:
                            tokens.append([(embed, weight)])
                        else:
                            tokens.append([(embed[x], weight) for x in range(embed.shape[0])])
                    if leftover != "":
                        word = leftover
                    else:
                        continue
                tokens.append([(t, weight) for t in self.tokenizer(word)[0][self.tokens_start:-1]])

        batched_tokens = []
        batch = []
        if self.start_token is not None:
            batch.append((self.start_token, 1.0, 0))
        batched_tokens.append(batch)
        for i, t_group in enumerate(tokens):
            is_large = len(t_group) >= self.max_word_length

            while len(t_group) > 0:
                if len(t_group) + len(batch) > self.max_length - 1:
                    remaining_length = self.max_length - len(batch) - 1
                    if is_large:
                        batch.extend([(t,w,i+1) for t,w in t_group[:remaining_length]])
                        batch.append((self.end_token, 1.0, 0))
                        t_group = t_group[remaining_length:]
                    else:
                        batch.append((self.end_token, 1.0, 0))
                        if self.pad_to_max_length:
                            batch.extend([(pad_token, 1.0, 0)] * (remaining_length))
                    batch = []
                    if self.start_token is not None:
                        batch.append((self.start_token, 1.0, 0))
                    batched_tokens.append(batch)
                else:
                    batch.extend([(t,w,i+1) for t,w in t_group])
                    t_group = []

        batch.append((self.end_token, 1.0, 0))
        if self.pad_to_max_length:
            batch.extend([(pad_token, 1.0, 0)] * (self.max_length - len(batch)))

        if not return_word_ids:
            batched_tokens = [[(t, w) for t, w,_ in x] for x in batched_tokens]
        return batched_tokens

    def untokenize(self, token_weight_pair):
        return list(map(lambda a: (a, self.inv_vocab[a[0]]), token_weight_pair))

def pad_tokens(tokens,clip,add_token_num):
    if clip.pad_with_end:
        pad_token = clip.end_token
    else:
        pad_token = 0
    while add_token_num > 0:
        batch = []
        batch.append((clip.end_token, 1.0, 0))
        add_pad = clip.max_length - 1
        batch.extend([(pad_token, 1.0, 0)] * add_pad)
        tokens.append(batch)
        add_token_num -= (add_pad+1)
    return tokens

def token_num(tokens):
    n = 0
    for token in tokens:
        n += len(token)
    return n

class SDXLLongClipModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.clip_l = None
        self.clip_g = None

    def set_clip_options(self, options):
        self.clip_l.set_clip_options(options)
        self.clip_g.set_clip_options(options)

    def reset_clip_options(self):
        self.clip_g.reset_clip_options()
        self.clip_l.reset_clip_options()

    def encode_token_weights(self, token_weight_pairs):
        token_weight_pairs_g = token_weight_pairs["g"]
        token_weight_pairs_l = token_weight_pairs["l"]
        g_out, g_pooled = self.clip_g.encode_token_weights(token_weight_pairs_g)
        l_out, l_pooled = self.clip_l.encode_token_weights(token_weight_pairs_l)
        g_tokens = g_out.shape[1]
        l_tokens = l_out.shape[1]
        min_tokens = min(g_tokens,l_tokens)
        g_out = g_out[:,:min_tokens,:]
        l_out = l_out[:,:min_tokens,:]
        return torch.cat([l_out, g_out], dim=-1), g_pooled

    def load_sd(self, sd):
        if "text_model.encoder.layers.30.mlp.fc1.weight" in sd:
            return self.clip_g.load_sd(sd)
        else:
            return self.clip_l.load_sd(sd)

class SDXLLongTokenizer:
    def __init__(self):
        self.clip_l = None
        self.clip_g = None

    def tokenize_with_weights(self, text:str, return_word_ids=False):
        out = {}
        out["g"] = self.clip_g.tokenize_with_weights(text, return_word_ids)
        out["l"] = self.clip_l.tokenize_with_weights(text, return_word_ids)
        g_tokens = token_num(out["g"])
        l_tokens = token_num(out["l"])
        if g_tokens > l_tokens:
            out["l"] = pad_tokens(out["l"],self.clip_l,g_tokens-l_tokens)
        elif l_tokens > g_tokens:
            out["g"] = pad_tokens(out["g"],self.clip_g,l_tokens-g_tokens)
        return out

    def untokenize(self, token_weight_pair):
        return self.clip_g.untokenize(token_weight_pair)

class LONGCLIP:
    def __init__(self, target=None, embedding_directory=None, no_init=False):
        if no_init:
            return
        params = target.params.copy()
        clip = target.clip
        tokenizer = target.tokenizer

        load_device = model_management.text_encoder_device()
        offload_device = model_management.text_encoder_offload_device()
        params['device'] = offload_device
        params['dtype'] = model_management.text_encoder_dtype(load_device)

        self.cond_stage_model = clip(**(params))

        self.tokenizer = tokenizer(embedding_directory=embedding_directory)
        self.patcher = comfy.model_patcher.ModelPatcher(self.cond_stage_model, load_device=load_device, offload_device=offload_device)
        self.layer_idx = None

    def clone(self):
        n = LONGCLIP(no_init=True)
        n.patcher = self.patcher.clone()
        n.cond_stage_model = self.cond_stage_model
        n.tokenizer = self.tokenizer
        n.layer_idx = self.layer_idx
        return n

    def add_patches(self, patches, strength_patch=1.0, strength_model=1.0):
        return self.patcher.add_patches(patches, strength_patch, strength_model)

    def clip_layer(self, layer_idx):
        self.layer_idx = layer_idx

    def tokenize(self, text, return_word_ids=False):
        return self.tokenizer.tokenize_with_weights(text, return_word_ids)

    def encode_from_tokens(self, tokens, return_pooled=False):
        self.cond_stage_model.reset_clip_options()

        if self.layer_idx is not None:
            self.cond_stage_model.set_clip_options({"layer": self.layer_idx})

        if return_pooled == "unprojected":
            self.cond_stage_model.set_clip_options({"projected_pooled": False})

        self.load_model()
        cond, pooled = self.cond_stage_model.encode_token_weights(tokens)
        if return_pooled:
            return cond, pooled
        return cond

    def encode(self, text):
        tokens = self.tokenize(text)
        return self.encode_from_tokens(tokens)

    def load_sd(self, sd, full_model=False):
        if full_model:
            return self.cond_stage_model.load_state_dict(sd, strict=False)
        else:
            return self.cond_stage_model.load_sd(sd)

    def get_sd(self):
        return self.cond_stage_model.state_dict()

    def load_model(self):
        model_management.load_model_gpu(self.patcher)
        return self.patcher

    def get_key_patches(self):
        return self.patcher.get_key_patches()

def HunyuanClipping(self, text, text_t5, CLIP, T5):
    # T5
    T5.load_model()
    t5_pre = T5.tokenizer(
        text_t5,
        max_length            = T5.cond_stage_model.max_length,
        padding               = 'max_length',
        truncation            = True,
        return_attention_mask = True,
        add_special_tokens    = True,
        return_tensors        = 'pt'
    )
    t5_mask = t5_pre["attention_mask"]
    with torch.no_grad():
        t5_outs = T5.cond_stage_model.transformer(
            input_ids = t5_pre["input_ids"].to(T5.load_device),
            attention_mask = t5_mask.to(T5.load_device),
            output_hidden_states = True,
        )
        # to-do: replace -1 for clip skip
        t5_embs = t5_outs["hidden_states"][-1].float().cpu()

    # "clip"
    CLIP.load_model()
    clip_pre = CLIP.tokenizer(
        text,
        max_length            = CLIP.cond_stage_model.max_length,
        padding               = 'max_length',
        truncation            = True,
        return_attention_mask = True,
        add_special_tokens    = True,
        return_tensors        = 'pt'
    )
    clip_mask = clip_pre["attention_mask"]
    with torch.no_grad():
        clip_outs = CLIP.cond_stage_model.transformer(
            input_ids = clip_pre["input_ids"].to(CLIP.load_device),
            attention_mask = clip_mask.to(CLIP.load_device),
        )
        # to-do: add hidden states
        clip_embs = clip_outs[0].float().cpu()

    # combined cond
    return ([[
        clip_embs, {
            "context_t5": t5_embs,
            "context_mask": clip_mask.float(),
            "context_t5_mask": t5_mask.float()
        }
    ]],)


def apply_weight(text, strength):
    if text is None or text.strip(' ,;') == '':
        return ''
    return f'({text}:{strength:.2f})'


def apply_weight_optional(text, strength):
    if text is None or text.strip(' ,;') == '':
        return ''
    result = f'({text}:{strength:.2f})' if strength != 1 else str(text)
    return result.replace(":1.00", "")


def inject_keyword(text, keyword_list):
    if keyword_list is None:
        return text
    items = list(filter(None, keyword_list))
    if len(items) != 2:
        return text
    keyword, placement = items
    return keyword + ', ' + text if placement == 'First' else text + ', ' + keyword


def build_prompt_context(
    model_concept, positive_prompt, negative_prompt,
    enhanced_prompt, enhanced_prompt_usage, enhanced_prompt_strength,
    style_pos_prompt, style_neg_prompt,
    style_handling, style_swap, style_position,
    style_pos_strength, style_neg_strength,
    opt_pos_prompt, opt_neg_prompt, opt_pos_strength, opt_neg_strength,
    negative_strength,
    int_style_pos, int_style_neg, int_style_pos_strength, int_style_neg_strength,
    use_int_style, default_pos, default_neg,
    l_strength, positive_l, negative_l,
    model_keywords, lora_keywords, lycoris_keywords,
    embedding_pos, embedding_neg,
):
    copy_prompt_to_l = True
    t5xxl_prompt = ""

    if len(enhanced_prompt) > 5:
        match enhanced_prompt_usage:
            case 'Add':
                if enhanced_prompt_strength != 1:
                    enhanced_prompt = f'({enhanced_prompt}:{enhanced_prompt_strength:.2f})'
                if enhanced_prompt_strength != 0:
                    positive_prompt = positive_prompt + ', ' + enhanced_prompt
            case 'Replace':
                positive_prompt = enhanced_prompt
            case 'T5-XXL':
                t5xxl_prompt = enhanced_prompt
    else:
        if len(style_pos_prompt) > 5 and style_handling == True:
            if style_swap == True:
                positive_prompt, style_pos_prompt = style_pos_prompt, positive_prompt
            t5xxl_prompt = style_pos_prompt
            style_pos_prompt = None
            positive_l = style_pos_prompt
            copy_prompt_to_l = False

    additional_positive = None
    additional_negative = None
    if use_int_style:
        if int_style_pos != 'None':
            additional_positive = default_pos[int_style_pos]['positive'].strip(' ,;')
        if int_style_neg != 'None':
            additional_negative = default_neg[int_style_neg]['negative'].strip(' ,;')

    additional_positive = apply_weight(additional_positive, int_style_pos_strength) if additional_positive else ''
    additional_negative = apply_weight(additional_negative, int_style_neg_strength) if additional_negative else ''
    negative_prompt = apply_weight(negative_prompt, negative_strength)
    opt_pos_prompt = apply_weight(opt_pos_prompt, opt_pos_strength)
    opt_neg_prompt = apply_weight(opt_neg_prompt, opt_neg_strength)
    style_pos_prompt = apply_weight_optional(style_pos_prompt, style_pos_strength)
    style_neg_prompt = apply_weight(style_neg_prompt, style_neg_strength)

    if style_pos_prompt or style_neg_prompt or model_concept != "Normal":
        copy_prompt_to_l = False

    if copy_prompt_to_l:
        positive_l = positive_prompt
        negative_l = negative_prompt

    positive_l = apply_weight_optional(positive_l, l_strength)
    negative_l = apply_weight_optional(negative_l, l_strength)

    if style_pos_prompt.startswith('((') and style_pos_prompt.endswith('))'):
        style_pos_prompt = '(' + style_pos_prompt.strip('()') + ')'
    if style_neg_prompt.startswith('((') and style_neg_prompt.endswith('))'):
        style_neg_prompt = '(' + style_neg_prompt.strip('()') + ')'

    _clean = lambda s: s.strip(' ,;').replace(", , ", ", ").replace(", , ", ", ").replace(":1.00", "")
    if not style_position:
        positive_text = _clean(f'{positive_prompt}, {opt_pos_prompt}, {style_pos_prompt}, {additional_positive}')
        negative_text = _clean(f'{negative_prompt}, {opt_neg_prompt}, {style_neg_prompt}, {additional_negative}')
    else:
        positive_text = _clean(f'{style_pos_prompt}, {opt_pos_prompt}, {positive_prompt}, {additional_positive}')
        negative_text = _clean(f'{style_neg_prompt}, {opt_neg_prompt}, {negative_prompt}, {additional_negative}')

    positive_text = inject_keyword(positive_text, model_keywords)
    positive_text = inject_keyword(positive_text, lora_keywords)
    positive_text = inject_keyword(positive_text, lycoris_keywords)
    positive_text = inject_keyword(positive_text, embedding_pos)
    negative_text = inject_keyword(negative_text, embedding_neg)

    return positive_text, negative_text, t5xxl_prompt, positive_l, negative_l


SDXL_CONCEPTS = {'SDXL', 'Illustrious', 'Pony', 'Playground'}

def apply_clip_overrides(loader_self, clip, workflow_tuple):
    if not workflow_tuple:
        return clip
    encoder_1 = workflow_tuple.get('encoder_1', None)
    last_layer = int(workflow_tuple.get('last_layer', 0))
    baked_clip = clip

    if encoder_1 and encoder_1 != 'None' and not workflow_tuple.get('clip_selection', False):
        try:
            model_concept = workflow_tuple.get('model_concept', 'SD1')
            is_longclip = 'longclip' in encoder_1.lower() or encoder_1.lower().endswith('.pt')
            if is_longclip:
                if model_concept in SDXL_CONCEPTS:
                    clip = long_clip_module.SDXLLongClip.sdxl_longclip(loader_self, encoder_1, baked_clip)[0]
                else:
                    clip = long_clip_module.SDLongClip.sd_longclip(loader_self, encoder_1)[0]
            else:
                clip = nodes.CLIPLoader.load_clip(loader_self, encoder_1, 'stable_diffusion')[0]
        except Exception:
            if baked_clip is None:
                raise RuntimeError(f"Clip model '{encoder_1}' is incompatible with this checkpoint and no baked CLIP is available.")
            clip = baked_clip

    if last_layer < 0:
        clip = nodes.CLIPSetLastLayer.set_last_layer(loader_self, clip, last_layer)[0]

    return clip


def encode_standard(clip, positive_text, negative_text, t5xxl_prompt, adv_encode, token_normalization, weight_interpretation, positive_l, negative_l, width, height, workflow_tuple, advanced_encode_fn):
    if adv_encode:
        tokens_p = clip.tokenize(positive_text)
        tokens_n = clip.tokenize(negative_text)
        if 'l' not in tokens_p or 'g' not in tokens_p or 'l' not in tokens_n or 'g' not in tokens_n:
            embeddings_final_pos, pooled_pos = advanced_encode_fn(clip, positive_text, token_normalization, weight_interpretation, w_max=1.0, apply_to_pooled=True)
            embeddings_final_neg, pooled_neg = advanced_encode_fn(clip, negative_text, token_normalization, weight_interpretation, w_max=1.0, apply_to_pooled=True)
            return ([[embeddings_final_pos, {"pooled_output": pooled_pos}]], [[embeddings_final_neg, {"pooled_output": pooled_neg}]], positive_text, negative_text, t5xxl_prompt, "", "", workflow_tuple)
        else:
            if 'l' in clip.tokenize(positive_l):
                tokens_p["l"] = clip.tokenize(positive_l)["l"]
                if len(tokens_p["l"]) != len(tokens_p["g"]):
                    empty = clip.tokenize("")
                    while len(tokens_p["l"]) < len(tokens_p["g"]):
                        tokens_p["l"] += empty["l"]
                    while len(tokens_p["l"]) > len(tokens_p["g"]):
                        tokens_p["g"] += empty["g"]
            if 'l' in clip.tokenize(negative_l):
                tokens_n["l"] = clip.tokenize(negative_l)["l"]
                if len(tokens_n["l"]) != len(tokens_n["g"]):
                    empty = clip.tokenize("")
                    while len(tokens_n["l"]) < len(tokens_n["g"]):
                        tokens_n["l"] += empty["l"]
                    while len(tokens_n["l"]) > len(tokens_n["g"]):
                        tokens_n["g"] += empty["g"]
            cond_p, pooled_p = clip.encode_from_tokens(tokens_p, return_pooled=True)
            cond_n, pooled_n = clip.encode_from_tokens(tokens_n, return_pooled=True)
            return ([[cond_p, {"pooled_output": pooled_p, "width": width, "height": height, "crop_w": 0, "crop_h": 0, "target_width": width, "target_height": height}]], [[cond_n, {"pooled_output": pooled_n, "width": width, "height": height, "crop_w": 0, "crop_h": 0, "target_width": width, "target_height": height}]], positive_text, negative_text, "", positive_l, negative_l, workflow_tuple)
    else:
        tokens_pos = clip.tokenize(positive_text)
        tokens_neg = clip.tokenize(negative_text)
        try:
            comfy.model_management.soft_empty_cache()
        except Exception:
            pass
        out_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True, return_dict=True)
        out_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True, return_dict=True)
        cond_pos = out_pos.pop("cond")
        cond_neg = out_neg.pop("cond")
        return ([[cond_pos, out_pos]], [[cond_neg, out_neg]], positive_text, negative_text, t5xxl_prompt, "", "", workflow_tuple)


def encode_sd3(clip, positive_text, negative_text, t5xxl_prompt, workflow_tuple):
    if t5xxl_prompt:
        pos_out = nodes_sd3.CLIPTextEncodeSD3.execute(clip, positive_text, positive_text, t5xxl_prompt, 'none')
        tokens_neg = clip.tokenize(negative_text)
        out_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True, return_dict=True)
        cond_neg = out_neg.pop("cond")
        return (pos_out[0], [[cond_neg, out_neg]], positive_text, negative_text, t5xxl_prompt, "", "", workflow_tuple)
    else:
        tokens_pos = clip.tokenize(positive_text)
        tokens_neg = clip.tokenize(negative_text)
        out_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True, return_dict=True)
        out_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True, return_dict=True)
        cond_pos = out_pos.pop("cond")
        cond_neg = out_neg.pop("cond")
        return ([[cond_pos, out_pos]], [[cond_neg, out_neg]], positive_text, negative_text, "", "", "", workflow_tuple)


def encode_stable_cascade(clip, positive_text, negative_text, workflow_tuple):
    positive_text = utility.DiT_cleaner(positive_text)
    negative_text = utility.DiT_cleaner(negative_text)
    tokens_pos = clip.tokenize(positive_text)
    tokens_neg = clip.tokenize(negative_text)
    cond_pos, pooled_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True)
    cond_neg, pooled_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True)
    return ([[cond_pos, {"pooled_output": pooled_pos}]], [[cond_neg, {"pooled_output": pooled_neg}]], positive_text, negative_text, "", "", "", workflow_tuple)


def encode_pixart_sigma(clip, positive_text, negative_text, workflow_tuple):
    positive_text = utility.DiT_cleaner(positive_text)
    negative_text = utility.DiT_cleaner(negative_text)

    cond_pos_ref = cond_neg_ref = out_pos_ref = out_neg_ref = None

    if clip['refiner'] is not None:
        clipRef = clip['refiner']
        tokens_pos_ref = clipRef.tokenize(positive_text)
        tokens_neg_ref = clipRef.tokenize(negative_text)
        out_pos_ref = clipRef.encode_from_tokens(tokens_pos_ref, return_pooled=True, return_dict=True)
        out_neg_ref = clipRef.encode_from_tokens(tokens_neg_ref, return_pooled=True, return_dict=True)
        cond_pos_ref = out_pos_ref.pop("cond")
        cond_neg_ref = out_neg_ref.pop("cond")

    clipMain = clip['main']
    tokens_pos_main = clipMain.tokenize(positive_text)
    tokens_neg_main = clipMain.tokenize(negative_text)
    out_pos_main = clipMain.encode_from_tokens(tokens_pos_main, return_pooled=True, return_dict=True)
    out_neg_main = clipMain.encode_from_tokens(tokens_neg_main, return_pooled=True, return_dict=True)
    cond_pos_main = out_pos_main.pop("cond")
    cond_neg_main = out_neg_main.pop("cond")

    return ({'refiner': [[cond_pos_ref, out_pos_ref]], 'main': [[cond_pos_main, out_pos_main]]}, {'refiner': [[cond_neg_ref, out_neg_ref]], 'main': [[cond_neg_main, out_neg_main]]}, positive_text, negative_text, "", "", "", workflow_tuple)


def encode_chroma(clip, positive_text, negative_text, workflow_tuple):
    tokens_pos = clip.tokenize(positive_text)
    tokens_neg = clip.tokenize(negative_text)
    out_pos = clip.encode_from_tokens(tokens_pos, return_pooled=True, return_dict=True)
    out_neg = clip.encode_from_tokens(tokens_neg, return_pooled=True, return_dict=True)
    cond_pos = out_pos.pop("cond")
    cond_neg = out_neg.pop("cond")
    return ([[cond_pos, out_pos]], [[cond_neg, out_neg]], positive_text, negative_text, "", "", "", workflow_tuple)


def encode_flux(clip, positive_text, negative_text, t5xxl_prompt, concept_data, workflow_tuple):
    positive_text = utility.DiT_cleaner(positive_text)
    negative_text = utility.DiT_cleaner(negative_text)

    flux_sampler = concept_data.get('sampler', 'ksampler') if concept_data else 'ksampler'
    flux_guidance = float(concept_data.get('guidance', 2.0)) if concept_data else 2.0
    if flux_sampler == 'ksampler':
        cond_pos = nodes_flux.CLIPTextEncodeFlux.execute(clip, positive_text, t5xxl_prompt, flux_guidance)[0]
        if concept_data is not None and float(concept_data.get('cfg', 2.0)) < 1.2:
            cond_neg = cond_pos
        else:
            cond_neg = nodes_flux.CLIPTextEncodeFlux.execute(clip, negative_text, negative_text, flux_guidance)[0]
        return (cond_pos, cond_neg, positive_text, negative_text, t5xxl_prompt, "", "", workflow_tuple)
    else:
        cond_pos = nodes_flux.CLIPTextEncodeFlux.execute(clip, positive_text, t5xxl_prompt, flux_guidance)[0]
        return (cond_pos, cond_pos, positive_text, negative_text, t5xxl_prompt, "", "", workflow_tuple)


_SANA_MAX_TOKENS = 300
_SANA_CHI_PROMPT = "\n".join([
    'Create one detailed perfect prompt from given User Prompt for stable diffusion text-to-image text2image modern DiT models.',
    'Generate only the one enhanced description for the prompt below, avoid including any additional questions comments or evaluations.',
    'User Prompt: ',
])


def _sana_encode_text(tokenizer, text_encoder, text, device):
    full_prompt = _SANA_CHI_PROMPT + text
    num_chi_tokens = len(tokenizer.encode(_SANA_CHI_PROMPT))
    max_length = num_chi_tokens + _SANA_MAX_TOKENS - 2
    tokens = tokenizer([full_prompt], max_length=max_length, padding="max_length", truncation=True, return_tensors="pt").to(device)
    select_idx = [0] + list(range(-_SANA_MAX_TOKENS + 1, 0))
    embs = text_encoder(tokens.input_ids, tokens.attention_mask)[0][:, None][:, :, select_idx]
    masks = tokens.attention_mask[:, select_idx]
    return embs * masks.unsqueeze(-1)


def encode_sana(clip, positive_text, negative_text, t5xxl_prompt, workflow_tuple):
    positive_text = utility.DiT_cleaner(positive_text)
    negative_text = utility.DiT_cleaner(negative_text)
    t5xxl_prompt = utility.DiT_cleaner(t5xxl_prompt)

    scheduler_name = workflow_tuple.get('scheduler_name', 'flow_dpm-solver') if workflow_tuple else 'flow_dpm-solver'
    device = model_management.get_torch_device()

    if scheduler_name == 'flow_dpm-solver' and hasattr(clip, 'text_encoder'):
        clip.text_encoder.to(device)
        null_token = clip.tokenizer(negative_text, max_length=_SANA_MAX_TOKENS, padding="max_length", truncation=True, return_tensors="pt").to(device)
        null_embs = clip.text_encoder(null_token.input_ids, null_token.attention_mask)[0]
        with torch.no_grad():
            prompts = [prepare_prompt_ar(positive_text, ASPECT_RATIO_1024_TEST, device=device, show=False)[0].strip()]
            num_chi_tokens = len(clip.tokenizer.encode(_SANA_CHI_PROMPT))
            max_length_all = num_chi_tokens + _SANA_MAX_TOKENS - 2
            caption_token = clip.tokenizer([_SANA_CHI_PROMPT + positive_text], max_length=max_length_all, padding="max_length", truncation=True, return_tensors="pt").to(device)
            select_index = [0] + list(range(-_SANA_MAX_TOKENS + 1, 0))
            caption_embs = clip.text_encoder(caption_token.input_ids, caption_token.attention_mask)[0][:, None][:, :, select_index]
            emb_masks = caption_token.attention_mask[:, select_index]
            null_y = null_embs.repeat(len(prompts), 1, 1)[:, None]
        clip.text_encoder.to(model_management.text_encoder_offload_device())
        comfy.model_management.soft_empty_cache(True)
        return ([[caption_embs, {"emb_masks": emb_masks}]], [[null_y, {}]], positive_text, negative_text, t5xxl_prompt, "", "", workflow_tuple)
    else:
        tokenizer = clip["tokenizer"]
        text_encoder = clip["text_encoder"]
        enc_device = text_encoder.device
        with torch.no_grad():
            sana_embs_pos = _sana_encode_text(tokenizer, text_encoder, positive_text, enc_device)
            sana_embs_neg = _sana_encode_text(tokenizer, text_encoder, negative_text, enc_device)
        return ([[sana_embs_pos, {}]], [[sana_embs_neg, {}]], positive_text, negative_text, t5xxl_prompt, "", "", workflow_tuple)


def encode_qwen_edit(loader_self, clip, positive_text, negative_text, t5xxl_prompt, edit_vae, edit_image_list, workflow_tuple):
    if type(edit_image_list).__name__ == "Tensor":
        edit_image_list = [edit_image_list]
    positive_text = utility.DiT_cleaner(positive_text)
    negative_text = utility.DiT_cleaner(negative_text)
    t5xxl_prompt = utility.DiT_cleaner(t5xxl_prompt)

    conditioning = utility.edit_encoder(clip, positive_text, edit_vae, edit_image_list)
    tokens_neg = clip.tokenize(negative_text, images=[])
    conditioning_neg = clip.encode_from_tokens_scheduled(tokens_neg)
    return (conditioning, conditioning_neg, positive_text, negative_text, t5xxl_prompt, "", "", workflow_tuple)


def encode_kolors(clip, positive_text, negative_text, t5xxl_prompt, workflow_tuple):
    positive_text = utility.DiT_cleaner(positive_text)
    negative_text = utility.DiT_cleaner(negative_text)
    t5xxl_prompt = utility.DiT_cleaner(t5xxl_prompt)

    device = model_management.text_encoder_device()
    try:
        model_management.unload_all_models()
        model_management.soft_empty_cache()
    except Exception:
        pass

    tokenizer = clip['tokenizer']
    text_encoder = clip['text_encoder']
    model_management.soft_empty_cache()

    prompt_embeds_dtype = text_encoder.dtype if text_encoder is not None else torch.float16
    try:
        text_encoder.to(dtype=prompt_embeds_dtype, device=device)
    except Exception:
        pass

    text_inputs = tokenizer(positive_text, padding="max_length", max_length=256, truncation=True, return_tensors="pt").to(device)
    output = text_encoder(input_ids=text_inputs['input_ids'], attention_mask=text_inputs['attention_mask'], position_ids=text_inputs['position_ids'], output_hidden_states=True)
    prompt_embeds = output.hidden_states[-2].permute(1, 0, 2).clone()
    text_proj = output.hidden_states[-1][-1, :, :].clone()
    bs_embed, seq_len, _ = prompt_embeds.shape
    prompt_embeds = prompt_embeds.view(bs_embed, seq_len, -1)

    uncond_input = tokenizer([negative_text], padding="max_length", max_length=prompt_embeds.shape[1], truncation=True, return_tensors="pt").to(device)
    output = text_encoder(input_ids=uncond_input['input_ids'], attention_mask=uncond_input['attention_mask'], position_ids=uncond_input['position_ids'], output_hidden_states=True)
    negative_prompt_embeds = output.hidden_states[-2].permute(1, 0, 2).clone()
    negative_text_proj = output.hidden_states[-1][-1, :, :].clone()
    negative_prompt_embeds = negative_prompt_embeds.to(dtype=text_encoder.dtype, device=device).view(1, negative_prompt_embeds.shape[1], -1)

    text_proj = text_proj.view(text_proj.shape[0], -1)
    negative_text_proj = negative_text_proj.view(negative_text_proj.shape[0], -1)

    try:
        model_management.soft_empty_cache()
    except Exception:
        pass
    gc.collect()

    kolors_embeds = {
        'prompt_embeds': prompt_embeds.half(),
        'negative_prompt_embeds': negative_prompt_embeds.half(),
        'pooled_prompt_embeds': text_proj.half(),
        'negative_pooled_prompt_embeds': negative_text_proj.half(),
    }
    return (kolors_embeds, None, positive_text, negative_text, t5xxl_prompt, "", "", workflow_tuple)


def encode_hunyuan(loader_self, clip, positive_text, negative_text, t5xxl_prompt, workflow_tuple):
    if clip['t5'] is not None:
        positive_text = utility.DiT_cleaner(positive_text)
        negative_text = utility.DiT_cleaner(negative_text)
        t5xxl_prompt = utility.DiT_cleaner(t5xxl_prompt)
        pos_out = HunyuanClipping(loader_self, positive_text, t5xxl_prompt, clip['clip'], clip['t5'])
        neg_out = HunyuanClipping(loader_self, negative_text, "", clip['clip'], clip['t5'])
        return (pos_out[0], neg_out[0], positive_text, negative_text, t5xxl_prompt, "", "", workflow_tuple)
    else:
        clip_model = clip['clip']
        positive_text = utility.DiT_cleaner(positive_text, 512)
        negative_text = utility.DiT_cleaner(negative_text, 512)
        out_pos = clip_model.encode_from_tokens(clip_model.tokenize(positive_text), return_pooled=True, return_dict=True)
        out_neg = clip_model.encode_from_tokens(clip_model.tokenize(negative_text), return_pooled=True, return_dict=True)
        cond_pos = out_pos.pop("cond")
        cond_neg = out_neg.pop("cond")
        return ([[cond_pos, out_pos]], [[cond_neg, out_neg]], positive_text, negative_text, t5xxl_prompt, "", "", workflow_tuple)