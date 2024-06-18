import json
from ..exif.base_format import BaseFormat
from ....components import utility

# comfyui node types
KSAMPLER_TYPES = ["KSampler", "KSamplerAdvanced", "PrimereKSampler"]
VAE_ENCODE_TYPE = ["VAEEncode", "VAEEncodeForInpaint"]
CHECKPOINT_LOADER_TYPE = [
    "CheckpointLoader",
    "CheckpointLoaderSimple",
    "unCLIPCheckpointLoader",
    "Checkpoint Loader (Simple)",
    "PrimereCKPTLoader",
    "PrimereVisualCKPT",
    "PrimereCKPT"
]
CLIP_TEXT_ENCODE_TYPE = [
    "CLIPTextEncode",
    "CLIPTextEncodeSDXL",
    "CLIPTextEncodeSDXLRefiner",
    "PrimereCLIPEncoder"
]
SAVE_IMAGE_TYPE = ["SaveImage", "Image Save", "PrimerePreviewImage"]

class ComfyUI(BaseFormat):
    def __init__(self, info: dict = None, raw: str = ""):
        super().__init__(info, raw)
        self._comfy_png()

    def _comfy_png(self):
        prompt = self._info.get("prompt") or {}
        gendata = self._info.get("gendata") or {}
        gendata_json = json.loads(gendata)
        workflow = self._info.get("workflow") or {}
        prompt_json = json.loads(prompt)

        if len(gendata_json) > 0:
            FINAL_DICT = gendata_json
            self._parameter = FINAL_DICT
        else:
            # find end node of each flow
            end_nodes = list(filter( lambda item: item[-1].get("class_type") in ["SaveImage"] + KSAMPLER_TYPES, prompt_json.items(),))
            longest_flow = {}
            longest_nodes = []
            longest_flow_len = 0

            for end_node in end_nodes:
                flow, nodes = self._comfy_traverse(prompt_json, str(end_node[0]))
                if len(nodes) > longest_flow_len:
                    longest_flow = flow
                    longest_nodes = nodes
                    longest_flow_len = len(nodes)

            SizeID = None
            ModelID = None
            PositiveID = None
            NegativeID = None

            try:
                if 'latent_image' in flow:
                    SizeID = flow['latent_image'][0]
            except Exception:
                SizeID = None

            try:
                if 'model' in flow:
                    ModelID = flow['model'][0]
            except Exception:
                ModelID = None

            try:
                if 'positive' in flow:
                    PositiveID = flow['positive'][0]
            except Exception:
                PositiveID = None

            try:
                if 'negative' in flow:
                    NegativeID = flow['negative'][0]
            except Exception:
                NegativeID = None

            FINAL_DICT = {}
            FINAL_DICT['negative'] = ""
            FINAL_DICT['positive'] = ""

            if PositiveID and NegativeID and 'text_g' in prompt_json[PositiveID]['inputs']:
                FINAL_DICT['positive'] = prompt_json[PositiveID]['inputs']['text_g']
                FINAL_DICT['negative'] = prompt_json[NegativeID]['inputs']['text_g']

            if PositiveID and NegativeID and 'text' in prompt_json[PositiveID]['inputs']:
                FINAL_DICT['positive'] = prompt_json[PositiveID]['inputs']['text']
                FINAL_DICT['negative'] = prompt_json[NegativeID]['inputs']['text']

            if PositiveID == None or ('text_g' not in prompt_json[PositiveID]['inputs'] and 'text' not in prompt_json[PositiveID]['inputs']):
                if hasattr(self, '_positive'):
                    FINAL_DICT['positive'] = self._positive
                if hasattr(self, '_negative'):
                    FINAL_DICT['negative'] = self._negative

            if 'steps' in flow and type(flow['steps']) == int:
                FINAL_DICT['steps'] = flow['steps']
            if 'sampler_name' in flow and 'scheduler' in flow and type(flow['sampler_name']) == str and type(flow['scheduler']) == str:
                FINAL_DICT['sampler'] = flow['sampler_name'] + ' ' + flow['scheduler']
            if 'seed' in flow and type(flow['seed']) == int:
                FINAL_DICT['seed'] = flow['seed']
            if 'cfg' in flow and (type(flow['cfg']) == int or type(flow['cfg']) == float):
                FINAL_DICT['cfg_scale'] = flow['cfg']

            if ModelID and 'ckpt_name' in prompt_json[ModelID]['inputs'] and type(prompt_json[ModelID]['inputs']['ckpt_name']) == str:
                FINAL_DICT['model_name'] = prompt_json[ModelID]['inputs']['ckpt_name'] # flow['ckpt_name']
            elif 'ckpt_name' in flow and type(flow['ckpt_name']) == str:
                FINAL_DICT['model_name'] = flow['ckpt_name']

            if SizeID and 'width' in prompt_json[SizeID]['inputs'] and 'height' in prompt_json[SizeID]['inputs'] and type(prompt_json[SizeID]['inputs']['width']) == int:
                origwidth = str(prompt_json[SizeID]['inputs']['width'])
                origheight = str(prompt_json[SizeID]['inputs']['height'])
                FINAL_DICT['width'] = int(origwidth)
                FINAL_DICT['height'] = int(origheight)
                FINAL_DICT['size_string'] = origwidth + 'x' + origheight

            self._parameter = FINAL_DICT

    def _comfy_traverse(self, prompt, end_node):
        flow = {}
        node = [end_node]
        inputs = {}
        try:
            inputs = prompt[end_node]["inputs"]
        except:
            print("node error")
            return flow, node
        match prompt[end_node]["class_type"]:
            case node_type if node_type in SAVE_IMAGE_TYPE:
                try:
                    last_flow, last_node = self._comfy_traverse(prompt, inputs["images"][0])
                    flow = utility.merge_dict(flow, last_flow)
                    node += last_node
                except:
                    print("comfyUI SaveImage error")
            case node_type if node_type in KSAMPLER_TYPES:
                try:
                    flow = inputs
                    last_flow1, last_node1 = self._comfy_traverse(prompt, inputs["model"][0])
                    last_flow2, last_node2 = self._comfy_traverse(prompt, inputs["latent_image"][0])
                    positive = self._comfy_traverse(prompt, inputs["positive"][0])
                    if isinstance(positive, str):
                        self._positive = positive
                    elif isinstance(positive, dict):
                        self._positive_sdxl.update(positive)
                    negative = self._comfy_traverse(prompt, inputs["negative"][0])
                    if isinstance(negative, str):
                        self._negative = negative
                    elif isinstance(negative, dict):
                        self._negative_sdxl.update(negative)
                    seed = None
                    # handle "CR Seed"
                    if inputs.get("seed") and isinstance(inputs.get("seed"), list):
                        seed = {"seed": self._comfy_traverse(prompt, inputs["seed"][0])}
                    elif inputs.get("noise_seed") and isinstance(inputs.get("noise_seed"), list):
                        seed = {
                            "noise_seed": self._comfy_traverse(prompt, inputs["noise_seed"][0])
                        }
                    if seed:
                        flow.update(seed)
                    flow = utility.merge_dict(flow, last_flow1)
                    flow = utility.merge_dict(flow, last_flow2)
                    node += last_node1 + last_node2
                except:
                    print("comfyUI KSampler error")
            case node_type if node_type in CLIP_TEXT_ENCODE_TYPE:
                try:
                    match node_type:
                        case "CLIPTextEncode":
                            # SDXLPromptStyler
                            if isinstance(inputs["text"], list):
                                text = int(inputs["text"][0])
                                prompt_styler = self._comfy_traverse(prompt, str(text))
                                self._positive = prompt_styler[0]
                                self._negative = prompt_styler[1]
                                return
                            elif isinstance(inputs["text"], str):
                                return inputs.get("text")
                        case "CLIPTextEncodeSDXL":
                            # SDXLPromptStyler
                            self._is_sdxl = True
                            if isinstance(inputs["text_g"], list):
                                text_g = int(inputs["text_g"][0])
                                text_l = int(inputs["text_l"][0])
                                prompt_styler_g = self._comfy_traverse(prompt, str(text_g))
                                prompt_styler_l = self._comfy_traverse(prompt, str(text_l))
                                self._positive_sdxl["Clip G"] = prompt_styler_g[0]
                                self._positive_sdxl["Clip L"] = prompt_styler_l[0]
                                self._negative_sdxl["Clip G"] = prompt_styler_g[1]
                                self._negative_sdxl["Clip L"] = prompt_styler_l[1]
                                return
                            elif isinstance(inputs["text_g"], str):
                                return {
                                    "Clip G": inputs.get("text_g"),
                                    "Clip L": inputs.get("text_l"),
                                }
                        case "CLIPTextEncodeSDXLRefiner":
                            self._is_sdxl = True
                            if isinstance(inputs["text"], list):
                                # SDXLPromptStyler
                                text = int(inputs["text"][0])
                                prompt_styler = self._comfy_traverse(prompt, str(text))
                                self._positive_sdxl["Refiner"] = prompt_styler[0]
                                self._negative_sdxl["Refiner"] = prompt_styler[1]
                                return
                            elif isinstance(inputs["text"], str):
                                return {"Refiner": inputs.get("text")}
                except:
                    print("comfyUI CLIPText error")
            case "LoraLoader":
                try:
                    flow = inputs
                    last_flow, last_node = self._comfy_traverse(prompt, inputs["model"][0])
                    flow = utility.merge_dict(flow, last_flow)
                    node += last_node
                except:
                    print("comfyUI LoraLoader error")
            case node_type if node_type in CHECKPOINT_LOADER_TYPE:
                try:
                    return inputs, node
                except:
                    print("comfyUI CheckpointLoader error")
            case node_type if node_type in VAE_ENCODE_TYPE:
                try:
                    last_flow, last_node = self._comfy_traverse(prompt, inputs["pixels"][0])
                    flow = utility.merge_dict(flow, last_flow)
                    node += last_node
                except:
                    print("comfyUI VAE error")
            case "ControlNetApplyAdvanced":
                try:
                    positive = self._comfy_traverse(prompt, inputs["positive"][0])
                    if isinstance(positive, str):
                        self._positive = positive
                    elif isinstance(positive, dict):
                        self._positive_sdxl.update(positive)
                    negative = self._comfy_traverse(prompt, inputs["negative"][0])
                    if isinstance(negative, str):
                        self._negative = negative
                    elif isinstance(negative, dict):
                        self._negative_sdxl.update(negative)

                    last_flow, last_node = self._comfy_traverse(prompt, inputs["image"][0])
                    flow = utility.merge_dict(flow, last_flow)
                    node += last_node
                except:
                    print("comfyUI ControlNetApply error")
            case "ImageScale":
                try:
                    flow = inputs
                    last_flow, last_node = self._comfy_traverse(prompt, inputs["image"][0])
                    flow = utility.merge_dict(flow, last_flow)
                    node += last_node
                except:
                    print("comfyUI ImageScale error")
            case "UpscaleModelLoader":
                try:
                    return {"upscaler": inputs["model_name"]}
                except:
                    print("comfyUI UpscaleLoader error")
            case "ImageUpscaleWithModel":
                try:
                    flow = inputs
                    last_flow, last_node = self._comfy_traverse(prompt, inputs["image"][0])
                    model = self._comfy_traverse(prompt, inputs["upscale_model"][0])
                    flow = utility.merge_dict(flow, last_flow)
                    flow = utility.merge_dict(flow, model)
                    node += last_node
                except:
                    print("comfyUI UpscaleModel error")
            case "ConditioningCombine":
                try:
                    last_flow1, last_node1 = self._comfy_traverse(prompt, inputs["conditioning_1"][0])
                    last_flow2, last_node2 = self._comfy_traverse(prompt, inputs["conditioning_2"][0])
                    flow = utility.merge_dict(flow, last_flow1)
                    flow = utility.merge_dict(flow, last_flow2)
                    node += last_node1 + last_node2
                except:
                    print("comfyUI ConditioningCombine error")
            # custom nodes
            case "SDXLPromptStyler":
                try:
                    return inputs.get("text_positive"), inputs.get("text_negative")
                except:
                    print("comfyUI SDXLPromptStyler error")
            case "CR Seed":
                try:
                    return inputs.get("seed")
                except:
                    print("comfyUI CR Seed error")
            case _:
                try:
                    last_flow = {}
                    last_node = []
                    if inputs.get("samples"):
                        last_flow, last_node = self._comfy_traverse(prompt, inputs["samples"][0])
                    elif inputs.get("image") and isinstance(inputs.get("image"), list):
                        last_flow, last_node = self._comfy_traverse(prompt, inputs["image"][0])
                    elif inputs.get("model"):
                        last_flow, last_node = self._comfy_traverse(prompt, inputs["model"][0])
                    elif inputs.get("clip"):
                        last_flow, last_node = self._comfy_traverse(prompt, inputs["clip"][0])
                    elif inputs.get("samples_from"):
                        last_flow, last_node = self._comfy_traverse(prompt, inputs["samples_from"][0])
                    elif inputs.get("conditioning"):
                        result = self._comfy_traverse(prompt, inputs["conditioning"][0])
                        if isinstance(result, str):
                            return result
                        elif isinstance(result, list):
                            last_flow, last_node = result
                    flow = utility.merge_dict(flow, last_flow)
                    node += last_node
                except:
                    print("comfyUI bridging node error")
        return flow, node