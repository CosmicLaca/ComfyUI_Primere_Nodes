from ..components.tree import TREE_SEGMENTS
from ..components.tree import PRIMERE_ROOT
import math
import os
import folder_paths
from ..utils import comfy_dir
from ..components import detectors
import comfy
from segment_anything import sam_model_registry
from ..components import utility
import torch
from urllib.parse import urlparse
from pathlib import Path
from .modules.adv_encode import advanced_encode

class PrimereImageSegments:
    RETURN_TYPES = ("IMAGE", "IMAGE", "DETECTOR", "SAM_MODEL", "SEGS", "TUPLE", "INT", "INT", "TUPLE", "CONDITIONING", "CONDITIONING")
    RETURN_NAMES = ("IMAGE", "IMAGE_SEGS", "DETECTOR", "SAM_MODEL", "SEGS", "CROP_REGIONS", "IMAGE_MAX", "IMAGE_MAX_PERCENT", "SEGMENT_SETTINGS", "COND+", "COND-")
    OUTPUT_IS_LIST = (False, True, False, False, False, False, False, False, False, False, False)
    FUNCTION = "primere_segments"
    CATEGORY = TREE_SEGMENTS

    BBOX = {}
    SEGM = {}
    GDINO = {}
    SAMS = {}

    BBOX['UBBOX_FACE_YOLOV8M'] = 'https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8m.pt?download=true'
    BBOX['UBBOX_FACE_YOLOV8N'] = 'https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n.pt?download=true'
    BBOX['UBBOX_FACE_YOLOV8N_V2'] = 'https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8n_v2.pt?download=true'
    BBOX['UBBOX_FACE_YOLOV8S'] = 'https://huggingface.co/Bingsu/adetailer/resolve/main/face_yolov8s.pt?download=true'
    BBOX['UBBOX_HAND_YOLOV8N'] = 'https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8n.pt?download=true'
    BBOX['UBBOX_HAND_YOLOV8S'] = 'https://huggingface.co/Bingsu/adetailer/resolve/main/hand_yolov8s.pt?download=true'
    BBOX['UBBOX_YOLOV8S'] = 'https://huggingface.co/ultralyticsplus/yolov8s/resolve/main/yolov8s.pt?download=true'

    SEGM['USEGM_DEEPFASHION2_YOLOV8S'] = 'https://huggingface.co/Bingsu/adetailer/resolve/main/deepfashion2_yolov8s-seg.pt?download=true'
    SEGM['USEGM_FACE_YOLOV8M'] = 'https://huggingface.co/jags/yolov8_model_segmentation-set/resolve/main/face_yolov8m-seg_60.pt?download=true'
    SEGM['USEGM_FACE_YOLOV8N'] = 'https://huggingface.co/jags/yolov8_model_segmentation-set/resolve/main/face_yolov8n-seg2_60.pt?download=true'
    SEGM['USEGM_FACIAL_FEATURES_YOLO8X'] = 'https://huggingface.co/jags/yolov8_model_segmentation-set/resolve/main/facial_features_yolo8x-seg.pt?download=true'
    SEGM['USEGM_FLOWERS_SEG_YOLOV8MODEL'] = 'https://huggingface.co/jags/yolov8_model_segmentation-set/resolve/main/flowers_seg_yolov8model.pt?download=true'
    SEGM['USEGM_HAIR_YOLOV8N'] = 'https://huggingface.co/jags/yolov8_model_segmentation-set/resolve/main/hair_yolov8n-seg_60.pt?download=true'
    SEGM['USEGM_PERSON_YOLOV8M'] = 'https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8m-seg.pt?download=true'
    SEGM['USEGM_PERSON_YOLOV8N'] = 'https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8n-seg.pt?download=true'
    SEGM['USEGM_PERSON_YOLOV8S'] = 'https://huggingface.co/Bingsu/adetailer/resolve/main/person_yolov8s-seg.pt?download=true'
    SEGM['USEGM_SKIN_YOLOV8M400'] = 'https://huggingface.co/jags/yolov8_model_segmentation-set/resolve/main/skin_yolov8m-seg_400.pt?download=true'
    SEGM['USEGM_SKIN_YOLOV8N400'] = 'https://huggingface.co/jags/yolov8_model_segmentation-set/resolve/main/skin_yolov8n-seg_400.pt?download=true'
    SEGM['USEGM_SKIN_YOLOV8N800'] = 'https://huggingface.co/jags/yolov8_model_segmentation-set/resolve/main/skin_yolov8n-seg_800.pt?download=true'
    SEGM['USEGM_YOLOV8L'] = 'https://huggingface.co/jags/yolov8_model_segmentation-set/resolve/main/yolov8l-seg.pt?download=true'
    SEGM['USEGM_YOLOV8M'] = 'https://huggingface.co/jags/yolov8_model_segmentation-set/resolve/main/yolov8m-seg.pt?download=true'
    SEGM['USEGM_YOLOV8N'] = 'https://huggingface.co/jags/yolov8_model_segmentation-set/resolve/main/yolov8n-seg.pt?download=true'
    SEGM['USEGM_YOLOV8S'] = 'https://huggingface.co/jags/yolov8_model_segmentation-set/resolve/main/yolov8s-seg.pt?download=true'
    SEGM['USEGM_YOLOV8X'] = 'https://huggingface.co/jags/yolov8_model_segmentation-set/resolve/main/yolov8x-seg.pt?download=true'
    SEGM['USEGM_YOLOV8_BUTTERFLY'] = 'https://huggingface.co/jags/yolov8_model_segmentation-set/resolve/main/yolov8_butterfly_custom.pt?download=true'

    GDINO['GDINO_GROUNDINGDINO_SWINB_COGCOOR'] = 'https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth?download=true'
    GDINO['GDINO_GROUNDINGDINO_SWINB_CFG'] = 'https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py?download=true'
    GDINO['GDINO_GROUNDINGDINO_SWINT_OGC'] = 'https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth?download=true'
    GDINO['GDINO_GROUNDINGDINO_SWINT_OGC_CFG'] = 'https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py?download=true'

    SAMS['SAM_VIT_B_01EC64'] = 'https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_b_01ec64.pth?download=true'
    SAMS['SAM_VIT_H_4B8939'] = 'https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_h_4b8939.pth?download=true'
    SAMS['SAM_VIT_L_0B3195'] = 'https://huggingface.co/ybelkada/segment-anything/resolve/main/checkpoints/sam_vit_l_0b3195.pth?download=true'

    BBOX_PATH = os.path.join(comfy_dir, 'models', 'ultralytics', 'bbox')
    SEGM_PATH = os.path.join(comfy_dir, 'models', 'ultralytics', 'segm')
    GDINO_PATH = os.path.join(comfy_dir, 'models', 'grounding-dino')
    SAMS_PATH = os.path.join(comfy_dir, 'models', 'sams')

    if os.path.exists(BBOX_PATH) == False:
        Path(BBOX_PATH).mkdir(parents=True, exist_ok=True)
    for BBOX_KEY in BBOX:
        FileUrl = BBOX[BBOX_KEY]
        pathparser = urlparse(FileUrl)
        TargetFilename = os.path.basename(pathparser.path)
        FullFilePath = os.path.join(BBOX_PATH, TargetFilename)
        if os.path.isfile(FullFilePath) == False:
            ModelDownload = utility.downloader(FileUrl, FullFilePath)

    if os.path.exists(SEGM_PATH) == False:
        Path(SEGM_PATH).mkdir(parents=True, exist_ok=True)
    for SEGM_KEY in SEGM:
        FileUrl = SEGM[SEGM_KEY]
        pathparser = urlparse(FileUrl)
        TargetFilename = os.path.basename(pathparser.path)
        FullFilePath = os.path.join(SEGM_PATH, TargetFilename)
        if os.path.isfile(FullFilePath) == False:
            ModelDownload = utility.downloader(FileUrl, FullFilePath)

    if os.path.exists(GDINO_PATH) == False:
        Path(GDINO_PATH).mkdir(parents=True, exist_ok=True)
    for GDINO_KEY in GDINO:
        FileUrl = GDINO[GDINO_KEY]
        pathparser = urlparse(FileUrl)
        TargetFilename = os.path.basename(pathparser.path)
        FullFilePath = os.path.join(GDINO_PATH, TargetFilename)
        if os.path.isfile(FullFilePath) == False:
            ModelDownload = utility.downloader(FileUrl, FullFilePath)

    if os.path.exists(SAMS_PATH) == False:
        Path(SAMS_PATH).mkdir(parents=True, exist_ok=True)
    for SAMS_KEY in SAMS:
        FileUrl = SAMS[SAMS_KEY]
        pathparser = urlparse(FileUrl)
        TargetFilename = os.path.basename(pathparser.path)
        FullFilePath = os.path.join(SAMS_PATH, TargetFilename)
        if os.path.isfile(FullFilePath) == False:
            ModelDownload = utility.downloader(FileUrl, FullFilePath)

    BBOX_DIR = os.path.join(comfy_dir, 'models', 'ultralytics', 'bbox')
    SEGM_DIR = os.path.join(comfy_dir, 'models', 'ultralytics', 'segm')
    UL_DIR = os.path.join(comfy_dir, 'models', 'ultralytics')

    folder_paths.add_model_folder_path("ultralytics_bbox", BBOX_DIR)
    folder_paths.add_model_folder_path("ultralytics_segm", SEGM_DIR)
    folder_paths.add_model_folder_path("ultralytics", UL_DIR)

    BBOX_LIST_ALL = folder_paths.get_filename_list("ultralytics_bbox")
    SEGM_LIST_ALL = folder_paths.get_filename_list("ultralytics_segm")

    BBOX_LIST = folder_paths.filter_files_extensions(BBOX_LIST_ALL, ['.pt'])
    SEGM_LIST = folder_paths.filter_files_extensions(SEGM_LIST_ALL, ['.pt'])

    DINO_DIR = os.path.join(comfy_dir, 'models', 'grounding-dino')
    folder_paths.add_model_folder_path("grounding-dino", DINO_DIR)
    DINO_LIST_ALL = folder_paths.get_filename_list("grounding-dino")
    DINO_LIST = folder_paths.filter_files_extensions(DINO_LIST_ALL, ['.pth'])
    DINO_CONFIG_LIST = folder_paths.filter_files_extensions(DINO_LIST_ALL, ['.cfg.py'])

    @classmethod
    def INPUT_TYPES(cls):
        bboxs = ["bbox/"+x for x in cls.BBOX_LIST]
        segms = ["segm/"+x for x in cls.SEGM_LIST]
        dinos = ["dino/"+x for x in cls.DINO_LIST]
        sams = list(filter(lambda x: x.startswith('sam_vit'), folder_paths.get_filename_list("sams")))

        return {
            "required": {
                "use_segments": ("BOOLEAN", {"default": True, "label_on": "ON", "label_off": "OFF"}),
                # "trigger_high_off": ("INT", {"default": 0, "min": 0, "max": utility.MAX_RESOLUTION ** 2, "step": 100}),
                # "trigger_low_off": ("INT", {"default": 0, "min": 0, "max": utility.MAX_RESOLUTION ** 2, "step": 100}),
                "trigger_high_off": ("FLOAT", {"default": 0, "min": 0, "max": 100, "step": 0.05}),
                "trigger_low_off": ("FLOAT", {"default": 0, "min": 0, "max": 100, "step": 0.05}),

                "bbox_segm_model_name": (bboxs + segms,),
                "sam_model_name": (sams,),
                "sam_device_mode": (["AUTO", "Prefer GPU", "CPU"],),

                "search_yolov8s": (['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'],),
                "search_deepfashion2_yolov8s": (['short_sleeved_shirt', 'long_sleeved_shirt', 'short_sleeved_outwear', 'long_sleeved_outwear', 'vest', 'sling', 'shorts', 'trousers', 'skirt', 'short_sleeved_dress', 'long_sleeved_dress', 'vest_dress', 'sling_dress'],),
                "search_facial_features_yolo8x": (['eye', 'eyebrown', 'nose', 'mouth'],),

                "image": ("IMAGE",),

                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                "crop_factor": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 100, "step": 0.1}),
                "drop_size": ("INT", {"min": 1, "max": utility.MAX_RESOLUTION, "step": 1, "default": 10}),
            },
            "optional": {
                "model_version": ("STRING", {"default": 'BaseModel_1024', "forceInput": True}),
                "square_shape": ("INT", {"default": 768, "forceInput": True}),
                "segment_prompt_data": ("TUPLE", {"forceInput": True}),
                "dino_search_prompt": ("STRING", {"default": None, "forceInput": True}),
                "dino_replace_prompt": ("STRING", {"default": None, "forceInput": True}),
            }
        }

    def primere_segments(self, use_segments, bbox_segm_model_name, sam_model_name, sam_device_mode, image, threshold, dilation, crop_factor, drop_size, segment_prompt_data, trigger_high_off = 0, trigger_low_off = 0, search_yolov8s = 'person', search_deepfashion2_yolov8s = "short_sleeved_shirt", search_facial_features_yolo8x = "eye", model_version = 'BaseModel_1024', square_shape = 768, dino_search_prompt = None, dino_replace_prompt = None):
        if segment_prompt_data is None:
            segment_prompt_data = {}

        segment_settings = dict()
        segment_settings['bbox_segm_model_name'] = bbox_segm_model_name
        segment_settings['sam_model_name'] = sam_model_name
        segment_settings['search_yolov8s'] = search_yolov8s
        segment_settings['search_deepfashion2_yolov8s'] = search_deepfashion2_yolov8s
        segment_settings['search_facial_features_yolo8x'] = search_facial_features_yolo8x
        segment_settings['threshold'] = threshold
        segment_settings['dilation'] = dilation
        segment_settings['crop_factor'] = crop_factor
        segment_settings['drop_size'] = drop_size
        segment_settings['model_version'] = model_version
        segment_settings['use_segments'] = use_segments
        segment_settings['trigger_high_off'] = trigger_high_off
        segment_settings['trigger_low_off'] = trigger_low_off
        empty_segs = [[image.shape[1], image.shape[2]], [], []]

        if use_segments == False:
            return image, [image], None, None, empty_segs, [], 0, 0, segment_settings, segment_prompt_data['cond_positive'], segment_prompt_data['cond_negative']

        image_size = [image.shape[2], image.shape[1]]
        input_image_area = (image.shape[2] * image.shape[1])
        segment_settings['input_image_size'] = [image.shape[2], image.shape[1]]
        segment_settings['input_image_area'] = input_image_area

        if image.shape[2] * image.shape[1] > square_shape ** 2:
            if (image.shape[2] > image.shape[1]):
                orientation = 'Horizontal'
            else:
                orientation = 'Vertical'

            image_sides = sorted(image_size)
            custom_side_b = round((image_sides[1] / image_sides[0]), 4)
            dimensions = utility.calculate_dimensions(self, "Square [1:1]", orientation, False, model_version, True, 1, custom_side_b)
            new_width = dimensions[0]
            new_height = dimensions[1]
            image = utility.img_resizer(image, new_width, new_height, 'bicubic')

        model_path = folder_paths.get_full_path("ultralytics", bbox_segm_model_name)
        model = detectors.load_yolo(model_path)

        sam_modelname = folder_paths.get_full_path("sams", sam_model_name)
        if 'vit_h' in sam_modelname:
            model_kind = 'vit_h'
        elif 'vit_l' in sam_modelname:
            model_kind = 'vit_l'
        else:
            model_kind = 'vit_b'

        sam = sam_model_registry[model_kind](checkpoint = sam_modelname)
        device = comfy.model_management.get_torch_device() if sam_device_mode == "Prefer GPU" else "CPU"
        if sam_device_mode == "Prefer GPU":
            sam.to(device = device)

        sam.is_auto_mode = sam_device_mode == "AUTO"

        if bbox_segm_model_name.startswith("bbox") or bbox_segm_model_name.startswith("segm"):
            if bbox_segm_model_name.startswith("bbox"):
                # DETECTOR_RESULT = detectors.NO_SEGM_DETECTOR()
                DETECTOR_RESULT = detectors.UltraBBoxDetector(model)
            else:
                DETECTOR_RESULT = detectors.UltraSegmDetector(model)

            bbox_segs = DETECTOR_RESULT.detect(image, threshold, dilation, crop_factor, drop_size)
            segs = bbox_segs
            if bbox_segm_model_name.startswith("segm"):
                segs = DETECTOR_RESULT.detect(image, threshold, dilation, crop_factor, drop_size)

        if bbox_segm_model_name.startswith("dino"):
            print('DINO')
            # dino_model = load_groundingdino_model(model_name)
            return image, [image], None, None, empty_segs, [], 0, segment_settings

        if 'yolov8s.pt' in bbox_segm_model_name:
            segs = detectors.filter_segs_by_label(segs, search_yolov8s)

        if 'deepfashion2_yolov8' in bbox_segm_model_name:
            segs = detectors.filter_segs_by_label(segs, search_deepfashion2_yolov8s)

        if 'facial_features_yolo8x' in bbox_segm_model_name:
            segs = detectors.filter_segs_by_label(segs, search_facial_features_yolo8x)

        if (trigger_high_off > 0) or (trigger_low_off > 0):
            segs = detectors.filter_segs_by_percent_trigger(segs, trigger_high_off, trigger_low_off, crop_factor, input_image_area)

        image_max_area = 0
        image_max_area_percent = 0
        if (len(segs[2]) > 0):
            for image_segs in segs[2]:
                image_area = (abs(image_segs[2] - image_segs[0])) * (abs(image_segs[3] - image_segs[1]))
                if (image_area > image_max_area):
                    image_max_area = image_area
                    image_max_area_percent = 100 / (input_image_area / image_max_area)

        image_max_area = int((image_max_area / (crop_factor**2)))
        segment_settings['crop_region'] = segs[2]
        segment_settings['image_size'] = [image.shape[2], image.shape[1]]
        segment_settings['image_max_area'] = image_max_area
        segment_settings['image_max_area_percent'] = image_max_area_percent
        input_img_segs = detectors.segmented_images(segs, image)

        if len(segment_prompt_data) == 7:
            embeddings_final_pos, pooled_pos = advanced_encode(segment_prompt_data['clip'], segment_prompt_data['final_positive'], segment_prompt_data['token_normalization'], segment_prompt_data['weight_interpretation'], w_max=1.0, apply_to_pooled=True)
            embeddings_final_neg, pooled_neg = advanced_encode(segment_prompt_data['clip'], segment_prompt_data['final_negative'], segment_prompt_data['token_normalization'], segment_prompt_data['weight_interpretation'], w_max=1.0, apply_to_pooled=True)
            return image, input_img_segs, DETECTOR_RESULT, sam, segs, segs[2], image_max_area, image_max_area_percent, segment_settings, [[embeddings_final_pos, {"pooled_output": pooled_pos}]], [[embeddings_final_neg, {"pooled_output": pooled_neg}]]
        else:
            return image, input_img_segs, DETECTOR_RESULT, sam, segs, segs[2], image_max_area, image_max_area_percent, segment_settings, segment_prompt_data['cond_positive'], segment_prompt_data['cond_negative']

class PrimereAnyDetailer:
    RETURN_TYPES = ("IMAGE", "IMAGE", "INT", "INT",)
    RETURN_NAMES = ("IMAGE", "CROPPED_REFINED", "WIDTH", "HEIGHT",)
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "any_detailer"
    CATEGORY = TREE_SEGMENTS

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                 "image": ("IMAGE", ),
                 "model": ("MODEL",),
                 "clip": ("CLIP",),
                 "vae": ("VAE",),
                 "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),

                 "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                 "scheduler_name": (comfy.samplers.KSampler.SCHEDULERS,),
                 "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                 "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),

                 "positive": ("CONDITIONING",),
                 "negative": ("CONDITIONING",),
                 "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                 "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                 "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                 "force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),

                 "segment_settings": ("TUPLE",),
                 "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
            },
            "optional": {
                "segs": ("SEGS",),
                "detector": ("DETECTOR",),

                "model_concept": ("STRING", {"default": "Normal", "forceInput": True}),
                "concept_sampler_name": (comfy.samplers.KSampler.SAMPLERS, {"forceInput": True, "default": "euler"}),
                "concept_scheduler_name": (comfy.samplers.KSampler.SCHEDULERS, {"forceInput": True, "default": "normal"}),
                "concept_steps": ("INT", {"default": 4, "forceInput": True}),
                "concept_cfg": ("FLOAT", {"default": 1.0, "forceInput": True}),
            }
        }

    @staticmethod
    def enhance_image(image, model, clip, vae, guide_size, guide_size_for_bbox, seed, steps, cfg, sampler_name, scheduler_name,
                     positive, negative, denoise, feather, noise_mask, force_inpaint,
                     segment_settings, detector, segs,
                     model_concept, cycle = 1):

        # base_multiplier = 1
        # if segment_settings['image_max_area_percent'] > 0:
        #     base_multiplier = round((100 / segment_settings['image_max_area_percent']) / 50, 1)
        # print(base_multiplier)

        # max_size = round(guide_size * 1.2, 2)

        if guide_size in range(0, 300):
            max_size = round(guide_size * 2, 2)
            cycle = cycle * 2
        elif guide_size in range(301, 600):
            max_size = round(guide_size * 1.8, 2)
            cycle = cycle * 2
        elif guide_size in range(601, 2000):
            max_size = round(guide_size * 1.5, 2)
        else:
            max_size = round(guide_size * 1.2, 2)

        if model_concept == "Turbo":
            cycle = 1
            guide_size = round(guide_size * 2, 2)
            max_size = round(max_size * (2 * 2), 2)

        detailer_hook = None
        wildcard_opt = None
        refiner_ratio = None
        refiner_model = None
        refiner_clip = None
        refiner_positive = None
        refiner_negative = None

        if detector is not None and segs is None:
            segm_segs = detector.detect(image, segment_settings['threshold'], segment_settings['dilation'], segment_settings['crop_factor'], segment_settings['drop_size'])

            if (hasattr(detector, 'override_bbox_by_segm') and detector.override_bbox_by_segm and not (detailer_hook is not None and not hasattr(detailer_hook, 'override_bbox_by_segm'))):
                segs = segm_segs
            else:
                segm_mask = detectors.segs_to_combined_mask(segm_segs)
                segs = detectors.segs_bitwise_and_mask(segs, segm_mask)

        if len(segs[1]) > 0:
            enhanced_img, _, cropped_enhanced, cropped_enhanced_alpha, cnet_pil_list, new_segs =  detectors.DetailerForEach.do_detail(image, segs, model, clip, vae, guide_size, guide_size_for_bbox,
                                                                                                                                      max_size, seed, steps, cfg,
                                                                                                                                   sampler_name, scheduler_name, positive, negative, denoise,
                                                                                                                                  feather, noise_mask, force_inpaint,
                                                                                                                                 wildcard_opt, detailer_hook,
                                                                                                                                            refiner_ratio=refiner_ratio, refiner_model=refiner_model, refiner_clip=refiner_clip,
                                                                                                                                            refiner_positive=refiner_positive, refiner_negative=refiner_negative,
                                                                                                                                            model_concept=model_concept, cycle=cycle)
        else:
            enhanced_img = image
            cropped_enhanced = []
            cropped_enhanced_alpha = []
            cnet_pil_list = []

        mask = detectors.segs_to_combined_mask(segs)

        if len(cropped_enhanced) == 0:
            cropped_enhanced = [detectors.empty_pil_tensor()]

        if len(cropped_enhanced_alpha) == 0:
            cropped_enhanced_alpha = [detectors.empty_pil_tensor()]

        if len(cnet_pil_list) == 0:
            cnet_pil_list = [detectors.empty_pil_tensor()]

        return enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list

    def any_detailer(self, image, model, clip, vae, seed,
                     steps, cfg, sampler_name, scheduler_name,
                     positive, negative, denoise, feather, noise_mask, force_inpaint,
                     segment_settings, detector = None, segs = None, cycle = 1,
                     model_concept = "Normal", concept_sampler_name = "euler", concept_scheduler_name = "normal", concept_steps = 20, concept_cfg = 8):

        if segment_settings['use_segments'] == False:
            return image, [image], 0, 0

        if model_concept != "Normal":
            sampler_name = concept_sampler_name
            scheduler_name = concept_scheduler_name
            steps = concept_steps
            cfg = concept_cfg

        result_img = None
        result_mask = None
        result_cropped_enhanced = []
        result_cropped_enhanced_alpha = []
        result_cnet_images = []
        crop_region = segment_settings['crop_region']
        guide_size_for_box = True
        full_area = segment_settings['image_size'][0] * segment_settings['image_size'][1]

        for i, single_image in enumerate(image):
            if i < len(crop_region):
                image_segs = crop_region[i]
                size_1 = (abs(image_segs[2] - image_segs[0]))
                size_2 =(abs(image_segs[3] - image_segs[1]))
                part_area = size_1 * size_2
                area_diff = full_area / part_area
                guided_size_multiplier = round(math.pow(area_diff, (1/3.4)), 2)
                if size_1 > size_2:
                   guide_size = size_1 * guided_size_multiplier
                else:
                   guide_size = size_2 * guided_size_multiplier
            else:
                guide_size = round(math.sqrt(full_area), 2)

            enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list = PrimereAnyDetailer.enhance_image(single_image.unsqueeze(0), model, clip, vae, guide_size, guide_size_for_box, seed + i, steps, cfg, sampler_name, scheduler_name, positive, negative, denoise, feather, noise_mask, force_inpaint, segment_settings, detector, segs, model_concept, cycle=cycle)

            result_img = torch.cat((result_img, enhanced_img), dim=0) if result_img is not None else enhanced_img
            result_mask = torch.cat((result_mask, mask), dim=0) if result_mask is not None else mask
            result_cropped_enhanced.extend(cropped_enhanced)
            result_cropped_enhanced_alpha.extend(cropped_enhanced_alpha)
            result_cnet_images.extend(cnet_pil_list)

        return result_img, result_cropped_enhanced, segment_settings['image_size'][0], segment_settings['image_size'][1]