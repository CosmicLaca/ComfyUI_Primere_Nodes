from ..components.tree import TREE_SEGMENTS
from ..components.tree import PRIMERE_ROOT

import os
import folder_paths
from ..utils import comfy_dir
from ..components import detectors
import comfy
from segment_anything import sam_model_registry, SamPredictor
from ..components import utility
import json

class PrimereImageSegments:
    RETURN_TYPES = ("BBOX_DETECTOR", "SEGM_DETECTOR", "SAM_MODEL", "SEGS", "SEGS", "TUPLE", "INT")
    RETURN_NAMES = ("BBOX_DETECTOR", "SEGM_DETECTOR", "SAM_MODEL", "BBOX_SEGS", "SEGM_SEGS", "CROP_REGIONS", "IMAGE_MAX")
    FUNCTION = "primere_segments"
    CATEGORY = TREE_SEGMENTS

    # model_path = folder_paths.models_dir
    BBOX_DIR = os.path.join(comfy_dir, 'models', 'ultralytics', 'bbox')
    SEGM_DIR = os.path.join(comfy_dir, 'models', 'ultralytics', 'segm')
    UL_DIR = os.path.join(comfy_dir, 'models', 'ultralytics')

    folder_paths.add_model_folder_path("ultralytics_bbox", BBOX_DIR)
    folder_paths.add_model_folder_path("ultralytics_segm", SEGM_DIR)
    folder_paths.add_model_folder_path("ultralytics", UL_DIR)

    BBOX_LIST = folder_paths.get_filename_list("ultralytics_bbox")
    SEGM_LIST = folder_paths.get_filename_list("ultralytics_segm")

    @classmethod
    def INPUT_TYPES(cls):
        bboxs = ["bbox/"+x for x in cls.BBOX_LIST]
        segms = ["segm/"+x for x in cls.SEGM_LIST]

        return {
            "required": {
                "bbox_segm_model_name": (bboxs + segms,),
                "sam_model_name": (folder_paths.get_filename_list("sams"),),
                "sam_device_mode": (["AUTO", "Prefer GPU", "CPU"],),

                # "bbox_detector": ("BBOX_DETECTOR",),
                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                "crop_factor": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 100, "step": 0.1}),
                "drop_size": ("INT", {"min": 1, "max": utility.MAX_RESOLUTION, "step": 1, "default": 10}),
                "labels": ("STRING", {"multiline": True, "default": "all", "placeholder": "List the types of segments to be allowed, separated by commas"}),
            },
            # "optional": {
            #     "segm_detector": ("SEGM_DETECTOR",),
            # }
        }

    def primere_segments(self, bbox_segm_model_name, sam_model_name, sam_device_mode, image, threshold, dilation, crop_factor, drop_size, labels=None):
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

        BBOX_DETECTOR_RESULT = detectors.UltraBBoxDetector(model)
        if bbox_segm_model_name.startswith("bbox"):
            SEGM_DETECTOR_RESULT = detectors.NO_SEGM_DETECTOR()
        else:
            SEGM_DETECTOR_RESULT = detectors.UltraSegmDetector(model)

        bbox_segs = BBOX_DETECTOR_RESULT.detect(image, threshold, dilation, crop_factor, drop_size)
        if labels is not None and labels != '':
            labels = labels.split(',')
            if len(labels) > 0:
                bbox_segs, _ = detectors.SEGSLabelFilter.filter(bbox_segs, labels)

        '''
        segm_segs = SEGM_DETECTOR_RESULT.detect(image, threshold, dilation, crop_factor, drop_size)
        if labels is not None and labels != '':
            labels = labels.split(',')
            if len(labels) > 0:
                segm_segs, _ = detectors.SEGSLabelFilter.filter(segm_segs, labels)
        '''

        image_max_area = 0
        if (len(bbox_segs[2]) > 0):
            for image_segs in bbox_segs[2]:
                image_area = (abs(image_segs[2] - image_segs[0])) * (abs(image_segs[3] - image_segs[1]))
                if (image_area > image_max_area):
                    image_max_area = image_area

        return BBOX_DETECTOR_RESULT, SEGM_DETECTOR_RESULT, sam, bbox_segs, bbox_segs, bbox_segs[2], image_max_area

        '''
        if bbox_segm_model_name.startswith("bbox"):
            return detectors.UltraBBoxDetector(model), detectors.NO_SEGM_DETECTOR(), sam, bbox_segs, segm_segs
        else:
            return detectors.UltraBBoxDetector(model), detectors.UltraSegmDetector(model), sam, bbox_segs, segm_segs
        '''