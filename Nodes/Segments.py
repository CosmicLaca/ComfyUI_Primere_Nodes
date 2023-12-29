from ..components.tree import TREE_SEGMENTS
from ..components.tree import PRIMERE_ROOT

import os
import folder_paths
from ..utils import comfy_dir
from ..components import detectors
import comfy
from segment_anything import sam_model_registry
from ..components import utility
import torch

class PrimereImageSegments:
    RETURN_TYPES = ("DETECTOR", "SAM_MODEL", "SEGS", "TUPLE", "INT", "TUPLE")
    RETURN_NAMES = ("DETECTOR", "SAM_MODEL", "SEGS", "CROP_REGIONS", "IMAGE_MAX", "SEGMENT_SETTINGS")
    FUNCTION = "primere_segments"
    CATEGORY = TREE_SEGMENTS

    # model_path = folder_paths.models_dir
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

    @classmethod
    def INPUT_TYPES(cls):
        bboxs = ["bbox/"+x for x in cls.BBOX_LIST]
        segms = ["segm/"+x for x in cls.SEGM_LIST]

        return {
            "required": {
                "bbox_segm_model_name": (bboxs + segms,),
                "sam_model_name": (folder_paths.get_filename_list("sams"),),
                "sam_device_mode": (["AUTO", "Prefer GPU", "CPU"],),

                "image": ("IMAGE",),
                "threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                "crop_factor": ("FLOAT", {"default": 1.2, "min": 1.0, "max": 100, "step": 0.1}),
                "drop_size": ("INT", {"min": 1, "max": utility.MAX_RESOLUTION, "step": 1, "default": 10}),
                # "labels": ("STRING", {"multiline": True, "default": "all", "placeholder": "List the types of segments to be allowed, separated by commas"}),
            }
        }

    def primere_segments(self, bbox_segm_model_name, sam_model_name, sam_device_mode, image, threshold, dilation, crop_factor, drop_size):
        model_path = folder_paths.get_full_path("ultralytics", bbox_segm_model_name)
        model = detectors.load_yolo(model_path)

        segment_settings = dict()
        segment_settings['threshold'] = threshold
        segment_settings['dilation'] = dilation
        segment_settings['crop_factor'] = crop_factor
        segment_settings['drop_size'] = drop_size

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

        if bbox_segm_model_name.startswith("bbox"):
            # DETECTOR_RESULT = detectors.NO_SEGM_DETECTOR()
            DETECTOR_RESULT = detectors.UltraBBoxDetector(model)
        else:
            DETECTOR_RESULT = detectors.UltraSegmDetector(model)

        bbox_segs = DETECTOR_RESULT.detect(image, threshold, dilation, crop_factor, drop_size)
        segs = bbox_segs
        if bbox_segm_model_name.startswith("segm"):
            segs = DETECTOR_RESULT.detect(image, threshold, dilation, crop_factor, drop_size)

        image_max_area = 0
        if (len(segs[2]) > 0):
            for image_segs in segs[2]:
                image_area = (abs(image_segs[2] - image_segs[0])) * (abs(image_segs[3] - image_segs[1]))
                if (image_area > image_max_area):
                    image_max_area = image_area

        segment_settings['crop_region'] = segs[2]
        return DETECTOR_RESULT, sam, segs, segs[2], image_max_area, segment_settings


class PrimereAnyDetailer:
    RETURN_TYPES = ("IMAGE", "IMAGE")
    RETURN_NAMES = ("IMAGE", "CROPPED_REFINED")
    OUTPUT_IS_LIST = (False, True)
    FUNCTION = "any_detailer"
    CATEGORY = TREE_SEGMENTS

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
                     "image": ("IMAGE", ),
                     "model": ("MODEL",),
                     "clip": ("CLIP",),
                     "vae": ("VAE",),
                     "guided_size_multiplier": ("FLOAT", {"default": 1.5, "min": 0.1, "max": 10.0, "step": 0.1}),
                     # "guide_size": ("FLOAT", {"default": 256, "min": 64, "max": utility.MAX_RESOLUTION, "step": 8}),
                     # "guide_size_for_box": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
                     # "max_size": ("FLOAT", {"default": 768, "min": 64, "max": utility.MAX_RESOLUTION, "step": 8}),
                     "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                     "steps": ("INT", {"default": 20, "min": 1, "max": 10000}),
                     "cfg": ("FLOAT", {"default": 8.0, "min": 0.0, "max": 100.0}),
                     "sampler_name": (comfy.samplers.KSampler.SAMPLERS,),
                     "scheduler": (comfy.samplers.KSampler.SCHEDULERS,),
                     "positive": ("CONDITIONING",),
                     "negative": ("CONDITIONING",),
                     "denoise": ("FLOAT", {"default": 0.5, "min": 0.0001, "max": 1.0, "step": 0.01}),
                     "feather": ("INT", {"default": 5, "min": 0, "max": 100, "step": 1}),
                     "noise_mask": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),
                     "force_inpaint": ("BOOLEAN", {"default": True, "label_on": "enabled", "label_off": "disabled"}),

                     # "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                     # "bbox_dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                     # "bbox_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),

                     # "sam_detection_hint": (["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "mask-area", "mask-points", "mask-point-bbox", "none"],),
                     # "sam_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                     # "sam_threshold": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01}),
                     # "sam_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                     # "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                     # "sam_mask_hint_use_negative": (["False", "Small", "Outter"],),
                     # "drop_size": ("INT", {"min": 1, "max": utility.MAX_RESOLUTION, "step": 1, "default": 10}),

                     # "detector": ("DETECTOR", ),
                     # "segs": ("SEGS",),
                     "segment_settings": ("TUPLE",),
                     # "wildcard": ("STRING", {"multiline": True, "dynamicPrompts": False}),

                     "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                },
                "optional": {
                    "segs": ("SEGS",),
                    "detector": ("DETECTOR",),
                    # "sam_model_opt": ("SAM_MODEL", ),
                    # "segm_detector": ("SEGM_DETECTOR", ),
                    # "detailer_hook": ("DETAILER_HOOK",)
                }
        }

    @staticmethod
    def enhance_face(image, model, clip, vae, guide_size, guide_size_for_bbox, seed, steps, cfg, sampler_name, scheduler,
                     positive, negative, denoise, feather, noise_mask, force_inpaint,
                     # bbox_threshold, bbox_dilation, bbox_crop_factor,
                     # sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                     # sam_mask_hint_use_negative, drop_size, bbox_detector,
                     segment_settings, detector = None, segs = None,
                     # segm_detector=None, sam_model_opt=None,
                     cycle=1):

        max_size = guide_size * 1.25
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
            enhanced_img, _, cropped_enhanced, cropped_enhanced_alpha, cnet_pil_list, new_segs = \
                detectors.DetailerForEach.do_detail(image, segs, model, clip, vae, guide_size, guide_size_for_bbox, max_size, seed, steps, cfg,
                                                    sampler_name, scheduler, positive, negative, denoise, feather, noise_mask, force_inpaint,
                                                    wildcard_opt, detailer_hook,
                                                    refiner_ratio=refiner_ratio, refiner_model=refiner_model, refiner_clip=refiner_clip, refiner_positive=refiner_positive, refiner_negative=refiner_negative, cycle=cycle)
        else:
            enhanced_img = image
            cropped_enhanced = []
            cropped_enhanced_alpha = []
            cnet_pil_list = []

        # Mask Generator
        mask = detectors.segs_to_combined_mask(segs)

        if len(cropped_enhanced) == 0:
            cropped_enhanced = [detectors.empty_pil_tensor()]

        if len(cropped_enhanced_alpha) == 0:
            cropped_enhanced_alpha = [detectors.empty_pil_tensor()]

        if len(cnet_pil_list) == 0:
            cnet_pil_list = [detectors.empty_pil_tensor()]

        return enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list

    def any_detailer(self, image, model, clip, vae, guided_size_multiplier,
                     # guide_size, guide_size_for_box,
                     seed, steps, cfg, sampler_name, scheduler,
             positive, negative, denoise, feather, noise_mask, force_inpaint,
             # bbox_threshold, bbox_dilation, bbox_crop_factor,
             # sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
             # sam_mask_hint_use_negative, drop_size, bbox_detector,
             detector, segs, segment_settings, cycle=1):

        result_img = None
        result_mask = None
        result_cropped_enhanced = []
        result_cropped_enhanced_alpha = []
        result_cnet_images = []
        crop_region = segment_settings['crop_region']
        guide_size_for_box = True

        for i, single_image in enumerate(image):
            if i < len(crop_region):
                image_segs = crop_region[i]
                size_1 = (abs(image_segs[2] - image_segs[0]))
                size_2 =(abs(image_segs[3] - image_segs[1]))
                if size_1 > size_2:
                    guide_size = size_1 * guided_size_multiplier
                else:
                    guide_size = size_2 * guided_size_multiplier
            else:
                guide_size = 512

            enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list = PrimereAnyDetailer.enhance_face(
                single_image.unsqueeze(0), model, clip, vae, guide_size, guide_size_for_box, seed + i, steps, cfg, sampler_name, scheduler,
                positive, negative, denoise, feather, noise_mask, force_inpaint,
                segment_settings, detector, segs, cycle=cycle
                # bbox_threshold, bbox_dilation, bbox_crop_factor,
                # sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                # sam_mask_hint_use_negative, drop_size, bbox_detector, segment_settings, segs, segm_detector, sam_model_opt, cycle=cycle
            )

            result_img = torch.cat((result_img, enhanced_img), dim=0) if result_img is not None else enhanced_img
            result_mask = torch.cat((result_mask, mask), dim=0) if result_mask is not None else mask
            result_cropped_enhanced.extend(cropped_enhanced)
            result_cropped_enhanced_alpha.extend(cropped_enhanced_alpha)
            result_cnet_images.extend(cnet_pil_list)

        # pipe = (model, clip, vae, positive, negative, wildcard, bbox_detector, segm_detector, sam_model_opt, detailer_hook, None, None, None, None)
        return result_img, result_cropped_enhanced #, result_cropped_enhanced_alpha, result_mask, pipe, result_cnet_images