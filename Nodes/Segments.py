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
        '''
        if labels is not None and labels != '':
            labels_bbox = labels.split(',')
            if len(labels_bbox) > 0:
                bbox_segs, _ = detectors.SEGSLabelFilter.filter(bbox_segs, labels_bbox)
        '''

        segm_segs = bbox_segs
        if bbox_segm_model_name.startswith("segm"):
            segm_segs = SEGM_DETECTOR_RESULT.detect(image, threshold, dilation, crop_factor, drop_size)
            '''
            if labels is not None and labels != '':
                labels_segm = labels.split(',')
                if len(labels_segm) > 0:
                    segm_segs, _ = detectors.SEGSLabelFilter.filter(segm_segs, labels_segm)
            '''

        image_max_area = 0
        if (len(segm_segs[2]) > 0):
            for image_segs in bbox_segs[2]:
                image_area = (abs(image_segs[2] - image_segs[0])) * (abs(image_segs[3] - image_segs[1]))
                if (image_area > image_max_area):
                    image_max_area = image_area

        return BBOX_DETECTOR_RESULT, SEGM_DETECTOR_RESULT, sam, bbox_segs, segm_segs, bbox_segs[2], image_max_area


class PrimereAnyDetailer:
    RETURN_TYPES = ("IMAGE", "IMAGE", "IMAGE", "MASK", "DETAILER_PIPE", "IMAGE")
    RETURN_NAMES = ("IMAGE", "CROPPED_REFINED", "CROPPED_ENHANCED_ALPHA", "MASK", "DETAILER_PIPE", "CNET_IMAGES")
    OUTPUT_IS_LIST = (False, True, True, False, False, True)
    FUNCTION = "any_detailer"
    CATEGORY = TREE_SEGMENTS

    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                     "image": ("IMAGE", ),
                     "model": ("MODEL",),
                     "clip": ("CLIP",),
                     "vae": ("VAE",),
                     "guide_size": ("FLOAT", {"default": 256, "min": 64, "max": utility.MAX_RESOLUTION, "step": 8}),
                     "guide_size_for": ("BOOLEAN", {"default": True, "label_on": "bbox", "label_off": "crop_region"}),
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

                     "bbox_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "bbox_dilation": ("INT", {"default": 10, "min": -512, "max": 512, "step": 1}),
                     "bbox_crop_factor": ("FLOAT", {"default": 3.0, "min": 1.0, "max": 10, "step": 0.1}),

                     "sam_detection_hint": (["center-1", "horizontal-2", "vertical-2", "rect-4", "diamond-4", "mask-area", "mask-points", "mask-point-bbox", "none"],),
                     "sam_dilation": ("INT", {"default": 0, "min": -512, "max": 512, "step": 1}),
                     "sam_threshold": ("FLOAT", {"default": 0.93, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "sam_bbox_expansion": ("INT", {"default": 0, "min": 0, "max": 1000, "step": 1}),
                     "sam_mask_hint_threshold": ("FLOAT", {"default": 0.7, "min": 0.0, "max": 1.0, "step": 0.01}),
                     "sam_mask_hint_use_negative": (["False", "Small", "Outter"],),

                     "drop_size": ("INT", {"min": 1, "max": utility.MAX_RESOLUTION, "step": 1, "default": 10}),

                     "bbox_detector": ("BBOX_DETECTOR", ),
                     # "wildcard": ("STRING", {"multiline": True, "dynamicPrompts": False}),

                     "cycle": ("INT", {"default": 1, "min": 1, "max": 10, "step": 1}),
                },
                "optional": {
                    "sam_model_opt": ("SAM_MODEL", ),
                    "segm_detector_opt": ("SEGM_DETECTOR", ),
                    # "detailer_hook": ("DETAILER_HOOK",)
                }
        }


    def detect(self, image, threshold, dilation, crop_factor, bbox_model, drop_size=1, detailer_hook=None):
        drop_size = max(drop_size, 1)
        mmdet_results = detectors.inference_bbox(bbox_model, image, threshold)
        segmasks = detectors.create_segmasks(mmdet_results)

        if dilation > 0:
            segmasks = detectors.dilate_masks(segmasks, dilation)

        items = []
        h = image.shape[1]
        w = image.shape[2]

        for x in segmasks:
            item_bbox = x[0]
            item_mask = x[1]

            y1, x1, y2, x2 = item_bbox

            if x2 - x1 > drop_size and y2 - y1 > drop_size:  # minimum dimension must be (2,2) to avoid squeeze issue
                crop_region = detectors.make_crop_region(w, h, item_bbox, crop_factor)
                cropped_image = detectors.crop_image(image, crop_region)
                cropped_mask = detectors.crop_ndarray2(item_mask, crop_region)
                confidence = x[2]
                # bbox_size = (item_bbox[2]-item_bbox[0],item_bbox[3]-item_bbox[1]) # (w,h)

                item = detectors.SEG(cropped_image, cropped_mask, confidence, crop_region, item_bbox, None, None)

                items.append(item)

        shape = image.shape[1], image.shape[2]
        return shape, items

    def setAux(self, x):
        pass

    @staticmethod
    def enhance_face(image, model, clip, vae, guide_size, guide_size_for_bbox, seed, steps, cfg, sampler_name, scheduler,
                     positive, negative, denoise, feather, noise_mask, force_inpaint,
                     bbox_threshold, bbox_dilation, bbox_crop_factor,
                     sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                     sam_mask_hint_use_negative, drop_size,
                     bbox_detector, segm_detector=None, sam_model_opt=None, wildcard_opt=None, detailer_hook=None,
                     refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None, refiner_negative=None, cycle=1):

        max_size = guide_size * 1.5

        # make default prompt as 'face' if empty prompt for CLIPSeg
        # bbox_detector.setAux('face')
        segs = bbox_detector.detect(image, bbox_threshold, bbox_dilation, bbox_crop_factor, sam_model_opt, drop_size, detailer_hook=detailer_hook)
        bbox_detector.setAux(None)

        # bbox + sam combination
        if sam_model_opt is not None:
            sam_mask = detectors.make_sam_mask(sam_model_opt, segs, image, sam_detection_hint, sam_dilation,
                                          sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                                          sam_mask_hint_use_negative, )
            segs = detectors.segs_bitwise_and_mask(segs, sam_mask)

        elif segm_detector is not None:
            segm_segs = segm_detector.detect(image, bbox_threshold, bbox_dilation, bbox_crop_factor, drop_size)

            if (hasattr(segm_detector, 'override_bbox_by_segm') and segm_detector.override_bbox_by_segm and
                    not (detailer_hook is not None and not hasattr(detailer_hook, 'override_bbox_by_segm'))):
                segs = segm_segs
            else:
                segm_mask = detectors.segs_to_combined_mask(segm_segs)
                segs = detectors.segs_bitwise_and_mask(segs, segm_mask)

        if len(segs[1]) > 0:
            enhanced_img, _, cropped_enhanced, cropped_enhanced_alpha, cnet_pil_list, new_segs = \
                detectors.DetailerForEach.do_detail(image, segs, model, clip, vae, guide_size, guide_size_for_bbox, max_size, seed, steps, cfg,
                                          sampler_name, scheduler, positive, negative, denoise, feather, noise_mask,
                                          force_inpaint, wildcard_opt, detailer_hook,
                                          refiner_ratio=refiner_ratio, refiner_model=refiner_model,
                                          refiner_clip=refiner_clip, refiner_positive=refiner_positive,
                                          refiner_negative=refiner_negative, cycle=cycle)
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

    def any_detailer(self, image, model, clip, vae, guide_size, guide_size_for, seed, steps, cfg, sampler_name, scheduler,
             positive, negative, denoise, feather, noise_mask, force_inpaint,
             bbox_threshold, bbox_dilation, bbox_crop_factor,
             sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
             sam_mask_hint_use_negative, drop_size, bbox_detector, cycle=1,
             sam_model_opt=None, segm_detector_opt=None, detailer_hook=None):

        max_size = guide_size * 1.5

        result_img = None
        result_mask = None
        result_cropped_enhanced = []
        result_cropped_enhanced_alpha = []
        result_cnet_images = []
        wildcard = False

        for i, single_image in enumerate(image):
            enhanced_img, cropped_enhanced, cropped_enhanced_alpha, mask, cnet_pil_list = PrimereAnyDetailer.enhance_face(
                single_image.unsqueeze(0), model, clip, vae, guide_size, guide_size_for, max_size, seed + i, steps, cfg, sampler_name, scheduler,
                positive, negative, denoise, feather, noise_mask, force_inpaint,
                bbox_threshold, bbox_dilation, bbox_crop_factor,
                sam_detection_hint, sam_dilation, sam_threshold, sam_bbox_expansion, sam_mask_hint_threshold,
                sam_mask_hint_use_negative, drop_size, bbox_detector, segm_detector_opt, sam_model_opt, wildcard, detailer_hook, cycle=cycle)

            result_img = torch.cat((result_img, enhanced_img), dim=0) if result_img is not None else enhanced_img
            result_mask = torch.cat((result_mask, mask), dim=0) if result_mask is not None else mask
            result_cropped_enhanced.extend(cropped_enhanced)
            result_cropped_enhanced_alpha.extend(cropped_enhanced_alpha)
            result_cnet_images.extend(cnet_pil_list)

        pipe = (model, clip, vae, positive, negative, wildcard, bbox_detector, segm_detector_opt, sam_model_opt, detailer_hook, None, None, None, None)
        return result_img, result_cropped_enhanced, result_cropped_enhanced_alpha, result_mask, pipe, result_cnet_images