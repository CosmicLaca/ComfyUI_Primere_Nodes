from ultralytics import YOLO
import cv2
from PIL import Image
import numpy as np
import torch
from collections import namedtuple
import comfy
from segment_anything import SamPredictor
import re
import random
import nodes
import os
import folder_paths
import threading
import torchvision
import math
import comfy_extras.nodes_custom_sampler as nodes_custom_sampler
from ..Nodes import Outputs
from ..Nodes import Segments
from ..components.tree import PRIMERE_ROOT
from ..components import utility
import comfy_extras.nodes_mask as nodes_mask
from ..Nodes.modules.adv_encode import advanced_encode

def inference_bbox(model, image: Image.Image, confidence: float = 0.3, device: str = "",):
    pred = model(image, conf=confidence, device=device)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    cv2_image = np.array(image)
    if len(cv2_image.shape) == 3:
        cv2_image = cv2_image[:, :, ::-1].copy()  # Convert RGB to BGR for cv2 processing
    else:
        # Handle the grayscale image here
        # For example, you might want to convert it to a 3-channel grayscale image for consistency:
        cv2_image = cv2.cvtColor(cv2_image, cv2.COLOR_GRAY2BGR)
    cv2_gray = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2GRAY)

    segms = []
    for x0, y0, x1, y1 in bboxes:
        cv2_mask = np.zeros(cv2_gray.shape, np.uint8)
        cv2.rectangle(cv2_mask, (int(x0), int(y0)), (int(x1), int(y1)), 255, -1)
        cv2_mask_bool = cv2_mask.astype(bool)
        segms.append(cv2_mask_bool)

    n, m = bboxes.shape
    if n == 0:
        return [[], [], [], []]

    results = [[], [], [], []]
    for i in range(len(bboxes)):
        results[0].append(pred[0].names[int(pred[0].boxes[i].cls.item())])
        results[1].append(bboxes[i])
        results[2].append(segms[i])
        results[3].append(pred[0].boxes[i].conf.cpu().numpy())

    return results

def inference_segm(model, image: Image.Image, confidence: float = 0.3, device: str = "",):
    pred = model(image, conf=confidence, device=device)

    bboxes = pred[0].boxes.xyxy.cpu().numpy()
    n, m = bboxes.shape
    if n == 0:
        return [[], [], [], []]

    # NOTE: masks.data will be None when n == 0
    segms = pred[0].masks.data.cpu().numpy()

    results = [[], [], [], []]
    for i in range(len(bboxes)):
        results[0].append(pred[0].names[int(pred[0].boxes[i].cls.item())])
        results[1].append(bboxes[i])

        mask = torch.from_numpy(segms[i])
        scaled_mask = torch.nn.functional.interpolate(mask.unsqueeze(0).unsqueeze(0), size=(image.size[1], image.size[0]), mode='bilinear', align_corners=False)
        scaled_mask = scaled_mask.squeeze().squeeze()

        results[2].append(scaled_mask.numpy())
        results[3].append(pred[0].boxes[i].conf.cpu().numpy())

    return results

def load_yolo(model_path: str):
    try:
        return YOLO(model_path)
    except ModuleNotFoundError:
        YOLO("yolov8n.pt")
        return YOLO(model_path)

def create_segmasks(results):
    bboxs = results[1]
    segms = results[2]
    confidence = results[3]

    results = []
    for i in range(len(segms)):
        item = (bboxs[i], segms[i].astype(np.float32), confidence[i])
        results.append(item)
    return results

def dilate_masks(segmasks, dilation_factor, iter=1):
    if dilation_factor == 0:
        return segmasks

    dilated_masks = []
    kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)

    gpu_kernel = cv2.UMat(kernel)

    for i in range(len(segmasks)):
        cv2_mask = segmasks[i][1]
        gpu_mask = cv2.UMat(cv2_mask)

        if dilation_factor > 0:
            dilated_mask = cv2.dilate(gpu_mask, gpu_kernel, iter).get()
        else:
            dilated_mask = cv2.erode(gpu_mask, gpu_kernel, iter).get()

        item = (segmasks[i][0], dilated_mask, segmasks[i][2])
        dilated_masks.append(item)

    return dilated_masks

def tensor2pil(image):
    return Image.fromarray(np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8))

def normalize_region(limit, startp, size):
    if startp < 0:
        new_endp = min(limit, size)
        new_startp = 0
    elif startp + size > limit:
        new_startp = max(0, limit - size)
        new_endp = limit
    else:
        new_startp = startp
        new_endp = min(limit, startp+size)

    return int(new_startp), int(new_endp)

def make_crop_region(w, h, bbox, crop_factor, crop_min_size=None):
    x1 = bbox[0]
    y1 = bbox[1]
    x2 = bbox[2]
    y2 = bbox[3]

    bbox_w = x2 - x1
    bbox_h = y2 - y1

    crop_w = bbox_w * crop_factor
    crop_h = bbox_h * crop_factor

    if crop_min_size is not None:
        crop_w = max(crop_min_size, crop_w)
        crop_h = max(crop_min_size, crop_h)

    kernel_x = x1 + bbox_w / 2
    kernel_y = y1 + bbox_h / 2

    new_x1 = int(kernel_x - crop_w / 2)
    new_y1 = int(kernel_y - crop_h / 2)

    # make sure position in (w,h)
    new_x1, new_x2 = normalize_region(w, new_x1, crop_w)
    new_y1, new_y2 = normalize_region(h, new_y1, crop_h)

    return [new_x1, new_y1, new_x2, new_y2]

def crop_ndarray2(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[y1:y2, x1:x2]

    return cropped

def crop_ndarray4(npimg, crop_region):
    x1 = crop_region[0]
    y1 = crop_region[1]
    x2 = crop_region[2]
    y2 = crop_region[3]

    cropped = npimg[:, y1:y2, x1:x2, :]

    return cropped

def crop_image(image, crop_region):
    return crop_ndarray4(np.array(image), crop_region)

SEG = namedtuple("SEG", ['cropped_image', 'cropped_mask', 'confidence', 'crop_region', 'bbox', 'label', 'control_net_wrapper'], defaults=[None])

def combine_masks(masks):
    if len(masks) == 0:
        return None
    else:
        initial_cv2_mask = np.array(masks[0][1])
        combined_cv2_mask = initial_cv2_mask

        for i in range(1, len(masks)):
            cv2_mask = np.array(masks[i][1])

            if combined_cv2_mask.shape == cv2_mask.shape:
                combined_cv2_mask = cv2.bitwise_or(combined_cv2_mask, cv2_mask)
            else:
                # do nothing - incompatible mask
                pass

        mask = torch.from_numpy(combined_cv2_mask)
        return mask

class UltraBBoxDetector:
    bbox_model = None

    def __init__(self, bbox_model):
        self.bbox_model = bbox_model

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        drop_size = max(drop_size, 1)
        detected_results = inference_bbox(self.bbox_model, tensor2pil(image), threshold)
        segmasks = create_segmasks(detected_results)
        crop_region_all = []

        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        items = []
        h = image.shape[1]
        w = image.shape[2]

        for x, label in zip(segmasks, detected_results[0]):
            item_bbox = x[0]
            item_mask = x[1]

            y1, x1, y2, x2 = item_bbox

            if x2 - x1 > drop_size and y2 - y1 > drop_size:  # minimum dimension must be (2,2) to avoid squeeze issue
                crop_region = make_crop_region(w, h, item_bbox, crop_factor)

                if detailer_hook is not None:
                    crop_region = detailer_hook.post_crop_region(w, h, item_bbox, crop_region)

                cropped_image = crop_image(image, crop_region)
                cropped_mask = crop_ndarray2(item_mask, crop_region)
                confidence = x[2]
                # bbox_size = (item_bbox[2]-item_bbox[0],item_bbox[3]-item_bbox[1]) # (w,h)

                crop_region_all.append(crop_region)
                item = SEG(cropped_image, cropped_mask, confidence, crop_region, item_bbox, label, None)
                items.append(item)

        shape = image.shape[1], image.shape[2]
        return shape, items, crop_region_all

    def detect_combined(self, image, threshold, dilation):
        detected_results = inference_bbox(self.bbox_model, tensor2pil(image), threshold)
        segmasks = create_segmasks(detected_results)
        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        return combine_masks(segmasks)

    def setAux(self, x):
        pass


class UltraSegmDetector:
    bbox_model = None

    def __init__(self, bbox_model):
        self.bbox_model = bbox_model

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        drop_size = max(drop_size, 1)
        detected_results = inference_segm(self.bbox_model, tensor2pil(image), threshold)
        segmasks = create_segmasks(detected_results)
        crop_region_all = []

        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        items = []
        h = image.shape[1]
        w = image.shape[2]

        for x, label in zip(segmasks, detected_results[0]):
            item_bbox = x[0]
            item_mask = x[1]

            y1, x1, y2, x2 = item_bbox

            if x2 - x1 > drop_size and y2 - y1 > drop_size:  # minimum dimension must be (2,2) to avoid squeeze issue
                crop_region = make_crop_region(w, h, item_bbox, crop_factor)

                if detailer_hook is not None:
                    crop_region = detailer_hook.post_crop_region(w, h, item_bbox, crop_region)

                cropped_image = crop_image(image, crop_region)
                cropped_mask = crop_ndarray2(item_mask, crop_region)
                confidence = x[2]
                # bbox_size = (item_bbox[2]-item_bbox[0],item_bbox[3]-item_bbox[1]) # (w,h)

                crop_region_all.append(crop_region)
                item = SEG(cropped_image, cropped_mask, confidence, crop_region, item_bbox, label, None)
                items.append(item)

        shape = image.shape[1], image.shape[2]
        return shape, items, crop_region_all

    def detect_combined(self, image, threshold, dilation):
        detected_results = inference_segm(self.bbox_model, tensor2pil(image), threshold)
        segmasks = create_segmasks(detected_results)
        if dilation > 0:
            segmasks = dilate_masks(segmasks, dilation)

        return combine_masks(segmasks)

    def setAux(self, x):
        pass

class NO_BBOX_DETECTOR:
    pass

class NO_SEGM_DETECTOR:
    pass

class SEGSLabelFilter:
    def filter(segs, labels):
        labels = set([label.strip() for label in labels])

        if 'all' in labels:
            return (segs, (segs[0], []), segs[2],)
        else:
            res_segs = []
            remained_segs = []

            for x in segs[1]:
                if x.label in labels:
                    res_segs.append(x)
                elif 'eyes' in labels and x.label in ['left_eye', 'right_eye']:
                    res_segs.append(x)
                elif 'eyebrows' in labels and x.label in ['left_eyebrow', 'right_eyebrow']:
                    res_segs.append(x)
                elif 'pupils' in labels and x.label in ['left_pupil', 'right_pupil']:
                    res_segs.append(x)
                else:
                    remained_segs.append(x)

        return ((segs[0], res_segs, segs[2]), (segs[0], remained_segs, segs[2]),)

def center_of_bbox(bbox):
    w, h = bbox[2] - bbox[0], bbox[3] - bbox[1]
    return bbox[0] + w/2, bbox[1] + h/2

def sam_predict(predictor, points, plabs, bbox, threshold):
    point_coords = None if not points else np.array(points)
    point_labels = None if not plabs else np.array(plabs)
    box = np.array([bbox]) if bbox is not None else None
    cur_masks, scores, _ = predictor.predict(point_coords=point_coords, point_labels=point_labels, box=box)
    total_masks = []
    selected = False
    max_score = 0

    for idx in range(len(scores)):
        if scores[idx] > max_score:
            max_score = scores[idx]
            max_mask = cur_masks[idx]

        if scores[idx] >= threshold:
            selected = True
            total_masks.append(cur_masks[idx])
        else:
            pass

    if not selected:
        total_masks.append(max_mask)

    return total_masks

def make_2d_mask(mask):
    if len(mask.shape) == 4:
        return mask.squeeze(0).squeeze(0)
    elif len(mask.shape) == 3:
        return mask.squeeze(0)

    return mask

def gen_detection_hints_from_mask_area(x, y, mask, threshold, use_negative):
    mask = make_2d_mask(mask)
    points = []
    plabs = []

    y_step = max(3, int(mask.shape[0] / 20))
    x_step = max(3, int(mask.shape[1] / 20))

    for i in range(0, len(mask), y_step):
        for j in range(0, len(mask[i]), x_step):
            if mask[i][j] > threshold:
                points.append((x + j, y + i))
                plabs.append(1)
            elif use_negative and mask[i][j] == 0:
                points.append((x + j, y + i))
                plabs.append(0)

    return points, plabs

def gen_negative_hints(w, h, x1, y1, x2, y2):
    npoints = []
    nplabs = []

    y_step = max(3, int(w / 20))
    x_step = max(3, int(h / 20))

    for i in range(10, h - 10, y_step):
        for j in range(10, w - 10, x_step):
            if not (x1 - 10 <= j and j <= x2 + 10 and y1 - 10 <= i and i <= y2 + 10):
                npoints.append((j, i))
                nplabs.append(0)

    return npoints, nplabs

def combine_masks2(masks):
    if len(masks) == 0:
        return None
    else:
        initial_cv2_mask = np.array(masks[0]).astype(np.uint8)
        combined_cv2_mask = initial_cv2_mask

        for i in range(1, len(masks)):
            cv2_mask = np.array(masks[i]).astype(np.uint8)

            if combined_cv2_mask.shape == cv2_mask.shape:
                combined_cv2_mask = cv2.bitwise_or(combined_cv2_mask, cv2_mask)
            else:
                pass

        mask = torch.from_numpy(combined_cv2_mask)
        return mask

def use_gpu_opencv():
    return not True

def dilate_mask(mask, dilation_factor, iter=1):
    if dilation_factor == 0:
        return mask

    mask = make_2d_mask(mask)
    kernel = np.ones((abs(dilation_factor), abs(dilation_factor)), np.uint8)

    if use_gpu_opencv():
        mask = cv2.UMat(mask)
        kernel = cv2.UMat(kernel)

    if dilation_factor > 0:
        result = cv2.dilate(mask, kernel, iter)
    else:
        result = cv2.erode(mask, kernel, iter)

    if use_gpu_opencv():
        return result.get()
    else:
        return result

def make_3d_mask(mask):
    if len(mask.shape) == 4:
        return mask.squeeze(0)

    elif len(mask.shape) == 2:
        return mask.unsqueeze(0)

    return mask
def make_sam_mask(sam_model, segs, image, detection_hint, dilation, threshold, bbox_expansion, mask_hint_threshold, mask_hint_use_negative):
    if sam_model.is_auto_mode:
        device = comfy.model_management.get_torch_device()
        sam_model.to(device=device)

    try:
        predictor = SamPredictor(sam_model)
        image = np.clip(255. * image.cpu().numpy().squeeze(), 0, 255).astype(np.uint8)
        predictor.set_image(image, "RGB")
        total_masks = []
        use_small_negative = mask_hint_use_negative == "Small"

        # seg_shape = segs[0]
        segs = segs[1]
        if detection_hint == "mask-points":
            points = []
            plabs = []

            for i in range(len(segs)):
                bbox = segs[i].bbox
                center = center_of_bbox(segs[i].bbox)
                points.append(center)

                # small point is background, big point is foreground
                if use_small_negative and bbox[2] - bbox[0] < 10:
                    plabs.append(0)
                else:
                    plabs.append(1)

            detected_masks = sam_predict(predictor, points, plabs, None, threshold)
            total_masks += detected_masks

        else:
            for i in range(len(segs)):
                bbox = segs[i].bbox
                center = center_of_bbox(bbox)

                x1 = max(bbox[0] - bbox_expansion, 0)
                y1 = max(bbox[1] - bbox_expansion, 0)
                x2 = min(bbox[2] + bbox_expansion, image.shape[1])
                y2 = min(bbox[3] + bbox_expansion, image.shape[0])

                dilated_bbox = [x1, y1, x2, y2]

                points = []
                plabs = []
                if detection_hint == "center-1":
                    points.append(center)
                    plabs = [1]  # 1 = foreground point, 0 = background point

                elif detection_hint == "horizontal-2":
                    gap = (x2 - x1) / 3
                    points.append((x1 + gap, center[1]))
                    points.append((x1 + gap * 2, center[1]))
                    plabs = [1, 1]

                elif detection_hint == "vertical-2":
                    gap = (y2 - y1) / 3
                    points.append((center[0], y1 + gap))
                    points.append((center[0], y1 + gap * 2))
                    plabs = [1, 1]

                elif detection_hint == "rect-4":
                    x_gap = (x2 - x1) / 3
                    y_gap = (y2 - y1) / 3
                    points.append((x1 + x_gap, center[1]))
                    points.append((x1 + x_gap * 2, center[1]))
                    points.append((center[0], y1 + y_gap))
                    points.append((center[0], y1 + y_gap * 2))
                    plabs = [1, 1, 1, 1]

                elif detection_hint == "diamond-4":
                    x_gap = (x2 - x1) / 3
                    y_gap = (y2 - y1) / 3
                    points.append((x1 + x_gap, y1 + y_gap))
                    points.append((x1 + x_gap * 2, y1 + y_gap))
                    points.append((x1 + x_gap, y1 + y_gap * 2))
                    points.append((x1 + x_gap * 2, y1 + y_gap * 2))
                    plabs = [1, 1, 1, 1]

                elif detection_hint == "mask-point-bbox":
                    center = center_of_bbox(segs[i].bbox)
                    points.append(center)
                    plabs = [1]

                elif detection_hint == "mask-area":
                    points, plabs = gen_detection_hints_from_mask_area(segs[i].crop_region[0], segs[i].crop_region[1], segs[i].cropped_mask, mask_hint_threshold, use_small_negative)

                if mask_hint_use_negative == "Outter":
                    npoints, nplabs = gen_negative_hints(image.shape[0], image.shape[1], segs[i].crop_region[0], segs[i].crop_region[1], segs[i].crop_region[2], segs[i].crop_region[3])
                    points += npoints
                    plabs += nplabs

                detected_masks = sam_predict(predictor, points, plabs, dilated_bbox, threshold)
                total_masks += detected_masks

        # merge every collected masks
        mask = combine_masks2(total_masks)

    finally:
        if sam_model.is_auto_mode:
            print(f"semd to {device}")
            sam_model.to(device="cpu")

    if mask is not None:
        mask = mask.float()
        mask = dilate_mask(mask.cpu().numpy(), dilation)
        mask = torch.from_numpy(mask)
    else:
        mask = torch.zeros((8, 8), dtype=torch.float32, device="cpu")  # empty mask

    mask = make_3d_mask(mask)
    return mask

def segs_bitwise_and_mask(segs, mask):
    mask = make_2d_mask(mask)

    if mask is None:
        return ([],)

    items = []
    mask = (mask.cpu().numpy() * 255).astype(np.uint8)

    for seg in segs[1]:
        cropped_mask = (seg.cropped_mask * 255).astype(np.uint8)
        crop_region = seg.crop_region
        cropped_mask2 = mask[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]]
        new_mask = np.bitwise_and(cropped_mask.astype(np.uint8), cropped_mask2)
        new_mask = new_mask.astype(np.float32) / 255.0
        item = SEG(seg.cropped_image, new_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, None)
        items.append(item)

    return segs[0], items

def segs_to_combined_mask(segs):
    shape = segs[0]
    h = shape[0]
    w = shape[1]
    mask = np.zeros((h, w), dtype=np.uint8)

    for seg in segs[1]:
        cropped_mask = seg.cropped_mask
        crop_region = seg.crop_region
        mask[crop_region[1]:crop_region[3], crop_region[0]:crop_region[2]] |= (cropped_mask * 255).astype(np.uint8)

    return torch.from_numpy(mask.astype(np.float32) / 255.0)

class WildcardChooserDict:
    def __init__(self, items):
        self.items = items

    def get(self, seg):
        text = ""
        if 'ALL' in self.items:
            text = self.items['ALL']

        if seg.label in self.items:
            text += self.items[seg.label]

        return text

def split_to_dict(text):
    pattern = r'\[([A-Za-z0-9_. ]+)\]([^\[]+)(?=\[|$)'
    matches = re.findall(pattern, text)
    result_dict = {key: value.strip() for key, value in matches}

    return result_dict

class WildcardChooser:
    def __init__(self, items, randomize_when_exhaust):
        self.i = 0
        self.items = items
        self.randomize_when_exhaust = randomize_when_exhaust

    def get(self, seg):
        if self.i >= len(self.items):
            self.i = 0
            if self.randomize_when_exhaust:
                random.shuffle(self.items)

        item = self.items[self.i]
        self.i += 1

        return item

def starts_with_regex(pattern, text):
    regex = re.compile(pattern)
    return bool(regex.match(text))

def process_wildcard_for_segs(wildcard):
    if wildcard.startswith('[LAB]'):
        raw_items = split_to_dict(wildcard)

        items = {}
        for k, v in raw_items.items():
            v = v.strip()
            if v != '':
                items[k] = v

        return 'LAB', WildcardChooserDict(items)

    elif starts_with_regex(r"\[(ASC|DSC|RND)\]", wildcard):
        mode = wildcard[1:4]
        raw_items = wildcard[5:].split('[SEP]')

        items = []
        for x in raw_items:
            x = x.strip()
            if x != '':
                items.append(x)

        if mode == 'RND':
            random.shuffle(items)
            return mode, WildcardChooser(items, True)
        else:
            return mode, WildcardChooser(items, False)

    else:
        return None, WildcardChooser([wildcard], False)

def segs_scale_match(segs, target_shape):
    h = segs[0][0]
    w = segs[0][1]

    th = target_shape[1]
    tw = target_shape[2]

    if (h == th and w == tw) or h == 0 or w == 0:
        return segs

    rh = th / h
    rw = tw / w

    new_segs = []
    for seg in segs[1]:
        cropped_image = seg.cropped_image
        cropped_mask = seg.cropped_mask
        x1, y1, x2, y2 = seg.crop_region
        bx1, by1, bx2, by2 = seg.bbox

        crop_region = int(x1*rw), int(y1*rw), int(x2*rh), int(y2*rh)
        bbox = int(bx1*rw), int(by1*rw), int(bx2*rh), int(by2*rh)
        new_w = crop_region[2] - crop_region[0]
        new_h = crop_region[3] - crop_region[1]

        cropped_mask = torch.from_numpy(cropped_mask)
        cropped_mask = torch.nn.functional.interpolate(cropped_mask.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode='bilinear', align_corners=False)
        cropped_mask = cropped_mask.squeeze(0).squeeze(0).numpy()

        if cropped_image is not None:
            cropped_image = tensor_resize(torch.from_numpy(cropped_image), new_w, new_h)
            cropped_image = cropped_image.numpy()

        new_seg = SEG(cropped_image, cropped_mask, seg.confidence, crop_region, bbox, seg.label, seg.control_net_wrapper)
        new_segs.append(new_seg)

    return ((th, tw), new_segs)

def resolve_lora_name(lora_name_cache, name):
    if os.path.exists(name):
        return name
    else:
        if len(lora_name_cache) == 0:
            lora_name_cache.extend(folder_paths.get_filename_list("loras"))

        for x in lora_name_cache:
            if x.endswith(name):
                return x

def is_numeric_string(input_str):
    return re.match(r'^-?\d+(\.\d+)?$', input_str) is not None


wildcard_lock = threading.Lock()
wildcard_dict = {}
def get_wildcard_dict():
    global wildcard_dict
    with wildcard_lock:
        return wildcard_dict

def wildcard_normalize(x):
    return x.replace("\\", "/").lower()

def process(text, seed=None):
    if seed is not None:
        random.seed(seed)

    def replace_options(string):
        replacements_found = False

        def replace_option(match):
            nonlocal replacements_found
            options = match.group(1).split('|')

            multi_select_pattern = options[0].split('$$')
            select_range = None
            select_sep = ' '
            range_pattern = r'(\d+)(-(\d+))?'
            range_pattern2 = r'-(\d+)'

            if len(multi_select_pattern) > 1:
                r = re.match(range_pattern, options[0])

                if r is None:
                    r = re.match(range_pattern2, options[0])
                    a = '1'
                    b = r.group(1).strip()
                else:
                    a = r.group(1).strip()
                    try:
                        b = r.group(3).strip()
                    except:
                        b = None

                if r is not None:
                    if b is not None and is_numeric_string(a) and is_numeric_string(b):
                        # PATTERN: num1-num2
                        select_range = int(a), int(b)
                    elif is_numeric_string(a):
                        # PATTERN: num
                        x = int(a)
                        select_range = (x, x)

                    if select_range is not None and len(multi_select_pattern) == 2:
                        # PATTERN: count$$
                        options[0] = multi_select_pattern[1]
                    elif select_range is not None and len(multi_select_pattern) == 3:
                        # PATTERN: count$$ sep $$
                        select_sep = multi_select_pattern[1]
                        options[0] = multi_select_pattern[2]

            adjusted_probabilities = []
            total_prob = 0

            for option in options:
                parts = option.split('::', 1)
                if len(parts) == 2 and is_numeric_string(parts[0].strip()):
                    config_value = float(parts[0].strip())
                else:
                    config_value = 1  # Default value if no configuration is provided

                adjusted_probabilities.append(config_value)
                total_prob += config_value

            normalized_probabilities = [prob / total_prob for prob in adjusted_probabilities]

            if select_range is None:
                select_count = 1
            else:
                select_count = random.randint(select_range[0], select_range[1])

            if select_count > len(options):
                selected_items = options
            else:
                selected_items = random.choices(options, weights=normalized_probabilities, k=select_count)
                selected_items = set(selected_items)

                try_count = 0
                while len(selected_items) < select_count and try_count < 10:
                    remaining_count = select_count - len(selected_items)
                    additional_items = random.choices(options, weights=normalized_probabilities, k=remaining_count)
                    selected_items |= set(additional_items)
                    try_count += 1

            selected_items2 = [re.sub(r'^\s*[0-9.]+::', '', x, 1) for x in selected_items]
            replacement = select_sep.join(selected_items2)
            if '::' in replacement:
                pass

            replacements_found = True
            return replacement

        pattern = r'{([^{}]*?)}'
        replaced_string = re.sub(pattern, replace_option, string)

        return replaced_string, replacements_found

    def replace_wildcard(string):
        local_wildcard_dict = get_wildcard_dict()
        pattern = r"__([\w.\-+/*\\]+)__"
        matches = re.findall(pattern, string)

        replacements_found = False

        for match in matches:
            keyword = match.lower()
            keyword = wildcard_normalize(keyword)
            if keyword in local_wildcard_dict:
                replacement = random.choice(local_wildcard_dict[keyword])
                replacements_found = True
                string = string.replace(f"__{match}__", replacement, 1)
            elif '*' in keyword:
                subpattern = keyword.replace('*', '.*').replace('+', '\+')
                total_patterns = []
                found = False
                for k, v in local_wildcard_dict.items():
                    if re.match(subpattern, k) is not None:
                        total_patterns += v
                        found = True

                if found:
                    replacement = random.choice(total_patterns)
                    replacements_found = True
                    string = string.replace(f"__{match}__", replacement, 1)
            elif '/' not in keyword:
                string_fallback = string.replace(f"__{match}__", f"__*/{match}__", 1)
                string, replacements_found = replace_wildcard(string_fallback)

        return string, replacements_found

    replace_depth = 100
    stop_unwrap = False
    while not stop_unwrap and replace_depth > 1:
        replace_depth -= 1  # prevent infinite loop
        pass1, is_replaced1 = replace_options(text)

        while is_replaced1:
            pass1, is_replaced1 = replace_options(pass1)

        text, is_replaced2 = replace_wildcard(pass1)
        stop_unwrap = not is_replaced1 and not is_replaced2

    return text

def safe_float(x):
    if is_numeric_string(x):
        return float(x)
    else:
        return 1.0

def extract_lora_values(string):
    pattern = r'<lora:([^>]+)>'
    matches = re.findall(pattern, string)

    def touch_lbw(text):
        return re.sub(r'LBW=[A-Za-z][A-Za-z0-9_-]*:', r'LBW=', text)

    items = [touch_lbw(match.strip(':')) for match in matches]

    added = set()
    result = []
    for item in items:
        item = item.split(':')

        lora = None
        a = None
        b = None
        lbw = None
        lbw_a = None
        lbw_b = None

        if len(item) > 0:
            lora = item[0]

            for sub_item in item[1:]:
                if is_numeric_string(sub_item):
                    if a is None:
                        a = float(sub_item)
                    elif b is None:
                        b = float(sub_item)
                elif sub_item.startswith("LBW="):
                    for lbw_item in sub_item[4:].split(';'):
                        if lbw_item.startswith("A="):
                            lbw_a = safe_float(lbw_item[2:].strip())
                        elif lbw_item.startswith("B="):
                            lbw_b = safe_float(lbw_item[2:].strip())
                        elif lbw_item.strip() != '':
                            lbw = lbw_item

        if a is None:
            a = 1.0
        if b is None:
            b = a

        if lora is not None and lora not in added:
            result.append((lora, a, b, lbw, lbw_a, lbw_b))
            added.add(lora)

    return result


def remove_lora_tags(string):
    pattern = r'<lora:[^>]+>'
    result = re.sub(pattern, '', string)

    return result

def try_install_custom_node(custom_node_url, msg):
    import sys
    try:
        confirm_try_install = sys.CM_api['cm.try-install-custom-node']
        print(f"confirm_try_install: {confirm_try_install}")
        confirm_try_install('Impact Pack', custom_node_url, msg)
    except Exception as e:
        print(msg)
        print(f"[Impact Pack] ComfyUI-Manager is outdated. The custom node installation feature is not available.")

def process_with_loras(wildcard_opt, model, clip, clip_encoder=None):
    lora_name_cache = []

    pass1 = process(wildcard_opt)
    loras = extract_lora_values(pass1)
    pass2 = remove_lora_tags(pass1)

    for lora_name, model_weight, clip_weight, lbw, lbw_a, lbw_b in loras:
        if (lora_name.split('.')[-1]) not in folder_paths.supported_pt_extensions:
            lora_name = lora_name+".safetensors"

        orig_lora_name = lora_name
        lora_name = resolve_lora_name(lora_name_cache, lora_name)

        if lora_name is not None:
            path = folder_paths.get_full_path("loras", lora_name)
        else:
            path = None

        if path is not None:
            print(f"LOAD LORA: {lora_name}: {model_weight}, {clip_weight}, LBW={lbw}, A={lbw_a}, B={lbw_b}")

            def default_lora():
                return nodes.LoraLoader().load_lora(model, clip, lora_name, model_weight, clip_weight)

            if lbw is not None:
                if 'LoraLoaderBlockWeight //Inspire' not in nodes.NODE_CLASS_MAPPINGS:
                    try_install_custom_node('https://github.com/ltdrdata/ComfyUI-Inspire-Pack', "To use 'LBW=' syntax in wildcards, 'Inspire Pack' extension is required.")
                    print(f"'LBW(Lora Block Weight)' is given, but the 'Inspire Pack' is not installed. The LBW= attribute is being ignored.")
                    model, clip = default_lora()
                else:
                    cls = nodes.NODE_CLASS_MAPPINGS['LoraLoaderBlockWeight //Inspire']
                    model, clip, _ = cls().doit(model, clip, lora_name, model_weight, clip_weight, False, 0, lbw_a, lbw_b, "", lbw)
            else:
                model, clip = default_lora()
        else:
            print(f"LORA NOT FOUND: {orig_lora_name}")

    print(f"CLIP: {pass2}")

    if clip_encoder is None:
        return model, clip, nodes.CLIPTextEncode().encode(clip, pass2)[0]
    else:
        return model, clip, clip_encoder.encode(clip, pass2)[0]

def _tensor_check_image(image):
    if image.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {image.ndim} dimensions")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3 or 4 channels for image, but found {image.shape[-1]} channels")
    return

def pil2tensor(image):
    return torch.from_numpy(np.array(image).astype(np.float32) / 255.0).unsqueeze(0)

def general_tensor_resize(image, w: int, h: int):
    _tensor_check_image(image)
    image = image.permute(0, 3, 1, 2)
    image = torch.nn.functional.interpolate(image, size=(h, w), mode="bilinear")
    image = image.permute(0, 2, 3, 1)
    return image

LANCZOS = (Image.Resampling.LANCZOS if hasattr(Image, 'Resampling') else Image.LANCZOS)
def tensor_resize(image, w: int, h: int):
    _tensor_check_image(image)
    if image.shape[3] >= 3:
        image = tensor2pil(image)
        scaled_image = image.resize((w, h), resample=LANCZOS)
        return pil2tensor(scaled_image)
    else:
        return general_tensor_resize(image, w, h)

def vae_encode_crop_pixels(pixels):
    x = (pixels.shape[1] // 8) * 8
    y = (pixels.shape[2] // 8) * 8
    if pixels.shape[1] != x or pixels.shape[2] != y:
        x_offset = (pixels.shape[1] % 8) // 2
        y_offset = (pixels.shape[2] % 8) // 2
        pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
    return pixels

'''
def vae_encode_crop_pixels_sd(self, pixels):
    x = (pixels.shape[1] // self.downscale_ratio) * self.downscale_ratio
    y = (pixels.shape[2] // self.downscale_ratio) * self.downscale_ratio
    if pixels.shape[1] != x or pixels.shape[2] != y:
        x_offset = (pixels.shape[1] % self.downscale_ratio) // 2
        y_offset = (pixels.shape[2] % self.downscale_ratio) // 2
        pixels = pixels[:, x_offset:x + x_offset, y_offset:y + y_offset, :]
    return pixels
'''

'''
    pixels = nodes.VAEEncode.vae_encode_crop_pixels(pixels)
    t = vae.encode(pixels[:, :, :, :3])
'''

#  itt hibás:
def to_latent_image(pixels, vae):
    x = pixels.shape[1]
    y = pixels.shape[2]
    if pixels.shape[1] != x or pixels.shape[2] != y:
        pixels = pixels[:, :x, :y, :]

    pixels = vae_encode_crop_pixels(pixels)
    t = vae.encode(pixels[:, :, :, :3])
    return {"samples":t}

def ksampler_wrapper(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise,
                     refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None,
                     refiner_negative=None, model_concept="Normal"):

    if refiner_ratio is None or refiner_model is None or refiner_clip is None or refiner_positive is None or refiner_negative is None:
        if model_concept == "Turbo":
            cfg = cfg * 1.5
            sigmas = nodes_custom_sampler.SDTurboScheduler().get_sigmas(model, steps, denoise)
            sampler = comfy.samplers.sampler_object(sampler_name)
            turbo_samples = nodes_custom_sampler.SamplerCustom().sample(model, True, seed, cfg, positive, negative, sampler, sigmas[0], latent_image)
            refined_latent = turbo_samples[0]
        else:
            try:
                refined_latent = nodes.KSampler().sample(model, seed, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)[0]
            except Exception:
                refined_latent = latent_image
    else:
        advanced_steps = math.floor(steps / denoise)
        start_at_step = advanced_steps - steps
        end_at_step = start_at_step + math.floor(steps * (1.0 - refiner_ratio))

        # print(f"pre: {start_at_step} .. {end_at_step} / {advanced_steps}")
        temp_latent = nodes.KSamplerAdvanced().sample(model, "enable", seed, advanced_steps, cfg, sampler_name, scheduler, positive, negative, latent_image, start_at_step, end_at_step, "enable")[0]

        if 'noise_mask' in latent_image:
            latent_compositor = nodes.NODE_CLASS_MAPPINGS['LatentCompositeMasked']()
            temp_latent = latent_compositor.composite(latent_image, temp_latent, 0, 0, False, latent_image['noise_mask'])[0]

        # print(f"post: {end_at_step} .. {advanced_steps + 1} / {advanced_steps}")
        refined_latent = nodes.KSamplerAdvanced().sample(refiner_model, "disable", seed, advanced_steps, cfg, sampler_name, scheduler, refiner_positive, refiner_negative, temp_latent, end_at_step, advanced_steps + 1, "disable")[0]

    return refined_latent

def enhance_detail(image, model, clip, vae, guide_size, guide_size_for_bbox, max_size, bbox, seed, steps, cfg,
                   sampler_name, scheduler, positive, negative, denoise, noise_mask, force_inpaint, segment_settings, multiplier,
                   wildcard_opt=None, wildcard_opt_concat_mode=None, detailer_hook=None,
                   refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None,
                   refiner_negative=None, control_net_wrapper=None, model_concept = "SD1", cycle=1):

    guide_size = max(image.shape[1], image.shape[2]) * multiplier
    if model_concept == 'Turbo':
        guide_size = guide_size * 1.8

    max_size = guide_size * 1.4

    '''
    print('--------------3---------------')
    print('Multiplier: ' + str(multiplier))
    print('Guided size: ' + str(guide_size))
    print('Model concept: ' + model_concept)
    print('Image H: ' + str(image.shape[1]))
    print('Image W: ' + str(image.shape[2]))
    print('Segment area: ' + str(image.shape[1] * image.shape[2]))
    print('Image size / Segment area: ' + str((segment_settings['image_size'][0] * segment_settings['image_size'][1]) / (image.shape[1] * image.shape[2])))
    print(segment_settings)
    print('--------------3---------------')
    '''

    if noise_mask is not None and len(noise_mask.shape) == 3:
        noise_mask = noise_mask.squeeze(0)

    if wildcard_opt is not None and wildcard_opt != "":
        model, _, wildcard_positive = process_with_loras(wildcard_opt, model, clip)

        if wildcard_opt_concat_mode == "concat":
            positive = nodes.ConditioningConcat().concat(positive, wildcard_positive)[0]
        else:
            positive = wildcard_positive

    h = image.shape[1]
    w = image.shape[2]

    bbox_h = bbox[3] - bbox[1]
    bbox_w = bbox[2] - bbox[0]

    if not force_inpaint and bbox_h >= guide_size and bbox_w >= guide_size:
        print(f"Detailer: segment skip (enough big)")
        return None, None

    if guide_size_for_bbox:  # == "bbox"
        upscale = guide_size / min(bbox_w, bbox_h)
    else:
        upscale = guide_size / min(w, h)

    new_w = int(w * upscale)
    new_h = int(h * upscale)

    if 'aitemplate_keep_loaded' in model.model_options:
        max_size = min(4096, max_size)

    if new_w > max_size or new_h > max_size:
        upscale *= max_size / max(new_w, new_h)
        new_w = int(w * upscale)
        new_h = int(h * upscale)

    if not force_inpaint:
        if upscale <= 1.0:
            print(f"Detailer: segment skip [determined upscale factor={upscale}]")
            return None, None

        if new_w == 0 or new_h == 0:
            print(f"Detailer: segment skip [zero size={new_w, new_h}]")
            return None, None
    else:
        if upscale <= 1.0 or new_w == 0 or new_h == 0:
            print(f"Detailer: force inpaint")
            upscale = 1.0
            new_w = w
            new_h = h

    if detailer_hook is not None:
        new_w, new_h = detailer_hook.touch_scaled_size(new_w, new_h)

    # print(f"Detailer: segment upscale for ({bbox_w, bbox_h}) | crop region {w, h} x {upscale} -> {new_w, new_h}")
    upscaled_image = tensor_resize(image, new_w, new_h)
    latent_image = to_latent_image(upscaled_image, vae)

    upscaled_mask = None
    if noise_mask is not None:
        noise_mask = torch.from_numpy(noise_mask)
        upscaled_mask = torch.nn.functional.interpolate(noise_mask.unsqueeze(0).unsqueeze(0), size=(new_h, new_w), mode='bicubic', align_corners=False)
        upscaled_mask = upscaled_mask.squeeze(0).squeeze(0)
        latent_image['noise_mask'] = upscaled_mask

    if detailer_hook is not None:
        latent_image = detailer_hook.post_encode(latent_image)

    cnet_pil = None
    if control_net_wrapper is not None:
        positive, cnet_pil = control_net_wrapper.apply(positive, upscaled_image, upscaled_mask)

    refined_latent = latent_image

    for i in range(0, cycle):
        if detailer_hook is not None:
            if detailer_hook is not None:
                detailer_hook.set_steps((i, cycle))
            refined_latent = detailer_hook.cycle_latent(refined_latent)
            model2, seed2, steps2, cfg2, sampler_name2, scheduler2, positive2, negative2, upscaled_latent2, denoise2 = detailer_hook.pre_ksample(model, seed+i, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise)
        else:
            model2, seed2, steps2, cfg2, sampler_name2, scheduler2, positive2, negative2, upscaled_latent2, denoise2 = model, seed + i, steps, cfg, sampler_name, scheduler, positive, negative, latent_image, denoise
        refined_latent = ksampler_wrapper(model2, seed2, steps2, cfg2, sampler_name2, scheduler2, positive2, negative2, refined_latent, denoise2, refiner_ratio, refiner_model, refiner_clip, refiner_positive, refiner_negative, model_concept)

    if detailer_hook is not None:
        refined_latent = detailer_hook.pre_decode(refined_latent)
    refined_image = vae.decode(refined_latent['samples'])

    if detailer_hook is not None:
        refined_image = detailer_hook.post_decode(refined_image)

    refined_image = tensor_resize(refined_image, w, h)
    refined_image = refined_image.cpu()
    return refined_image, cnet_pil

def to_tensor(image):
    if isinstance(image, Image.Image):
        return torch.from_numpy(np.array(image))
    if isinstance(image, torch.Tensor):
        return image
    if isinstance(image, np.ndarray):
        return torch.from_numpy(image)
    raise ValueError(f"Cannot convert {type(image)} to torch.Tensor")

def _tensor_check_mask(mask):
    if mask.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {mask.ndim} dimensions")
    if mask.shape[-1] != 1:
        raise ValueError(f"Expected 1 channel for mask, but found {mask.shape[-1]} channels")
    return

def tensor_gaussian_blur_mask(mask, kernel_size, sigma=10.0):
    """Return NHWC torch.Tenser from ndim == 2 or 4 `np.ndarray` or `torch.Tensor`"""
    if isinstance(mask, np.ndarray):
        mask = torch.from_numpy(mask)

    if mask.ndim == 2:
        mask = mask[None, ..., None]
    elif mask.ndim == 3:
        mask = mask[..., None]

    _tensor_check_mask(mask)

    if kernel_size <= 0:
        return mask

    prev_device = mask.device
    device = comfy.model_management.get_torch_device()
    mask.to(device)

    # apply gaussian blur
    mask = mask[:, None, ..., 0]
    blurred_mask = torchvision.transforms.GaussianBlur(kernel_size=kernel_size*2+1, sigma=sigma)(mask)
    blurred_mask = blurred_mask[:, 0, ..., None]
    blurred_mask.to(prev_device)

    return blurred_mask

def tensor_paste(image1, image2, left_top, mask):
    _tensor_check_image(image1)
    _tensor_check_image(image2)
    _tensor_check_mask(mask)
    if image2.shape[1:3] != mask.shape[1:3]:
        raise ValueError(f"Inconsistent size: Image ({image2.shape[1:3]}) != Mask ({mask.shape[1:3]})")

    x, y = left_top
    _, h1, w1, _ = image1.shape
    _, h2, w2, _ = image2.shape

    w = min(w1, x + w2) - x
    h = min(h1, y + h2) - y

    if w <= 0 or h <= 0:
        return

    mask = mask[:, :h, :w, :]
    image1[:, y:y+h, x:x+w, :] = ((1 - mask) * image1[:, y:y+h, x:x+w, :] + mask * image2[:, :h, :w, :])
    return

def tensor_convert_rgba(image, prefer_copy=True):
    _tensor_check_image(image)
    n_channel = image.shape[-1]
    if n_channel == 4:
        return image

    if n_channel == 3:
        alpha = torch.ones((*image.shape[:-1], 1))
        return torch.cat((image, alpha), axis=-1)

    if n_channel == 1:
        if prefer_copy:
            image = image.repeat(1, -1, -1, 4)
        else:
            image = image.expand(1, -1, -1, 3)
        return image

    raise ValueError(f"illegal conversion (channels: {n_channel} -> 4)")

def tensor_convert_rgb(image, prefer_copy=True):
    _tensor_check_image(image)
    n_channel = image.shape[-1]
    if n_channel == 3:
        return image

    if n_channel == 4:
        image = image[..., :3]
        if prefer_copy:
            image = image.copy()
        return image

    if n_channel == 1:
        if prefer_copy:
            image = image.repeat(1, -1, -1, 4)
        else:
            image = image.expand(1, -1, -1, 3)
        return image

    raise ValueError(f"illegal conversion (channels: {n_channel} -> 3)")

def tensor_get_size(image):
    _tensor_check_image(image)
    _, h, w, _ = image.shape
    return (w, h)

def tensor_putalpha(image, mask):
    _tensor_check_image(image)
    _tensor_check_mask(mask)
    image[..., -1] = mask[..., 0]

class DetailerForEach:
    def do_detail(image, segs, model, clip, vae, guide_size, guide_size_for_bbox, max_size, seed, steps, cfg,
                  sampler_name, scheduler, positive, negative, denoise, feather, noise_mask, force_inpaint, segment_settings,
                  wildcard_opt=None, detailer_hook=None,
                  refiner_ratio=None, refiner_model=None, refiner_clip=None, refiner_positive=None, refiner_negative=None, model_concept="Normal", cycle=1, use_aesthetic_scorer=False):

        if len(image) > 1:
            raise Exception('[Primere] ERROR: does not allow image batches.')

        image = image.clone()
        enhanced_alpha_list = []
        enhanced_list = []
        cropped_list = []
        cnet_pil_list = []

        segs = segs_scale_match(segs, image.shape)
        new_segs = []

        wildcard_concat_mode = None
        if wildcard_opt is not None:
            if wildcard_opt.startswith('[CONCAT]'):
                wildcard_concat_mode = 'concat'
                wildcard_opt = wildcard_opt[8:]
            wmode, wildcard_chooser = process_wildcard_for_segs(wildcard_opt)
        else:
            wmode, wildcard_chooser = None, None

        if wmode in ['ASC', 'DSC']:
            if wmode == 'ASC':
                ordered_segs = sorted(segs[1], key=lambda x: (x.bbox[0], x.bbox[1]))
            else:
                ordered_segs = sorted(segs[1], key=lambda x: (x.bbox[0], x.bbox[1]), reverse=True)
        else:
            ordered_segs = segs[1]

        for seg in ordered_segs:
            cropped_image = seg.cropped_image if seg.cropped_image is not None else crop_ndarray4(image.numpy(), seg.crop_region)
            cropped_image = to_tensor(cropped_image)
            mask = to_tensor(seg.cropped_mask)
            mask = tensor_gaussian_blur_mask(mask, feather)

            is_mask_all_zeros = (seg.cropped_mask == 0).all().item()
            if is_mask_all_zeros:
                # print(f"Detailer: segment skip [empty mask]")
                continue

            if noise_mask:
                cropped_mask = seg.cropped_mask
            else:
                cropped_mask = None

            if wildcard_chooser is not None:
                wildcard_item = wildcard_chooser.get(seg)
            else:
                wildcard_item = None

            # for dev only!!!!
            # multiplierList = np.arange(0.6, 3.2, 0.2).tolist()
            # for multiplier in multiplierList:

            if 'final_positive' in segment_settings and (segment_settings['detect_age'] == True or segment_settings['detect_gender'] == True or segment_settings['detect_emotion'] == True or segment_settings['detect_race'] == True):
                coordinates = {'age': 'detect_age', 'dominant_gender': 'detect_gender', 'dominant_emotion': 'detect_emotion', 'dominant_race': 'detect_race'}
                refiner_prompt = segment_settings['final_positive']
                if 'final_positive' in segment_settings:
                    is_prompt = re.findall(r"\[(.*?)]", refiner_prompt)
                    if len(is_prompt) > 0:
                        face_analyzed = segment_analyzer([cropped_image])[0]
                        result_keys = list(face_analyzed.keys())
                        common_keys = list(np.intersect1d(is_prompt, result_keys))
                        for substring in common_keys:
                            substring_full = '[' + substring + ']'
                            if substring in face_analyzed and face_analyzed[substring] is not None and segment_settings[coordinates[substring]] == True:
                                if substring == 'dominant_gender':
                                    as_man = round(face_analyzed['gender']['Man'], 0)
                                    as_woman = round(face_analyzed['gender']['Woman'], 0)
                                    gender_diff = abs(as_man - as_woman)
                                    if gender_diff < 30:
                                        face_analyzed['dominant_gender'] = f"feminine:{(as_woman / 100) + 1} mixed masculine:{(as_man / 100) + 1}"
                                    if gender_diff > 80:
                                        if as_man > as_woman:
                                            face_analyzed['dominant_gender'] = 'masculine man'
                                        else:
                                            face_analyzed['dominant_gender'] = 'feminine woman'
                                refiner_prompt = refiner_prompt.replace(substring_full, str(face_analyzed[substring]).lower())
                        refiner_prompt = re.sub("[\[].*?[\]]", "unspecified", refiner_prompt).strip()

                embeddings_final_pos, pooled_pos = advanced_encode(clip, refiner_prompt, segment_settings['token_normalization'], segment_settings['weight_interpretation'], w_max=1.0, apply_to_pooled=True)
                embeddings_final_neg, pooled_neg = advanced_encode(clip, segment_settings['final_negative'], segment_settings['token_normalization'], segment_settings['weight_interpretation'], w_max=1.0, apply_to_pooled=True)

                positive = [[embeddings_final_pos, {"pooled_output": pooled_pos}]]
                negative = [[embeddings_final_neg, {"pooled_output": pooled_neg}]]

            SegmentedRelative = (segment_settings['image_size'][0] * segment_settings['image_size'][1]) / (cropped_image.shape[1] * cropped_image.shape[2])
            multiplier = round((math.sqrt((SegmentedRelative / 7)) / 2) + 1, 2)

            if multiplier < 1:
                multiplier = 1
            if multiplier > 6:
                multiplier = 6

            enhanced_image, cnet_pil = enhance_detail(cropped_image, model, clip, vae, guide_size,
                                                      guide_size_for_bbox, max_size,
                                                      seg.bbox, seed, steps, cfg, sampler_name, scheduler,
                                                      positive, negative, denoise, cropped_mask, force_inpaint, segment_settings, multiplier,
                                                      wildcard_opt=wildcard_item,
                                                      wildcard_opt_concat_mode=wildcard_concat_mode,
                                                      detailer_hook=detailer_hook,
                                                      refiner_ratio=refiner_ratio, refiner_model=refiner_model,
                                                      refiner_clip=refiner_clip, refiner_positive=refiner_positive,
                                                      refiner_negative=refiner_negative,
                                                      control_net_wrapper=seg.control_net_wrapper, model_concept=model_concept, cycle=cycle)

            original_score = 0
            enhanced_score = 0
            asthetic_hysteresis = 50

            if use_aesthetic_scorer == True:
                AE_MODEL_ROOT = os.path.join(folder_paths.models_dir, 'aesthetic')
                AEMODELS_ENCODERS_PATHS = utility.getValidAscorerPaths(AE_MODEL_ROOT)
                if len(AEMODELS_ENCODERS_PATHS) > 0:
                    if 'cafe_style' in AEMODELS_ENCODERS_PATHS and 'cafe_aesthetic' in AEMODELS_ENCODERS_PATHS:
                        ae_model_access = os.path.join(AE_MODEL_ROOT, 'cafe_aesthetic')
                        style_model_access = os.path.join(AE_MODEL_ROOT, 'cafe_style')
                        if os.path.isdir(ae_model_access) == True and os.path.isdir(style_model_access) == True:
                            try:
                                original_score = int(Outputs.PrimereAestheticCKPTScorer.aesthetic_scorer(None, cropped_image, True, False, None, {}, False)['result'][0])
                                enhanced_score = int(Outputs.PrimereAestheticCKPTScorer.aesthetic_scorer(None, enhanced_image, True, False, None, {}, False)['result'][0])
                            except ImportError:
                                use_aesthetic_scorer = False
                        else:
                            use_aesthetic_scorer = False
                    else:
                        use_aesthetic_scorer = False
                else:
                    use_aesthetic_scorer = False

            if cnet_pil is not None:
                cnet_pil_list.append(cnet_pil)

            if not (enhanced_image is None):
                image = image.cpu()

                original_enhanced_image = enhanced_image
                SEGMENT_IMAGE_PATH = os.path.join(PRIMERE_ROOT, 'Nodes')
                if (original_score > (enhanced_score + asthetic_hysteresis)):
                    enhanced_image = cropped_image
                    SEGMENT_BADGE = os.path.join(SEGMENT_IMAGE_PATH, "segment_ignored.jpg")
                else:
                    SEGMENT_BADGE = os.path.join(SEGMENT_IMAGE_PATH, "segment_passed.jpg")

                enhanced_image = enhanced_image.cpu()
                tensor_paste(image, enhanced_image, (seg.crop_region[0], seg.crop_region[1]), mask)

                if use_aesthetic_scorer == True:
                    enhanced_width = original_enhanced_image.shape[2]
                    enhanced_heigth = original_enhanced_image.shape[1]

                    divider = 4
                    new_icon_width = round(enhanced_width / divider)
                    if new_icon_width > enhanced_heigth / 2:
                        new_icon_width = round(enhanced_heigth / (divider - 1))
                    segment_ignored = utility.ImageLoaderFromPath(SEGMENT_BADGE, new_icon_width, new_icon_width)

                    x = enhanced_width - new_icon_width
                    y = 0
                    original_enhanced_image = original_enhanced_image.clone().movedim(-1, 1)
                    enhanced_image = nodes_mask.composite(original_enhanced_image, segment_ignored.movedim(-1, 1), x, y, None, 1, False).movedim(1, -1)

                enhanced_list.append(enhanced_image)

            if not (enhanced_image is None):
                enhanced_image_alpha = tensor_convert_rgba(enhanced_image)
                new_seg_image = enhanced_image.numpy()  # alpha should not be applied to seg_image

                mask = tensor_resize(mask, *tensor_get_size(enhanced_image))
                tensor_putalpha(enhanced_image_alpha, mask)
                enhanced_alpha_list.append(enhanced_image_alpha)
            else:
                new_seg_image = None

            cropped_list.append(cropped_image)

            new_seg = SEG(new_seg_image, seg.cropped_mask, seg.confidence, seg.crop_region, seg.bbox, seg.label, seg.control_net_wrapper)
            new_segs.append(new_seg)

        image_tensor = tensor_convert_rgb(image)
        # cropped_list.sort(key=lambda x: x.shape, reverse=True)
        # enhanced_list.sort(key=lambda x: x.shape, reverse=True)
        # enhanced_alpha_list.sort(key=lambda x: x.shape, reverse=True)

        return image_tensor, cropped_list, enhanced_list, enhanced_alpha_list, cnet_pil_list, (segs[0], new_segs)

def empty_pil_tensor(w=64, h=64):
    return torch.zeros((1, h, w, 3), dtype=torch.float32)

def segmented_images(segs, input_image):
    result_image_list = []

    if len(segs[1]) > 0:
        for seg in segs[1]:
            result_image_batch = None
            def stack_image(image):
                nonlocal result_image_batch
                if isinstance(image, np.ndarray):
                    image = torch.from_numpy(image)

                if result_image_batch is None:
                    result_image_batch = image
                else:
                    result_image_batch = torch.concat((result_image_batch, image), dim=0)

            ref_image = input_image[0].unsqueeze(0)
            cropped_image = crop_image(ref_image, seg.crop_region)
            if isinstance(cropped_image, np.ndarray):
                cropped_image = torch.from_numpy(cropped_image)
                cropped_image = cropped_image.clone()
                stack_image(cropped_image)

            if result_image_batch is not None:
                result_image_list.append(result_image_batch)
    else:
        result_image_list.append(input_image)

    return result_image_list


def segment_analyzer(images):
    analyzed_obj = []
    analyzed = {}
    for image in images:
        if image.shape[1] * image.shape[2] < 100000:
            image = utility.img_resizer(image, image.shape[2] * 2, image.shape[1] * 2, 'bicubic')
        try:
            analyzed = Segments.PrimereFaceAnalyzer.face_analyzer(None, image)[0][0]
        except Exception:
            analyzed['age'] = None
            analyzed['dominant_gender'] = None
            analyzed['dominant_race'] = None
            analyzed['dominant_emotion'] = None
        analyzed_obj.append(analyzed)

    return analyzed_obj

def filter_segs_by_label(segs, label):
    remained_segs = []
    remained_crops = []
    final_segs = []
    final_segs.append(segs[0])
    for segment in segs[1]:
        if segment.label in label:
           remained_segs.append(segment)
           remained_crops.append(segment.crop_region)

    final_segs = final_segs + [remained_segs] + [remained_crops]
    return final_segs

def filter_segs_by_trigger(segs, trigger_high_off, trigger_low_off, crop_factor):
    remained_segs = []
    remained_crops = []
    final_segs = []
    final_segs.append(segs[0])
    for segment in segs[1]:
        image_area = (abs(segment.crop_region[2] - segment.crop_region[0])) * (abs(segment.crop_region[3] - segment.crop_region[1]))
        image_area = int((image_area / (crop_factor ** 2)))
        if ((trigger_high_off == 0) or (image_area <= trigger_high_off and trigger_high_off > 0)) and ((trigger_low_off == 0) or (image_area >= trigger_low_off and trigger_low_off > 0)):
            remained_segs.append(segment)
            remained_crops.append(segment.crop_region)

    final_segs = final_segs + [remained_segs] + [remained_crops]
    return final_segs

def filter_segs_by_percent_trigger(segs, trigger_high_off, trigger_low_off, crop_factor, input_image_area):
    remained_segs = []
    remained_crops = []
    final_segs = []
    final_segs.append(segs[0])
    for segment in segs[1]:
        image_area = (abs(segment.crop_region[2] - segment.crop_region[0])) * (abs(segment.crop_region[3] - segment.crop_region[1]))
        image_area = int((image_area / (crop_factor ** 2)))
        image_area_percent = 100 / (input_image_area / image_area)
        if ((trigger_high_off == 0) or (image_area_percent <= trigger_high_off and trigger_high_off > 0)) and ((trigger_low_off == 0) or (image_area_percent >= trigger_low_off and trigger_low_off > 0)):
            remained_segs.append(segment)
            remained_crops.append(segment.crop_region)

    final_segs = final_segs + [remained_segs] + [remained_crops]
    return final_segs

'''
def get_bert_base_uncased_model_path():
    comfy_bert_model_base = os.path.join(folder_paths.models_dir, 'bert-base-uncased')
    if glob.glob(os.path.join(comfy_bert_model_base, '**/model.safetensors'), recursive=True):
        print('grounding-dino is using models/bert-base-uncased')
        return comfy_bert_model_base
    return 'bert-base-uncased'

def load_groundingdino_model(model_name):
    config_destination = folder_paths.get_full_path('grounding-dino', model_name)
    dino_model_args = local_groundingdino_SLConfig.fromfile(
        get_local_filepath(
            groundingdino_model_list[model_name]["config_url"],
            'grounding-dino'
        ),
    )

    if dino_model_args.text_encoder_type == 'bert-base-uncased':
        dino_model_args.text_encoder_type = get_bert_base_uncased_model_path()

    dino = local_groundingdino_build_model(dino_model_args)
    model_destination = folder_paths.get_full_path('grounding-dino', model_name)
    checkpoint = torch.load(model_destination,)

    dino.load_state_dict(local_groundingdino_clean_state_dict(
        checkpoint['model']), strict=False)
    device = comfy.model_management.get_torch_device()
    dino.to(device=device)
    dino.eval()
    return dino
'''