"""Ember HD Ultralytics BBox Loader: a YOLO bbox detector with a controllable inference size.

Why it exists: a stock ultralytics predict letterboxes to 640 px on the long side,
where an eye in a full-body 2 MP portrait is a handful of pixels and often missed.
This node exposes `imgsz` (default 1280) and runs detection at that size.

Written directly against the ultralytics API. It does not import, subclass or
copy ComfyUI-Impact-Subpack. It produces the BBOX_DETECTOR interface and the
SEGS structure that Impact-Pack-style detailers consume:

    detect(image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None)
        -> ((H, W), [SEG, ...])
    detect_combined(image, threshold, dilation) -> MASK tensor or None
    setAux(x) -> no-op

SEG fields and their meaning (a namedtuple, compared field by field in tests):
    cropped_image  image[:, y1:y2, x1:x2, :] for the crop region
    cropped_mask   float32 box mask (1 inside the detected box, dilated), cropped
    confidence     np.ndarray shape (1,), float32, as ultralytics reports it
    crop_region    [x1, y1, x2, y2] ints: the box scaled by crop_factor about its
                   centre, then shifted (not clipped) to stay inside the image
    bbox           np.ndarray [x0, y0, x1, y1] float32 from ultralytics (xyxy)
    label          class name
    control_net_wrapper  None

Only bbox models (models/ultralytics/bbox) are supported. Choosing a segm/ model
raises instead of silently behaving differently.

No network use: ultralytics' anonymous usage events (Google Analytics, fired at
the end of every predict when its `sync` setting is on) are switched off
in-process before each predict.
"""

import importlib
import os
from collections import namedtuple

import numpy as np
import torch
from PIL import Image

import folder_paths

SEG = namedtuple("SEG",
                 ["cropped_image", "cropped_mask", "confidence", "crop_region", "bbox", "label",
                  "control_net_wrapper"],
                 defaults=[None])


class EmberNoSegmDetector:
    """Placeholder SEGM_DETECTOR for bbox-only models (carries no detect method)."""


def _register_model_folders():
    base = os.path.join(folder_paths.models_dir, "ultralytics")
    wanted = {
        "ultralytics_bbox": os.path.join(base, "bbox"),
        "ultralytics_segm": os.path.join(base, "segm"),
        "ultralytics": base,
    }
    for name, path in wanted.items():
        if name not in folder_paths.folder_names_and_paths:
            folder_paths.folder_names_and_paths[name] = ([path], set(folder_paths.supported_pt_extensions))


_register_model_folders()


def _telemetry_off():
    try:
        importlib.import_module("ultralytics.utils.events").events.enabled = False
    except Exception:
        pass


def _to_pil(image):
    if image.ndim != 4:
        raise ValueError(f"Expected NHWC tensor, but found {image.ndim} dimensions")
    if image.shape[-1] not in (1, 3, 4):
        raise ValueError(f"Expected 1, 3 or 4 channels for image, but found {image.shape[-1]} channels")
    pixels = np.clip(255. * image.cpu().numpy().squeeze(0), 0, 255).astype(np.uint8)
    return Image.fromarray(pixels)


def _run_yolo(model, picture, confidence, imgsz):
    """Detections as a list of (label, xyxy float32 array, confidence (1,) array)."""
    _telemetry_off()
    result = model(picture, conf=confidence, device="", imgsz=imgsz)[0]
    boxes = result.boxes
    xyxy = boxes.xyxy.cpu().numpy()
    return [(result.names[int(boxes.cls[i].item())], xyxy[i], boxes.conf[i:i + 1].cpu().numpy())
            for i in range(xyxy.shape[0])]


def _box_mask(height, width, xyxy):
    """float32 HxW mask, 1.0 on the box including both integer corner pixels."""
    mask = np.zeros((height, width), dtype=np.float32)
    left, top, right, bottom = (int(v) for v in xyxy)
    mask[top:bottom + 1, left:right + 1] = 1.0
    return mask


def _grow(mask, amount):
    import cv2
    return cv2.dilate(mask, np.ones((amount, amount), np.uint8))


def _span(limit, start, length):
    """Place [start, start+length) inside [0, limit] by shifting, not shrinking."""
    if start < 0:
        return 0, int(min(limit, length))
    if start + length > limit:
        return int(max(0, limit - length)), int(limit)
    return int(start), int(min(limit, start + length))


def _crop_region(width, height, xyxy, crop_factor):
    box_w = xyxy[2] - xyxy[0]
    box_h = xyxy[3] - xyxy[1]
    region_w = box_w * crop_factor
    region_h = box_h * crop_factor
    centre_x = xyxy[0] + box_w / 2
    centre_y = xyxy[1] + box_h / 2
    x1, x2 = _span(width, int(centre_x - region_w / 2), region_w)
    y1, y2 = _span(height, int(centre_y - region_h / 2), region_h)
    return [x1, y1, x2, y2]


class EmberHDBBoxDetector:
    def __init__(self, model, imgsz=1280):
        self.bbox_model = model
        self.imgsz = imgsz

    def _masks(self, image, threshold, dilation):
        picture = _to_pil(image)
        found = []
        for label, xyxy, conf in _run_yolo(self.bbox_model, picture, threshold, self.imgsz):
            mask = _box_mask(picture.size[1], picture.size[0], xyxy)
            if dilation > 0:
                mask = _grow(mask, dilation)
            found.append((label, xyxy, mask, conf))
        return found

    def detect(self, image, threshold, dilation, crop_factor, drop_size=1, detailer_hook=None):
        drop_size = max(drop_size, 1)
        height, width = image.shape[1], image.shape[2]
        items = []
        for label, xyxy, mask, conf in self._masks(image, threshold, dilation):
            if not (xyxy[3] - xyxy[1] > drop_size and xyxy[2] - xyxy[0] > drop_size):
                continue
            region = _crop_region(width, height, xyxy, crop_factor)
            if detailer_hook is not None:
                region = detailer_hook.post_crop_region(width, height, xyxy, region)
            x1, y1, x2, y2 = region
            items.append(SEG(image[:, y1:y2, x1:x2, :], mask[y1:y2, x1:x2], conf, region, xyxy, label, None))

        segs = (height, width), items
        if detailer_hook is not None and hasattr(detailer_hook, "post_detection"):
            segs = detailer_hook.post_detection(segs)
        return segs

    def detect_combined(self, image, threshold, dilation):
        import cv2
        masks = [mask for _, _, mask, _ in self._masks(image, threshold, dilation)]
        if not masks:
            return None
        combined = np.array(masks[0])
        for mask in masks[1:]:
            if mask.shape == combined.shape:
                combined = cv2.bitwise_or(combined, np.array(mask))
        return torch.from_numpy(combined)

    def setAux(self, x):
        pass


class EmberHDBBoxDetectorProvider:
    @classmethod
    def INPUT_TYPES(cls):
        bbox = ["bbox/" + name for name in folder_paths.get_filename_list("ultralytics_bbox")]
        segm = ["segm/" + name for name in folder_paths.get_filename_list("ultralytics_segm")]
        return {
            "required": {
                "model_name": (bbox + segm,),
                "imgsz": ("INT", {
                    "default": 1280, "min": 320, "max": 2048, "step": 32,
                    "tooltip": "YOLO inference size. Higher finds small objects (eyes) better, slower.",
                }),
            }
        }

    RETURN_TYPES = ("BBOX_DETECTOR", "SEGM_DETECTOR")
    FUNCTION = "doit"
    CATEGORY = "Ember/Detailer"
    DESCRIPTION = "Ultralytics bbox detector with an explicit inference size (imgsz) for small features."

    def doit(self, model_name, imgsz=1280):
        if not model_name.startswith("bbox/"):
            raise ValueError(f"[Ember BBox] Only bbox/ models are supported, got '{model_name}'.")
        path = folder_paths.get_full_path("ultralytics", model_name)
        if path is None:
            path = folder_paths.get_full_path("ultralytics_bbox", model_name[len("bbox/"):])
        if path is None:
            raise ValueError(f"[Ember BBox] model file '{model_name}' not found.")

        _telemetry_off()
        from ultralytics import YOLO
        return EmberHDBBoxDetector(YOLO(path), imgsz), EmberNoSegmDetector()


NODE_CLASS_MAPPINGS = {"EmberHDBBoxDetectorProvider": EmberHDBBoxDetectorProvider}
NODE_DISPLAY_NAME_MAPPINGS = {"EmberHDBBoxDetectorProvider": "Ember HD Ultralytics BBox Loader"}
