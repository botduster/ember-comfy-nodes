"""Black out the face in an image so an edit model will regenerate it.

WHAT THIS IS FOR. Face-swapping through an image-edit model (Nano Banana Pro, Seedream)
works far better if you hand the model a hole rather than a face. Given
`[reference photo, target with the face blacked out]` and a prompt, the model renders
the reference person into the gap. Given `[reference photo, untouched target]` it tends
to preserve what is already there and produce a blend.

So: detect the face, paint over it, and let the edit model do the rest.

WHY THERE IS NO SEGMENTATION HERE. A rectangle is enough. The edit model is already
generating a whole coherent image; it does not need a pixel-accurate matte to know where
the face goes, and a hard-edged box actually reads as a clearer instruction than a soft
mask. This runs on CPU in milliseconds and needs no GPU, which is the whole reason the
face step does not belong on a paid GPU pod.

WHY IT PADS. The face box from a detector is jaw-to-forehead. Swapping only that leaves
the original hairline, ears and neck, which is exactly where a swap looks wrong. The
default expands the box by 39% on each side, which takes those in.

⚠️ THE ONE NETWORK CALL IN THIS PACK. The pack's __init__ says nothing here calls out to
a network and that a node needing to is worth arguing about. This one downloads YuNet
(~350KB, from OpenCV's own model zoo) the first time it runs, then caches it next to the
node forever. The argument for it: the alternative is shipping a binary blob in the repo
or making every user hunt for the file. Set `model_path` to a local file and no download
happens at all.
"""

import os
import urllib.request

import numpy as np
import torch

YUNET_URL = (
    "https://github.com/opencv/opencv_zoo/raw/main/models/"
    "face_detection_yunet/face_detection_yunet_2023mar.onnx"
)
YUNET_FILENAME = "face_detection_yunet_2023mar.onnx"

TARGET_LARGEST = "largest"
TARGET_ALL = "all"
TARGET_MOST_CENTRAL = "most_central"
TARGETS = [TARGET_LARGEST, TARGET_ALL, TARGET_MOST_CENTRAL]

ON_NO_FACE_ERROR = "error"
ON_NO_FACE_PASSTHROUGH = "passthrough"
ON_NO_FACE = [ON_NO_FACE_ERROR, ON_NO_FACE_PASSTHROUGH]


def _cache_dir():
    return os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".model_cache")


def ensure_yunet(model_path: str = "") -> str:
    """Return a path to the YuNet weights, downloading once if needed."""
    if model_path:
        if not os.path.exists(model_path):
            raise RuntimeError(
                "[Ember Face Mask] model_path was set but does not exist: %s" % model_path
            )
        return model_path

    d = _cache_dir()
    os.makedirs(d, exist_ok=True)
    path = os.path.join(d, YUNET_FILENAME)
    if not os.path.exists(path):
        print("[Ember Face Mask] Fetching YuNet weights once -> %s" % path)
        urllib.request.urlretrieve(YUNET_URL, path)
    return path


def detect_faces(img_bgr, model_path: str, score_threshold: float, min_side_fraction: float):
    """(x, y, w, h) per face, dropping anything smaller than min_side_fraction of the short side.

    The size floor is what keeps a face in a poster on the wall, or a reflection, from
    being treated as the subject.
    """
    import cv2

    h, w = img_bgr.shape[:2]
    detector = cv2.FaceDetectorYN.create(
        model_path, "", (w, h),
        score_threshold=float(score_threshold), nms_threshold=0.3, top_k=5000,
    )
    _, detections = detector.detect(img_bgr)
    if detections is None:
        return []

    floor = min(w, h) * float(min_side_fraction)
    out = []
    for det in detections:
        x, y, fw, fh = int(det[0]), int(det[1]), int(det[2]), int(det[3])
        if fw < floor or fh < floor:
            continue
        out.append((x, y, fw, fh))
    return out


def select_faces(faces, target: str, image_w: int, image_h: int):
    if not faces or target == TARGET_ALL:
        return faces
    if target == TARGET_LARGEST:
        return [max(faces, key=lambda f: f[2] * f[3])]
    # most_central — by distance from the frame centre to the face centre.
    cx, cy = image_w / 2.0, image_h / 2.0
    def dist(f):
        fx, fy, fw, fh = f
        return ((fx + fw / 2.0) - cx) ** 2 + ((fy + fh / 2.0) - cy) ** 2
    return [min(faces, key=dist)]


def padded_box(face, padding: float, image_w: int, image_h: int):
    x, y, w, h = face
    px, py = int(w * padding), int(h * padding)
    return (
        max(0, x - px),
        max(0, y - py),
        min(image_w, x + w + px),
        min(image_h, y + h + py),
    )


class FaceMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "target": (TARGETS, {
                    "default": TARGET_LARGEST,
                    "tooltip": "Which face to blank. 'largest' suits a single subject; 'all' "
                               "will also blank bystanders, which is rarely what a creator swap "
                               "wants."}),
                "padding": ("FLOAT", {
                    "default": 0.39, "min": 0.0, "max": 1.5, "step": 0.01,
                    "tooltip": "Grows the detected box by this fraction on each side. Too little "
                               "leaves the original hairline and jaw, which is where a swap "
                               "usually looks wrong."}),
                "score_threshold": ("FLOAT", {
                    "default": 0.7, "min": 0.1, "max": 1.0, "step": 0.05,
                    "tooltip": "Detector confidence. Lower it if a turned or shadowed face is "
                               "being missed."}),
                "min_side_fraction": ("FLOAT", {
                    "default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01,
                    "tooltip": "Ignore faces smaller than this fraction of the short edge — "
                               "posters, reflections, people in the background."}),
                "on_no_face": (ON_NO_FACE, {
                    "default": ON_NO_FACE_ERROR,
                    "tooltip": "Default is to STOP. Passing an unmasked image on to a swap "
                               "silently produces a bad result that looks like a model failure."}),
            },
            "optional": {
                "model_path": ("STRING", {
                    "default": "", "multiline": False,
                    "tooltip": "Local YuNet .onnx. Leave empty to fetch and cache it once."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT")
    RETURN_NAMES = ("image", "mask", "faces_found")
    FUNCTION = "run"
    CATEGORY = "Ember"

    def run(self, image, target, padding, score_threshold, min_side_fraction,
            on_no_face, model_path=""):
        try:
            import cv2  # noqa: F401
        except ImportError as exc:
            raise RuntimeError(
                "[Ember Face Mask] OpenCV is required. `pip install opencv-python`."
            ) from exc

        weights = ensure_yunet(model_path)

        arr = image[0].detach().cpu().numpy()
        img_rgb = np.clip(arr * 255.0, 0, 255).astype(np.uint8)
        img_bgr = img_rgb[:, :, ::-1].copy()
        h, w = img_bgr.shape[:2]

        faces = detect_faces(img_bgr, weights, score_threshold, min_side_fraction)
        faces = select_faces(faces, target, w, h)

        if not faces:
            if on_no_face == ON_NO_FACE_ERROR:
                raise RuntimeError(
                    "[Ember Face Mask] No face found.\n"
                    "-> Lower score_threshold, lower min_side_fraction, or check the frame "
                    "actually shows a face. Set on_no_face=passthrough to continue anyway."
                )
            print("[Ember Face Mask] No face found — passing the image through unmasked.")
            empty = torch.zeros((1, h, w), dtype=torch.float32)
            return (image, empty, 0)

        out = img_rgb.copy()
        mask = np.zeros((h, w), dtype=np.float32)
        for face in faces:
            x1, y1, x2, y2 = padded_box(face, padding, w, h)
            out[y1:y2, x1:x2, :] = 0
            mask[y1:y2, x1:x2] = 1.0
            print("[Ember Face Mask] Blanked %dx%d at (%d, %d)" % (x2 - x1, y2 - y1, x1, y1))

        img_t = torch.from_numpy(out.astype(np.float32) / 255.0)[None, ...]
        mask_t = torch.from_numpy(mask)[None, ...]
        return (img_t, mask_t, len(faces))


NODE_CLASS_MAPPINGS = {"EmberFaceMask": FaceMask}
NODE_DISPLAY_NAME_MAPPINGS = {"EmberFaceMask": "Ember Face Mask"}
