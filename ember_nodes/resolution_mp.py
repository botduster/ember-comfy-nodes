"""Ember Resolution (MP): width and height from a megapixel budget and an aspect ratio.

Behavioural contract: the same (width, height, aspect_ratio) as the reference
"Resolution (MP)" node for every input. tests/ember_nodes proves it over the whole
widget range, so any change here has to keep that test at zero mismatches.

The method
----------
Snapping a ratio-scaled size onto a grid (multiple_of) moves both the pixel count
and the ratio. So instead of rounding once, this looks at a 7x7 neighbourhood
around the ideal cell and scores every candidate:

    score = relative megapixel error + 0.25 * relative ratio error

The megapixel budget wins because it drives VRAM use and how far the model sits
from its training resolution. A fraction of a percent of ratio drift is not
visible. Ties go to the first candidate visited (strict <), in the order
height -3..+3, then width -3..+3.

Rounding uses Python's round(), which rounds half to even, applied to v / multiple.
Both sides are clamped to 64..8192.
"""

import math

FROM_IMAGE = "from image (input)"

# Order matters twice: it is the combo order shown in the UI, and the tie-break order
# for nearest_ratio_label (first best wins).
ASPECT_PRESETS = (
    (FROM_IMAGE, None),
    ("1:1", (1, 1)),
    ("4:3", (4, 3)),
    ("3:4", (3, 4)),
    ("3:2", (3, 2)),
    ("2:3", (2, 3)),
    ("16:9", (16, 9)),
    ("9:16", (9, 16)),
    ("21:9", (21, 9)),
    ("5:4", (5, 4)),
    ("4:5", (4, 5)),
)
_PRESET_MAP = dict(ASPECT_PRESETS)

MIN_SIDE = 64
MAX_SIDE = 8192
RATIO_WEIGHT = 0.25
SEARCH_RADIUS = 3


def nearest_ratio_label(ar):
    """The named ratio closest to `ar`, measured in log space.

    Log space because ratios are multiplicative. In raw values 21:9 and 16:9 are
    0.55 apart while 3:4 and 2:3 are only 0.08 apart, which is the same kind of gap.
    """
    best_label, best_err = None, None
    for label, pair in ASPECT_PRESETS:
        if pair is None:
            continue
        err = abs(math.log((pair[0] / pair[1]) / ar))
        if best_err is None or err < best_err:
            best_label, best_err = label, err
    return best_label


def _grid(value, multiple):
    return max(multiple, int(round(value / multiple)) * multiple)


def snap_to_budget(megapixels, ar, multiple):
    """Best (width, height) on the `multiple` grid for a megapixel budget at ratio `ar`."""
    target_px = max(1.0, megapixels * 1_000_000.0)
    centre_h = _grid(math.sqrt(target_px / ar), multiple)

    best = None  # (score, w, h)
    for dh in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
        h = centre_h + dh * multiple
        if h < MIN_SIDE or h > MAX_SIDE:
            continue
        centre_w = _grid(h * ar, multiple)
        for dw in range(-SEARCH_RADIUS, SEARCH_RADIUS + 1):
            w = centre_w + dw * multiple
            if w < MIN_SIDE or w > MAX_SIDE:
                continue
            mp_err = abs(w * h - target_px) / target_px
            ratio_err = abs((w / h) - ar) / ar
            score = mp_err + RATIO_WEIGHT * ratio_err
            if best is None or score < best[0]:
                best = (score, w, h)

    if best is None:
        # A ratio so extreme that no cell fits inside the side limits: clamp instead.
        h = max(MIN_SIDE, min(MAX_SIDE, _grid(math.sqrt(target_px / ar), multiple)))
        w = max(MIN_SIDE, min(MAX_SIDE, _grid(h * ar, multiple)))
        return w, h
    return best[1], best[2]


class EmberResolutionMP:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "megapixels": ("FLOAT", {
                    "default": 0.98, "min": 0.05, "max": 8.0, "step": 0.01, "round": 0.01,
                    "tooltip": "Pixel budget in megapixels (two decimals). Cost and VRAM follow "
                               "this number, not the aspect ratio.",
                }),
                "aspect_ratio": ([label for label, _ in ASPECT_PRESETS], {
                    "default": "16:9",
                    "tooltip": "Target ratio. 'from image (input)' reads it from the connected image.",
                }),
                "multiple_of": ("INT", {
                    "default": 32, "min": 1, "max": 128, "step": 1,
                    "tooltip": "Both sides are snapped to a multiple of this (32 for most video "
                               "models).",
                }),
            },
            "optional": {
                "image": ("IMAGE", {
                    "tooltip": "Only read when aspect_ratio is 'from image (input)'. It is not "
                               "resized or passed through.",
                }),
            },
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "aspect_ratio")
    FUNCTION = "compute"
    CATEGORY = "Ember/Utils"
    DESCRIPTION = "Megapixel budget + aspect ratio -> width/height snapped to a multiple."

    def compute(self, megapixels, aspect_ratio, multiple_of, image=None):
        pair = _PRESET_MAP.get(aspect_ratio)
        if pair is None:
            if image is None:
                raise RuntimeError(
                    "[Ember Resolution] 'from image (input)' is selected but no image is "
                    "connected.\n-> Connect an image, or pick a fixed ratio."
                )
            src_h, src_w = int(image.shape[1]), int(image.shape[2])  # IMAGE is [B, H, W, C]
            if src_h <= 0 or src_w <= 0:
                raise RuntimeError(f"[Ember Resolution] Invalid image dimensions: {src_w}x{src_h}")
            ar = src_w / src_h
            source = f"{src_w}x{src_h}"
            label = nearest_ratio_label(ar)
        else:
            ar = pair[0] / pair[1]
            source = aspect_ratio
            label = aspect_ratio

        width, height = snap_to_budget(float(megapixels), ar, int(multiple_of))
        actual = (width * height) / 1_000_000.0
        drift = (actual - megapixels) / megapixels * 100.0
        print(f"[Ember Resolution] {source} -> {width}x{height} | {actual:.3f} MP "
              f"({drift:+.1f}% vs {megapixels:.2f}) | ratio {width / height:.4f} "
              f"(target {ar:.4f}) | /{multiple_of} | aspect_ratio out: {label}")
        return (width, height, label)


NODE_CLASS_MAPPINGS = {"EmberResolutionMP": EmberResolutionMP}
NODE_DISPLAY_NAME_MAPPINGS = {"EmberResolutionMP": "Ember Resolution (MP)"}
