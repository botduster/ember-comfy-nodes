"""Ember Resolution (MP) — width and height from a megapixel budget and an aspect ratio.

WHY THIS IS NOT ONE LINE. Video models want dimensions that are a multiple of 32 —
MiniMax H3 states it, and its own reference 1344x768 obeys it. You cannot simply
scale a ratio to hit a pixel count and round: snapping afterwards moves BOTH the
megapixel total and the aspect ratio, and a single rounding pass can land several
percent from the target. A widget reading 0.98 MP while the model receives 1.04 is
worse than no widget at all.

So this searches the valid grid instead of rounding once, and prints what it actually
produced rather than what was asked for.

Thinking in megapixels also makes cost predictable across aspect ratios: 1.0 MP costs
about the same whether it is 1024x1024 or 768x1344.

Pure arithmetic, reimplemented from the constraint rather than from any third-party
pack's source.
"""

import math

ASPECT_PRESETS = {
    "1:1": (1, 1),
    "4:3": (4, 3),
    "3:4": (3, 4),
    "3:2": (3, 2),
    "2:3": (2, 3),
    "16:9": (16, 9),
    "9:16": (9, 16),
    "21:9": (21, 9),
}
FROM_IMAGE = "from image (input)"
ASPECT_CHOICES = list(ASPECT_PRESETS.keys()) + [FROM_IMAGE]


def nearest_ratio_label(ar: float) -> str:
    """Closest preset label to a measured ratio.

    Compared in LOG space. In raw values 21:9 (2.33) and 16:9 (1.78) sit 0.55 apart
    while 3:4 (0.75) and 2:3 (0.67) sit 0.08 apart, so a linear nearest-match is
    biased toward portrait ratios. Log distance treats a given proportional error
    the same at both ends.
    """
    target = math.log(ar)
    return min(ASPECT_PRESETS.items(), key=lambda kv: abs(math.log(kv[1][0] / kv[1][1]) - target))[0]


def snap_to_budget(megapixels: float, ar: float, multiple_of: int):
    """Width/height closest to the budget at ratio `ar`, both on the multiple grid."""
    mp = max(0.01, float(megapixels))
    m = max(1, int(multiple_of))
    target_px = mp * 1_000_000.0

    # Ideal continuous solution, then snap each side and keep the best of the four
    # neighbours. Snapping both independently can miss the closest legal pair —
    # rounding one down and the other up is often nearer the budget than rounding
    # both the same way.
    ideal_w = math.sqrt(target_px * ar)
    ideal_h = ideal_w / ar

    def grid(v):
        low = max(m, int(math.floor(v / m)) * m)
        return {low, low + m}

    best = None
    for w in grid(ideal_w):
        for h in grid(ideal_h):
            err = abs(w * h - target_px)
            # Tie-break on ratio fidelity: two pairs equally close on pixels are not
            # equally good if one distorts the frame.
            ratio_err = abs(math.log((w / h) / ar))
            key = (err, ratio_err)
            if best is None or key < best[0]:
                best = (key, w, h)
    return best[1], best[2]


class ResolutionMP:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "megapixels": ("FLOAT", {"default": 1.03, "min": 0.05, "max": 16.0, "step": 0.01,
                                         "tooltip": "Pixel budget. Cost scales with this, not with "
                                                    "the aspect ratio."}),
                "aspect_ratio": (ASPECT_CHOICES, {"default": "9:16"}),
                "multiple_of": ("INT", {"default": 32, "min": 1, "max": 256, "step": 1,
                                        "tooltip": "8 for most image models, 32 for video paths."}),
            },
            "optional": {
                "image": ("IMAGE", {"tooltip": "Only used when aspect_ratio is 'from image (input)'."}),
            },
        }

    RETURN_TYPES = ("INT", "INT", "STRING")
    RETURN_NAMES = ("width", "height", "aspect_ratio")
    FUNCTION = "compute"
    CATEGORY = "Ember/Utils"
    DESCRIPTION = "Turn a megapixel budget and aspect ratio into snapped width/height."

    def compute(self, megapixels, aspect_ratio, multiple_of, image=None):
        if aspect_ratio == FROM_IMAGE:
            if image is None:
                raise RuntimeError(
                    "[Ember Resolution] 'from image (input)' is selected but no image is "
                    "connected.\n-> Connect an image, or pick a fixed ratio."
                )
            # ComfyUI IMAGE tensors are [B, H, W, C].
            src_h, src_w = int(image.shape[1]), int(image.shape[2])
            if src_h <= 0 or src_w <= 0:
                raise RuntimeError("[Ember Resolution] Invalid image dimensions: %dx%d" % (src_w, src_h))
            ar = src_w / src_h
            label = nearest_ratio_label(ar)
        else:
            w_r, h_r = ASPECT_PRESETS[aspect_ratio]
            ar = w_r / h_r
            label = aspect_ratio

        width, height = snap_to_budget(float(megapixels), ar, int(multiple_of))
        actual_mp = (width * height) / 1_000_000.0
        drift = (actual_mp - float(megapixels)) / float(megapixels) * 100.0
        print("[Ember Resolution] %dx%d = %.3f MP (asked %.3f, %+.1f%%) ratio %s"
              % (width, height, actual_mp, float(megapixels), drift, label))
        return (width, height, label)


NODE_CLASS_MAPPINGS = {"EmberResolutionMP": ResolutionMP}
NODE_DISPLAY_NAME_MAPPINGS = {"EmberResolutionMP": "Ember Resolution (MP)"}
