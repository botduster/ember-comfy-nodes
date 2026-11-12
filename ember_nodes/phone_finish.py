"""Ember Phone Finish: three phone-camera output stages, each off by default.

Stages, in the order a phone's image pipeline applies them:
  1. Phone HDR (local tone mapping): a smooth local-mean luminance map is computed at low
     resolution; each pixel moves toward mid-grey by `hdr_amount` x (0.5 - local mean). Dark
     regions lift, bright regions come down, local detail is kept. Same offset on R, G and B.
  2. Highlight clip: values are divided by `clip_level` and clamped to 1, so everything at or
     above the level blows out to white the way a small sensor clips (0.97 = top 3% clips).
  3. Chroma 4:2:0: RGB -> YCbCr (BT.601 full range), Cb/Cr averaged over 2x2 blocks and
     repeated back, luma untouched, back to RGB. The colour-resolution loss of a phone JPEG.

With every stage off the input tensor is returned as is (same object, bit-identical).
Deterministic: no random draws. Torch only, runs on the image's own device.
"""

import torch
import torch.nn.functional as F

# BT.601 full-range (JFIF) RGB <-> YCbCr
_RGB2YCC = torch.tensor([[0.299, 0.587, 0.114],
                         [-0.168736, -0.331264, 0.5],
                         [0.5, -0.418688, -0.081312]])
_YCC2RGB = torch.tensor([[1.0, 0.0, 1.402],
                         [1.0, -0.344136, -0.714136],
                         [1.0, 1.772, 0.0]])
_LUMA = (0.299, 0.587, 0.114)
_HDR_GRID = 32  # the local-mean map's short side, in cells


def apply_phone_hdr(image, amount):
    """image: [B, H, W, 3] in 0..1."""
    if amount <= 0:
        return image
    b, h, w, _ = image.shape
    luma = (image[..., 0] * _LUMA[0] + image[..., 1] * _LUMA[1] + image[..., 2] * _LUMA[2]).unsqueeze(1)
    scale = _HDR_GRID / max(1, min(h, w))
    low_h, low_w = max(1, round(h * scale)), max(1, round(w * scale))
    local = F.interpolate(luma, size=(low_h, low_w), mode="area")
    local = F.avg_pool2d(F.pad(local, (1, 1, 1, 1), mode="replicate"), 3, stride=1)
    local = F.interpolate(local, size=(h, w), mode="bilinear", align_corners=False)
    offset = (amount * (0.5 - local)).squeeze(1).unsqueeze(-1)
    return torch.clamp(image + offset, 0.0, 1.0)


def apply_highlight_clip(image, level):
    if level >= 1.0:
        return image
    return torch.clamp(image / level, 0.0, 1.0)


def apply_chroma_420(image):
    b, h, w, _ = image.shape
    m = _RGB2YCC.to(image.device, image.dtype)
    ycc = image @ m.T
    chroma = ycc[..., 1:].permute(0, 3, 1, 2)
    pad_h, pad_w = h % 2, w % 2
    if pad_h or pad_w:
        chroma = F.pad(chroma, (0, pad_w, 0, pad_h), mode="replicate")
    chroma = F.avg_pool2d(chroma, 2)
    chroma = chroma.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)[:, :, :h, :w]
    ycc = torch.cat([ycc[..., :1], chroma.permute(0, 2, 3, 1)], dim=-1)
    return torch.clamp(ycc @ _YCC2RGB.to(image.device, image.dtype).T, 0.0, 1.0)


class EmberPhoneFinish:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "phone_hdr": ("BOOLEAN", {"default": False}),
                "hdr_amount": ("FLOAT", {"default": 0.05, "min": 0.0, "max": 0.5, "step": 0.01,
                                         "tooltip": "Pull toward mid-grey by local brightness. 0.05 = subtle."}),
                "highlight_clip": ("BOOLEAN", {"default": False}),
                "clip_level": ("FLOAT", {"default": 0.97, "min": 0.8, "max": 1.0, "step": 0.005,
                                         "tooltip": "Values at or above this blow out to white."}),
                "chroma_420": ("BOOLEAN", {"default": False,
                                           "tooltip": "Halve colour resolution (2x2), as a phone JPEG does."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "Ember/Post-Processing"
    DESCRIPTION = "Phone HDR -> highlight clip -> chroma 4:2:0; each stage switchable, all off by default."

    def execute(self, image, phone_hdr=False, hdr_amount=0.05, highlight_clip=False, clip_level=0.97,
                chroma_420=False):
        out = image
        if phone_hdr:
            out = apply_phone_hdr(out, float(hdr_amount))
        if highlight_clip:
            out = apply_highlight_clip(out, float(clip_level))
        if chroma_420:
            out = apply_chroma_420(out)
        return (out,)


NODE_CLASS_MAPPINGS = {"EmberPhoneFinish": EmberPhoneFinish}
NODE_DISPLAY_NAME_MAPPINGS = {"EmberPhoneFinish": "Ember Phone Finish"}
