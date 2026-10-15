"""Ember Skin Tone: pull skin hue back onto the skin-tone line without touching brightness.

Clean-room implementation, written 2026-10-15 from Ember's own spec (README "Ember Skin Tone"). It uses
only public colour science: the NTSC YIQ transform, in which human skin of every complexion sits close
to the +I axis (the "skin-tone line" on a vectorscope). Krea and refine passes often push skin toward
green or yellow-green; this rotates it back.

Per image:
  1. RGB -> YIQ. Chroma is the (I, Q) vector: hue h = atan2(Q, I), chroma c = |(I, Q)|.
  2. Target hue t = 0 deg (the +I axis) + target_offset_deg. Angular error d = wrap(h - t).
  3. Skin weight, in two passes, so detection follows the cast instead of fading out exactly
     where the cast is worst:
       pass 1  w1 = hue window around the TARGET * chroma window * luma window;
               m1 = w1-weighted circular mean of d (where this image's skin actually sits)
       pass 2  w  = hue window around m1 * chroma window * luma window, softened with a Gaussian
       hue window    1 within half of detection_width_deg, cosine falloff to 0 at the full width
       chroma window excludes near-grey pixels (c < ~0.02) and neon colours (c > ~0.45)
       luma window   excludes near-black and clipped highlights
     An optional `mask` multiplies w (it restricts, it never adds). Like any colour-based skin
     detector, skin-coloured surroundings (wood, beige walls) are caught too; pass a mask to exclude them.
  4. Weighted circular mean error m of the skin (with w), and weighted mean chroma cm.
  5. New error  d' = d - strength * w * (m + uniformity * (d - m))
       uniformity 0: remove the shared cast only, keep natural variation between pixels
       uniformity 1: every skin pixel lands on the target hue
     New chroma c' = c + saturation_fix * strength * w * (cm - c)   (evens out blotchy saturation)
  6. Only I and Q change. The RGB change is added as a delta, so pixels with w = 0 come back
     bit-identical, and Y (luma) is unchanged up to float rounding (and the final clamp to 0..1).

Outputs: `image` and `skin_mask` (w, the weight that was actually applied, before strength).
strength 0 returns the input tensor values exactly. Deterministic: no randomness anywhere.
"""

import math

import torch
import torch.nn.functional as F

# NTSC 1953 YIQ (FCC). Rows: Y, I, Q.
_RGB_TO_YIQ = torch.tensor([
    [0.299, 0.587, 0.114],
    [0.595716, -0.274453, -0.321263],
    [0.211456, -0.522591, 0.311135],
], dtype=torch.float64)
_YIQ_TO_RGB = torch.linalg.inv(_RGB_TO_YIQ)

CHROMA_LOW = (0.015, 0.04)     # smoothstep in: below = grey, not skin
CHROMA_HIGH = (0.35, 0.45)     # smoothstep out: above = saturated paint/neon, not skin
LUMA_LOW = (0.04, 0.12)
LUMA_HIGH = (0.94, 0.995)
BLUR_FRACTION = 0.004          # mask softening sigma, as a fraction of the shorter side
EPS = 1e-6


def _smoothstep(edge0, edge1, x):
    t = ((x - edge0) / (edge1 - edge0)).clamp(0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _wrap(angle):
    return torch.atan2(torch.sin(angle), torch.cos(angle))


def _gaussian_blur(mask, sigma):
    """mask: (B, H, W). Separable Gaussian, reflect padding."""
    if sigma <= 0:
        return mask
    radius = max(1, int(math.ceil(3.0 * sigma)))
    x = torch.arange(-radius, radius + 1, dtype=mask.dtype, device=mask.device)
    kernel = torch.exp(-(x * x) / (2.0 * sigma * sigma))
    kernel = kernel / kernel.sum()
    m = mask.unsqueeze(1)
    h, w = m.shape[-2:]
    pad_x, pad_y = min(radius, w - 1), min(radius, h - 1)
    kx = kernel[radius - pad_x: radius + pad_x + 1].view(1, 1, 1, -1)
    ky = kernel[radius - pad_y: radius + pad_y + 1].view(1, 1, -1, 1)
    kx, ky = kx / kx.sum(), ky / ky.sum()
    if pad_x:
        m = F.conv2d(F.pad(m, (pad_x, pad_x, 0, 0), mode="reflect"), kx)
    if pad_y:
        m = F.conv2d(F.pad(m, (0, 0, pad_y, pad_y), mode="reflect"), ky)
    return m.squeeze(1)


def hue_window(err, width_rad):
    """1 for |err| <= width/2, cosine falloff to 0 at |err| = width."""
    half = 0.5 * width_rad
    t = ((err.abs() - half) / half).clamp(0.0, 1.0)
    return 0.5 * (1.0 + torch.cos(math.pi * t))


def _circular_mean(err, weight):
    return torch.atan2((weight * torch.sin(err)).sum(dim=(1, 2), keepdim=True),
                       (weight * torch.cos(err)).sum(dim=(1, 2), keepdim=True))


def skin_weight(yiq, target_rad, width_rad):
    """yiq: (B, H, W, 3) float64. Returns the unblurred two-pass skin weight (B, H, W)."""
    y, i, q = yiq[..., 0], yiq[..., 1], yiq[..., 2]
    chroma = torch.sqrt(i * i + q * q)
    err = _wrap(torch.atan2(q, i) - target_rad)
    region = _smoothstep(*CHROMA_LOW, chroma) * (1.0 - _smoothstep(*CHROMA_HIGH, chroma)) \
        * _smoothstep(*LUMA_LOW, y) * (1.0 - _smoothstep(*LUMA_HIGH, y))
    first = hue_window(err, width_rad) * region
    centre = _circular_mean(err, first)
    second = hue_window(_wrap(err - centre), width_rad) * region
    found = first.sum(dim=(1, 2), keepdim=True) > 0
    return torch.where(found, second, torch.zeros_like(second))


def apply_skin_tone(image, strength=0.8, target_offset_deg=0.0, uniformity=0.25, saturation_fix=0.4,
                    detection_width_deg=35.0, mask=None):
    """image: (B, H, W, C>=3) in 0..1. Returns (image, skin_mask (B, H, W))."""
    if image.dim() != 4 or image.shape[-1] < 3:
        raise ValueError(f"expected an IMAGE tensor (B, H, W, 3), got shape {tuple(image.shape)}")
    b, h, w = image.shape[:3]
    rgb = image[..., :3].to(torch.float64)
    yiq = rgb @ _RGB_TO_YIQ.to(rgb.device).T
    target = math.radians(float(target_offset_deg))
    width = math.radians(float(detection_width_deg))

    weight = skin_weight(yiq, target, width)
    weight = _gaussian_blur(weight, BLUR_FRACTION * min(h, w))
    if mask is not None:
        m = mask.to(weight.device, torch.float64)
        if m.dim() == 2:
            m = m.unsqueeze(0)
        if m.shape[-2:] != (h, w):
            m = F.interpolate(m.unsqueeze(1), size=(h, w), mode="bilinear", align_corners=False).squeeze(1)
        if m.shape[0] not in (1, b):
            raise ValueError(f"mask batch {m.shape[0]} does not match image batch {b}")
        weight = weight * m.clamp(0.0, 1.0)
    weight = weight.clamp(0.0, 1.0)

    if strength == 0 or float(weight.max()) <= 0.0:
        return image.clone(), weight.to(image.dtype)

    i, q = yiq[..., 1], yiq[..., 2]
    chroma = torch.sqrt(i * i + q * q)
    err = _wrap(torch.atan2(q, i) - target)

    total = weight.sum(dim=(1, 2), keepdim=True).clamp_min(EPS)
    mean_err = _circular_mean(err, weight)
    mean_chroma = (weight * chroma).sum(dim=(1, 2), keepdim=True) / total

    k = float(strength) * weight
    new_err = err - k * (mean_err + float(uniformity) * _wrap(err - mean_err))
    new_chroma = (chroma + float(saturation_fix) * k * (mean_chroma - chroma)).clamp_min(0.0)
    new_i = new_chroma * torch.cos(new_err + target)
    new_q = new_chroma * torch.sin(new_err + target)

    d_iq = torch.stack([torch.zeros_like(i), new_i - i, new_q - q], dim=-1)
    d_iq = torch.where(weight.unsqueeze(-1) > 0, d_iq, torch.zeros_like(d_iq))
    d_rgb = d_iq @ _YIQ_TO_RGB.to(rgb.device).T
    out = image.clone()
    out[..., :3] = (rgb + d_rgb).clamp(0.0, 1.0).to(image.dtype)
    # untouched pixels: copy the original values back exactly (float64 round trip is not exact)
    keep = (weight <= 0).unsqueeze(-1).expand_as(out[..., :3])
    out[..., :3] = torch.where(keep, image[..., :3], out[..., :3])
    return out, weight.to(image.dtype)


class EmberSkinTone:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "strength": ("FLOAT", {"default": 0.8, "min": 0.0, "max": 1.5, "step": 0.01}),
                "target_offset_deg": ("FLOAT", {"default": 0.0, "min": -20.0, "max": 20.0, "step": 0.5,
                                                "tooltip": "Rotate the target off the skin-tone line (+ = toward red/magenta)."}),
                "uniformity": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.01,
                                         "tooltip": "0 = remove the cast only; 1 = every skin pixel on the target hue."}),
                "saturation_fix": ("FLOAT", {"default": 0.4, "min": 0.0, "max": 1.0, "step": 0.01,
                                             "tooltip": "Pull skin saturation toward its own average."}),
                "detection_width_deg": ("FLOAT", {"default": 35.0, "min": 15.0, "max": 60.0, "step": 1.0,
                                                  "tooltip": "How far from the target hue still counts as skin."}),
            },
            "optional": {
                "mask": ("MASK", {"tooltip": "Optional: limit the effect to this region (multiplies the skin weight)."}),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK")
    RETURN_NAMES = ("image", "skin_mask")
    FUNCTION = "apply"
    CATEGORY = "Ember/Post-Processing"
    DESCRIPTION = "Rotates skin hue onto the skin-tone line (YIQ +I axis). Brightness is untouched."

    def apply(self, image, strength, target_offset_deg, uniformity, saturation_fix, detection_width_deg, mask=None):
        return apply_skin_tone(image, strength, target_offset_deg, uniformity, saturation_fix,
                               detection_width_deg, mask)


NODE_CLASS_MAPPINGS = {"EmberSkinTone": EmberSkinTone}
NODE_DISPLAY_NAME_MAPPINGS = {"EmberSkinTone": "Ember Skin Tone"}
