"""Ember Clarity: midtone local contrast ("clarity"), edge-aware, colour-preserving.

Clean-room implementation, written 2026-10-15 from Ember's own spec (README "Ember Clarity"). It uses
only public methods: Rec.709 luma and the guided filter (He, Sun & Tang, "Guided Image Filtering",
ECCV 2010) with the image as its own guide, which smooths texture but keeps strong edges, so boosting
the difference adds punch without halos.

Per image:
  1. L = Rec.709 luma of RGB.
  2. base = guided_filter(L, guide=L, radius, eps=edge_preserve), with
       radius = max(1, floor(radius_pct / 100 * shorter side + 0.5)) pixels  (box windows, edge-clamped)
     A larger eps smooths across more edges (more halo-prone, punchier); a smaller one keeps edges.
  3. detail = L - base;  midtone weight = (4 * L * (1 - L)) ** midtone_focus
       (1 at mid-grey, 0 at pure black and white; midtone_focus 0 = equal weight everywhere)
  4. L' = clamp(L + amount * detail * weight, 0, 1)
  5. Colour is kept by scaling RGB by L'/L (where L is above a small floor; below it, the luma
     change is added to all three channels), then clamped to 0..1.
amount 0 returns the input values exactly. amount < 0 softens (negative clarity). Deterministic.
"""

import math

import torch
import torch.nn.functional as F

_REC709 = (0.2126, 0.7152, 0.0722)
LUMA_FLOOR = 1e-3


def _box_mean(x, r):
    """Mean over a (2r+1)^2 window, the window clipped at the image border. x: (B, 1, H, W)."""
    h, w = x.shape[-2:]
    ii = F.pad(x, (1, 0, 1, 0)).cumsum(-1).cumsum(-2)  # integral image with a zero row/col
    ys = torch.arange(h, device=x.device)
    xs = torch.arange(w, device=x.device)
    y0, y1 = (ys - r).clamp(0, h), (ys + r + 1).clamp(0, h)
    x0, x1 = (xs - r).clamp(0, w), (xs + r + 1).clamp(0, w)
    s = (ii[..., y1[:, None], x1[None, :]] - ii[..., y0[:, None], x1[None, :]]
         - ii[..., y1[:, None], x0[None, :]] + ii[..., y0[:, None], x0[None, :]])
    area = ((y1 - y0)[:, None] * (x1 - x0)[None, :]).to(x.dtype)
    return s / area


def guided_filter_self(p, r, eps):
    """Guided filter of p with itself as the guide (He et al. 2010, Algorithm 1 with I = p)."""
    mean_p = _box_mean(p, r)
    var_p = _box_mean(p * p, r) - mean_p * mean_p
    a = var_p / (var_p + eps)
    b = mean_p - a * mean_p
    return _box_mean(a, r) * p + _box_mean(b, r)


def apply_clarity(image, amount=0.35, radius_pct=2.5, edge_preserve=0.01, midtone_focus=1.0):
    """image: (B, H, W, C>=3) in 0..1."""
    if image.dim() != 4 or image.shape[-1] < 3:
        raise ValueError(f"expected an IMAGE tensor (B, H, W, 3), got shape {tuple(image.shape)}")
    if amount == 0:
        return image.clone()
    h, w = image.shape[1:3]
    rgb = image[..., :3].to(torch.float64)
    luma = rgb[..., 0] * _REC709[0] + rgb[..., 1] * _REC709[1] + rgb[..., 2] * _REC709[2]
    radius = max(1, int(math.floor(float(radius_pct) / 100.0 * min(h, w) + 0.5)))
    base = guided_filter_self(luma.unsqueeze(1), radius, float(edge_preserve)).squeeze(1)
    detail = luma - base
    lc = luma.clamp(0.0, 1.0)
    weight = (4.0 * lc * (1.0 - lc)).clamp(0.0, 1.0)
    if midtone_focus != 1:
        weight = weight ** float(midtone_focus) if midtone_focus != 0 else torch.ones_like(weight)
    new_luma = (luma + float(amount) * detail * weight).clamp(0.0, 1.0)

    ratio = new_luma / luma.clamp_min(LUMA_FLOOR)
    scaled = rgb * ratio.unsqueeze(-1)
    shifted = rgb + (new_luma - luma).unsqueeze(-1)
    out_rgb = torch.where((luma > LUMA_FLOOR).unsqueeze(-1), scaled, shifted).clamp(0.0, 1.0)
    out = image.clone()
    out[..., :3] = out_rgb.to(image.dtype)
    return out


class EmberClarity:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "amount": ("FLOAT", {"default": 0.35, "min": -1.0, "max": 2.0, "step": 0.01,
                                     "tooltip": "Local contrast added to the midtones. Negative softens."}),
                "radius_pct": ("FLOAT", {"default": 2.5, "min": 0.2, "max": 10.0, "step": 0.1,
                                         "tooltip": "Size of the local area, as % of the image's shorter side."}),
                "edge_preserve": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 0.2, "step": 0.0001,
                                            "tooltip": "Guided-filter eps. Lower keeps strong edges out of the effect (fewer halos)."}),
                "midtone_focus": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05,
                                            "tooltip": "Higher = the effect stays further from shadows and highlights."}),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "apply"
    CATEGORY = "Ember/Post-Processing"
    DESCRIPTION = "Edge-aware midtone local contrast on luma; colour ratios are preserved."

    def apply(self, image, amount, radius_pct, edge_preserve, midtone_focus):
        return (apply_clarity(image, amount, radius_pct, edge_preserve, midtone_focus),)


NODE_CLASS_MAPPINGS = {"EmberClarity": EmberClarity}
NODE_DISPLAY_NAME_MAPPINGS = {"EmberClarity": "Ember Clarity"}
