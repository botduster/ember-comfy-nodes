"""Ember Renoise: adds seeded digital-sensor grain after all other processing.

Real sensor noise has two parts, and they look different:
  * luminance grain: one noise field shared by R, G and B (fine texture)
  * chroma scatter: an independent field per channel (colour blotches)

Each part gets its own Gaussian softening, then the total is weighted toward the
shadows (a mask falling from 1 in black to 0 above the preset's shadow threshold),
because noise is most visible in the darks.

grain_size changes the size of the grain, never its strength (the ISO preset sets
strength). 1.0 draws noise per pixel. Above 1 it draws at reduced resolution and
upscales with nearest-neighbour. Below 1 it draws at higher resolution and
area-averages down.

Determinism contract, identical to the reference node:
- torch.manual_seed(seed) is set globally, then luma noise is drawn before chroma
  noise, on the CUDA device when available, otherwise CPU.
- The same seed gives the same output on the same device type. CPU and CUDA random
  streams differ, so a CPU run does not reproduce a GPU run.
- The global torch RNG is left advanced by exactly those draws, which matters to any
  node that runs afterwards without its own seed.
"""

import math

import torch
import torch.nn.functional as F

try:
    import kornia.color as kornia_color
    import kornia.filters as kornia_filters
    KORNIA_AVAILABLE = True
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
except ImportError:
    KORNIA_AVAILABLE = False
    DEVICE = torch.device("cpu")

# name: (luma grain, chroma scatter, luma softness sigma, chroma softness sigma, shadow threshold)
ISO_PRESETS = {
    "ISO 0 (Off)":     (0.000, 0.000, 0.00, 0.0, 0.00),
    "ISO 50":          (0.004, 0.002, 0.35, 1.2, 0.98),
    "ISO 100 (Clean)": (0.008, 0.005, 0.40, 1.5, 0.95),
    "ISO 200":         (0.012, 0.008, 0.45, 1.8, 0.90),
    "ISO 400":         (0.018, 0.012, 0.50, 2.0, 0.85),
    "ISO 800":         (0.025, 0.016, 0.55, 2.5, 0.78),
    "ISO 1600":        (0.035, 0.025, 0.65, 3.0, 0.70),
    "Night Mode":      (0.025, 0.035, 1.20, 4.0, 0.50),
}


def _soften(field, sigma):
    if sigma <= 0:
        return field
    size = 2 * math.ceil(3.0 * sigma) + 1
    return kornia_filters.gaussian_blur2d(field, (size, size), (sigma, sigma))


def _noise_field(batch, channels, height, width, grain_size, device):
    if grain_size == 1.0:
        return torch.randn(batch, channels, height, width, device=device)
    low_h = max(1, round(height / grain_size))
    low_w = max(1, round(width / grain_size))
    field = torch.randn(batch, channels, low_h, low_w, device=device)
    return F.interpolate(field, size=(height, width), mode="nearest" if grain_size > 1.0 else "area")


class EmberRenoise:
    @classmethod
    def INPUT_TYPES(cls):
        if not KORNIA_AVAILABLE:
            # Kept identical to the reference so a missing dependency shows up the same way.
            return {"required": {"error": ("STRING", {
                "default": "Kornia not installed. Run: pip install kornia", "multiline": True})}}
        return {
            "required": {
                "image": ("IMAGE",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff,
                                 "control_after_generate": "randomize"}),
                "iso_preset": (list(ISO_PRESETS.keys()), {
                    "default": "ISO 400",
                    "tooltip": "Noise level. ISO 0 = off | ISO 100 = clean | ISO 1600 = heavy | "
                               "Night Mode = chroma-heavy.",
                }),
                "grain_size": ("FLOAT", {
                    "default": 1.0, "min": 0.1, "max": 8.0, "step": 0.1,
                    "tooltip": "Grain size, not strength. 1.0 = per pixel, >1 = bigger grains, <1 = finer.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "renoise"
    CATEGORY = "Ember/Post-Processing"
    DESCRIPTION = "Seeded sensor grain: luma grain + chroma scatter, weighted toward shadows."

    def renoise(self, image, seed, iso_preset, grain_size=1.0):
        if not KORNIA_AVAILABLE:
            raise ImportError("Kornia is required. Install with: pip install kornia")

        preset = ISO_PRESETS.get(iso_preset)
        if preset is None:
            return (image,)
        luma_amount, chroma_amount, luma_sigma, chroma_sigma, shadow_threshold = preset
        if luma_amount == 0.0 and chroma_amount == 0.0:
            return (image,)

        torch.manual_seed(seed)

        batch, height, width, _ = image.shape
        img = image.permute(0, 3, 1, 2).to(DEVICE)
        noise = torch.zeros_like(img)

        if luma_amount > 0:
            luma = _noise_field(batch, 1, height, width, grain_size, DEVICE).expand(batch, 3, height, width).clone()
            noise = noise + _soften(luma, luma_sigma) * luma_amount

        if chroma_amount > 0:
            chroma = _noise_field(batch, 3, height, width, grain_size, DEVICE)
            noise = noise + _soften(chroma, chroma_sigma) * chroma_amount

        if shadow_threshold > 0:
            luminance = kornia_color.rgb_to_grayscale(img)
            weight = 1.0 - torch.clamp((luminance - shadow_threshold) / (1.0 - shadow_threshold + 1e-6), 0.0, 1.0)
            noise = noise * weight

        out = torch.clamp(img + noise, 0.0, 1.0)
        return (out.permute(0, 2, 3, 1).to(image.device),)


NODE_CLASS_MAPPINGS = {"EmberRenoise": EmberRenoise}
NODE_DISPLAY_NAME_MAPPINGS = {"EmberRenoise": "Ember Renoise"}
