"""Ember Camera Look: makes a clean render read like a photo from a real sensor.

Stages, in order (each can be switched off by its widget):
  1. Sensor noise: Poisson shot noise (fixed 80-photon full scale) plus Gaussian read
     noise. Strength follows (noise_strength / 5)^2, so low values barely register.
  2. Bayer displacement: R and B shifted diagonally in opposite directions by
     demosaic_pixel_blur / 2 px (sub-pixel, bilinear, mirror edges). G stays put.
     This is channel misregistration, not blur.
  3. Horizontal motion blur: a box kernel with mirror edges. A fractional width
     cross-fades the two nearest odd kernel sizes.
  4. A JPEG round trip through Pillow at jpeg_compression quality.

Every stage works on float32 and truncates to uint8 at the same points as the
reference node, because a pixel-exact match depends on it. Rows are convolved one
at a time with np.convolve on purpose: a vectorised box filter sums in a different
order, and a 1-ulp float change can flip a uint8 truncation.

Reproducibility: the noise seed for each frame comes from Python's global
`random`, exactly as in the reference. With noise_strength = 0 the node is fully
deterministic. With noise > 0 two runs differ, unless `random` is seeded to the
same state before each run.
"""

import io
import math
import random

import numpy as np
import torch
from PIL import Image

_PHOTONS_FULL_SCALE = 80.0


def _noise_amounts(strength):
    t = (strength / 5.0) ** 2
    return t * 2.0, t * 4.0  # (shot-noise gain 0..2, read-noise std 0..4)


def _subpixel_shift(channel, dx, dy):
    height, width = channel.shape
    out = channel.astype(np.float32)
    if dx != 0:
        whole = int(np.floor(dx))
        frac = dx - whole
        pad = abs(whole) + 2
        padded = np.pad(out, ((0, 0), (pad, pad)), mode="reflect")
        a = np.roll(padded, whole, axis=1)
        b = np.roll(padded, whole + 1, axis=1)
        out = (a * (1.0 - frac) + b * frac)[:, pad:pad + width]
    if dy != 0:
        whole = int(np.floor(dy))
        frac = dy - whole
        pad = abs(whole) + 2
        padded = np.pad(out, ((pad, pad), (0, 0)), mode="reflect")
        a = np.roll(padded, whole, axis=0)
        b = np.roll(padded, whole + 1, axis=0)
        out = (a * (1.0 - frac) + b * frac)[pad:pad + height, :]
    return out


def _bayer_displacement(rgb_u8, strength):
    if strength <= 0:
        return rgb_u8.astype(np.float32)
    rgb = rgb_u8.astype(np.float32)
    half = strength * 0.5
    red = _subpixel_shift(rgb[:, :, 0], -half, -half)
    blue = _subpixel_shift(rgb[:, :, 2], half, half)
    return np.stack([red, rgb[:, :, 1], blue], axis=2)


def _box_blur_rows(rgb_u8, size):
    if size <= 1:
        return rgb_u8.astype(np.float32)
    size = size if size % 2 == 1 else size + 1
    pad = size // 2
    kernel = np.ones(size, dtype=np.float32) / size
    src = rgb_u8.astype(np.float32)
    padded = np.pad(src, ((0, 0), (pad, pad), (0, 0)), mode="reflect")
    out = np.zeros_like(src)
    for c in range(src.shape[2]):
        for row in range(src.shape[0]):
            out[row, :, c] = np.convolve(padded[row, :, c], kernel, mode="valid")
    return out


def _motion_blur(rgb_u8, width):
    if width <= 1:
        return rgb_u8.astype(np.float32)
    lower = int(math.floor(width))
    if lower % 2 == 0:
        lower -= 1
    lower = max(1, lower)
    upper = lower + 2
    mix = min(max((width - lower) / 2.0, 0.0), 1.0)
    low = _box_blur_rows(rgb_u8, lower)
    if mix <= 0.0:
        return low
    high = _box_blur_rows(rgb_u8, upper)
    return low * (1.0 - mix) + high * mix


def camera_pipeline(rgb_u8, demosaic, shot_gain, read_std, blur_width, jpeg_quality, seed):
    rng = np.random.default_rng(seed)
    img = rgb_u8.astype(np.float32)

    if shot_gain > 0:
        expected = np.clip(img, 0, 255) / 255.0 * _PHOTONS_FULL_SCALE
        shot = rng.poisson(expected).astype(np.float32) / _PHOTONS_FULL_SCALE * 255.0 - img
        img = np.clip(img + shot * (shot_gain / 2.0), 0.0, 255.0)
    if read_std > 0:
        img = np.clip(img + rng.normal(0.0, read_std, img.shape).astype(np.float32), 0.0, 255.0)

    if demosaic > 0:
        img = np.clip(_bayer_displacement(np.clip(img, 0, 255).astype(np.uint8), demosaic), 0, 255)

    if blur_width > 1:
        img = np.clip(_motion_blur(np.clip(img, 0, 255).astype(np.uint8), blur_width), 0, 255)

    buffer = io.BytesIO()
    Image.fromarray(np.clip(img, 0, 255).astype(np.uint8)).save(buffer, format="JPEG", quality=jpeg_quality)
    buffer.seek(0)
    return np.array(Image.open(buffer))


class EmberCameraLook:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "enabled": ("BOOLEAN", {"default": True, "label_on": "Enabled", "label_off": "Bypassed"}),
                "demosaic_pixel_blur": ("FLOAT", {
                    "default": 1.0, "min": 0.0, "max": 5.0, "step": 0.05,
                    "tooltip": "Bayer R/B diagonal displacement in px (not a blur). 0 = off, 1 = natural sensor offset.",
                }),
                "noise_strength": ("FLOAT", {
                    "default": 3.0, "min": 0.0, "max": 5.0, "step": 0.1,
                    "tooltip": "Sensor noise. 0 = none (fully deterministic) | 3 = standard | 5 = heavy.",
                }),
                "kernel_motion_blur": ("FLOAT", {
                    "default": 1.0, "min": 1.0, "max": 51.0, "step": 0.1,
                    "tooltip": "Horizontal box-blur width in px. 1 = none. Fractional values blend smoothly.",
                }),
                "jpeg_compression": ("INT", {
                    "default": 98, "min": 85, "max": 100, "step": 1,
                    "tooltip": "JPEG quality of the round trip. 100 = near-lossless.",
                }),
            },
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "execute"
    CATEGORY = "Ember/Post-Processing"
    DESCRIPTION = "Camera pipeline: sensor noise -> Bayer displacement -> motion blur -> JPEG."

    def execute(self, image, enabled, demosaic_pixel_blur, noise_strength, kernel_motion_blur, jpeg_compression):
        if not enabled:
            return (image,)

        shot_gain, read_std = _noise_amounts(noise_strength)
        frames = []
        for i in range(image.shape[0]):
            seed = random.randint(0, 2_147_483_647)
            rgb = (image[i].cpu().numpy() * 255).astype(np.uint8)
            out = camera_pipeline(rgb, demosaic_pixel_blur, shot_gain, read_std,
                                  kernel_motion_blur, jpeg_compression, seed)
            frames.append(torch.from_numpy(out.astype(np.float32) / 255.0).unsqueeze(0))

        if not frames:
            return (image,)
        return (torch.cat(frames, dim=0),)


NODE_CLASS_MAPPINGS = {"EmberCameraLook": EmberCameraLook}
NODE_DISPLAY_NAME_MAPPINGS = {"EmberCameraLook": "Ember Camera Look"}
