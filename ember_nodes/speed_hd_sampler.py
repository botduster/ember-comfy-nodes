"""Ember Speed HD Sampler: spectral progressive diffusion as a SAMPLER for SamplerCustomAdvanced.

The algorithm is SPEED, "Spectral Progressive Diffusion for Efficient Image and Video
Generation" (Xiao, Chao, Yariv & Wetzstein, 2026), reference code MIT-licensed at
https://github.com/howardhx/speed (see NOTICE). This module is written for this pack and
replaces "Aiorbust Speed HD Sampler", which runs the same algorithm.

What it does. Sampling starts on a smaller latent and grows it at chosen points in the
schedule, so the early, noise-dominated steps are cheap:
  1. The incoming latent is shrunk to the first scale by keeping its low DCT frequencies.
  2. The base k-diffusion solver runs over the schedule up to the next transition.
  3. At the transition the latent is grown to the next scale in a spectral basis
     (DCT, Haar DWT or FFT). The new high-frequency coefficients are Gaussian noise of
     amplitude sigma, and the whole latent is rescaled by kappa = r / (1 + (r - 1) * sigma).
  4. The sigma at that step is replaced by the aligned time sigma * kappa, and the solver
     carries on at the larger size. Repeat until scale 1.0.

Where transitions happen:
  * delta_optimal: from a power-law latent spectrum P(w) = A * w^-beta (preset or custom),
    the first step whose sigma <= 1 / (1 + sqrt(delta / (P * (1 + P - delta)))), with
    w = scale * min(H, W) / 2.
  * manual: the first step whose sigma <= each given threshold.
  A transition the schedule never reaches stops the growing there, and the output stays at
  that smaller scale (the reference does the same).

Determinism contract, identical to the reference node:
  * Transition noise comes from numpy.random.default_rng(seed + 10000 * k) for the k-th
    transition, drawn per (batch * frames, channel) plane in that order. DWT draws LH, HL,
    HH per plane; FFT draws real then imaginary per plane.
  * The spectral maths runs on CPU in float32 numpy, so the transition noise is the same
    on CPU and GPU. The base solver's own noise is NOT (torch CPU and CUDA streams differ).
  * The spectral code never touches the global torch or numpy RNG.
  * The base solver is looked up on comfy.k_diffusion.sampling at run time, and every
    segment restarts it: multistep history does not carry across a transition, and an
    ancestral solver re-seeds its noise from extra_args["seed"] at each segment.
"""

import math

import numpy as np
import torch
from scipy.fft import dctn, idctn

import comfy.k_diffusion.sampling as k_sampling
import comfy.samplers

# name: (A, beta) of the latent power spectrum; None = take spectrum_A / spectrum_beta.
SPECTRUM_PRESETS = {
    "flux": (203.615097, 1.915461),
    "wan21": (219.484718, 2.422687),
    "custom": None,
}

# Solvers that cannot run on a segmented schedule (adaptive step size / own noise model).
UNSUPPORTED_SOLVERS = {"dpm_fast", "dpm_adaptive", "lcm"}


# ---------------------------------------------------------------------------
# Widget parsing
# ---------------------------------------------------------------------------

def _numbers(text):
    return [float(part.strip()) for part in text.split(",") if part.strip()]


def check_scales(scales):
    # The comparisons are written so NaN passes them exactly as it does in the reference.
    if len(scales) == 0:
        raise ValueError("[Ember Speed HD] scales is empty; give at least one value, ending at 1.0.")
    if any(s <= 0.0 or s > 1.0 for s in scales):
        raise ValueError(f"[Ember Speed HD] every scale must be in (0, 1]; got {list(scales)}")
    if abs(scales[-1] - 1.0) > 1e-6:
        raise ValueError(f"[Ember Speed HD] the last scale must be 1.0; got {scales[-1]}")
    for lower, higher in zip(scales[:-1], scales[1:]):
        if not (lower < higher):
            raise ValueError(f"[Ember Speed HD] scales must strictly increase; got {list(scales)}")


def parse_scales(text):
    scales = _numbers(text)
    check_scales(scales)
    return scales


def parse_thresholds(text):
    thresholds = _numbers(text)
    if any(not (0.0 < v < 1.0) for v in thresholds):
        raise ValueError(f"[Ember Speed HD] every manual sigma must be in (0, 1); got {thresholds}")
    for above, below in zip(thresholds[:-1], thresholds[1:]):
        if not (above > below):
            raise ValueError(f"[Ember Speed HD] manual sigmas must strictly decrease; got {thresholds}")
    return thresholds


# ---------------------------------------------------------------------------
# Transition schedule
# ---------------------------------------------------------------------------

def kappa(t, r):
    return r / (1.0 + (r - 1.0) * t)


def spectrum_thresholds(scales, delta, A, beta, height, width):
    """Sigma below which each scale's top frequency is no longer noise-dominated (paper Eq. 9-10)."""
    check_scales(scales)
    nyquist = min(height, width) / 2.0
    thresholds = []
    for scale in scales[:-1]:
        power = A * abs(scale * nyquist) ** (-beta)
        if delta >= 1.0:
            raise ValueError(f"[Ember Speed HD] delta must be below 1; got {delta}")
        thresholds.append(1.0 / (1.0 + math.sqrt(delta / (power * (1.0 + power - delta)))))
    return thresholds


def _first_step_at_or_below(sigmas, threshold):
    steps = len(sigmas) - 1
    return next((j for j in range(steps) if float(sigmas[j]) <= threshold), steps)


def plan_transitions(sigmas, scales, thresholds):
    """[(step index, scale before, scale after)], cut short at the first threshold the schedule never reaches."""
    plan = []
    steps = len(sigmas) - 1
    for scale_from, scale_to, threshold in zip(scales[:-1], scales[1:], thresholds):
        step = _first_step_at_or_below(sigmas, threshold)
        if step >= steps:
            break
        plan.append((step, scale_from, scale_to))
    return plan


# ---------------------------------------------------------------------------
# Spectral growth (numpy, float32, per plane)
# ---------------------------------------------------------------------------

def _grow_dct(planes, size, t, seed):
    height, width = size
    src_h, src_w = planes.shape[-2], planes.shape[-1]
    if height < src_h or width < src_w:
        raise ValueError(f"[Ember Speed HD] DCT cannot grow ({src_h}, {src_w}) to the smaller {size}.")
    rng = np.random.default_rng(seed)
    out = np.empty(planes.shape[:-2] + (height, width), dtype=np.float32)
    for index in np.ndindex(*planes.shape[:-2]):
        low = dctn(planes[index], type=2, norm="ortho")
        coeffs = t * rng.standard_normal((height, width)).astype(np.float32)
        coeffs[:src_h, :src_w] = low
        out[index] = idctn(coeffs, type=2, norm="ortho").astype(np.float32)
    return out


def _grow_dwt(planes, t, seed):
    try:
        import pywt
    except ImportError as exc:
        raise ImportError("[Ember Speed HD] transform=dwt needs PyWavelets (pip install PyWavelets), "
                          "or use transform=dct / fft.") from exc
    src_h, src_w = planes.shape[-2], planes.shape[-1]
    rng = np.random.default_rng(seed)
    out = np.empty(planes.shape[:-2] + (src_h * 2, src_w * 2), dtype=np.float32)
    for index in np.ndindex(*planes.shape[:-2]):
        approx = planes[index]
        horizontal = t * rng.standard_normal(approx.shape).astype(np.float32)
        vertical = t * rng.standard_normal(approx.shape).astype(np.float32)
        diagonal = t * rng.standard_normal(approx.shape).astype(np.float32)
        out[index] = pywt.waverec2([approx, (horizontal, vertical, diagonal)], "haar",
                                   mode="periodization").astype(np.float32)
    return out


def _grow_fft(planes, size, t, seed):
    height, width = size
    src_h, src_w = planes.shape[-2], planes.shape[-1]
    if height < src_h or width < src_w:
        raise ValueError(f"[Ember Speed HD] FFT cannot grow ({src_h}, {src_w}) to the smaller {size}.")
    rng = np.random.default_rng(seed)
    top, left = (height - src_h) // 2, (width - src_w) // 2
    out = np.empty(planes.shape[:-2] + (height, width), dtype=np.float32)
    for index in np.ndindex(*planes.shape[:-2]):
        low = np.fft.fftshift(np.fft.fft2(planes[index], norm="ortho"))
        real = rng.standard_normal((height, width)).astype(np.float32)
        imag = rng.standard_normal((height, width)).astype(np.float32)
        # np.sqrt (a float64 scalar), not math.sqrt: it sets the precision of this array.
        spectrum = np.fft.fftshift(t * (real + 1j * imag) / np.sqrt(2.0))
        spectrum[top:top + src_h, left:left + src_w] = low
        out[index] = np.fft.ifft2(np.fft.ifftshift(spectrum), norm="ortho").real.astype(np.float32)
    return out


def _fold(x):
    """5-D video (B,C,T,H,W) folded to planes (B*T,C,H,W); anything else unchanged."""
    if x.ndim == 5:
        b, c, frames, h, w = x.shape
        return x.permute(0, 2, 1, 3, 4).reshape(b * frames, c, h, w)
    return x


def _from_planes(planes, like, height, width):
    if like.ndim == 5:
        b, c, frames = like.shape[0], like.shape[1], like.shape[2]
        return planes.reshape(b, frames, c, height, width).permute(0, 2, 1, 3, 4)
    return planes


def shrink_to_scale(x, scale):
    """Keep the low DCT frequencies of each plane: the starting latent at the first scale."""
    if scale >= 1.0:
        return x
    height, width = round(x.shape[-2] * scale), round(x.shape[-1] * scale)
    planes = _fold(x).detach().cpu().float().numpy()
    out = np.empty(planes.shape[:-2] + (height, width), dtype=np.float32)
    for index in np.ndindex(*planes.shape[:-2]):
        coeffs = dctn(planes[index], type=2, norm="ortho")
        out[index] = idctn(coeffs[:height, :width], type=2, norm="ortho").astype(np.float32)
    small = torch.from_numpy(out).to(device=x.device, dtype=x.dtype)
    return _from_planes(small, x, height, width)


def grow_at_transition(x, scale_from, scale_to, t, transform, seed, full_h, full_w):
    """Grow x from scale_from to scale_to at sigma t. Returns (grown latent, aligned sigma)."""
    if transform not in ("dct", "dwt", "fft"):
        raise ValueError(f"[Ember Speed HD] transform must be dct, dwt or fft; got {transform!r}")
    ratio = scale_to / scale_from
    height, width = round(scale_to * full_h), round(scale_to * full_w)
    if x.ndim not in (4, 5):
        raise ValueError(f"[Ember Speed HD] expected a 4-D or 5-D latent, got shape {tuple(x.shape)}")
    planes = _fold(x).detach().cpu().float().numpy()
    if transform == "dwt":
        if abs(ratio - 2.0) > 1e-6:
            raise ValueError(f"[Ember Speed HD] DWT needs consecutive scales exactly 2x apart; got {ratio:.4f}x. "
                             "Use transform=dct or fft for other ratios.")
        grown = _grow_dwt(planes, t, seed)
    elif transform == "dct":
        grown = _grow_dct(planes, (height, width), t, seed)
    else:
        grown = _grow_fft(planes, (height, width), t, seed)
    scaled = (kappa(t, ratio) * grown).astype(np.float32)
    grown_t = torch.from_numpy(scaled).to(device=x.device, dtype=x.dtype)
    return _from_planes(grown_t, x, height, width), t * kappa(t, ratio)


# ---------------------------------------------------------------------------
# The sampler function (called by comfy.samplers.KSAMPLER with extra_options as kwargs)
# ---------------------------------------------------------------------------

def _shift_callback(callback, offset):
    if callback is None:
        return None

    def shifted(info):
        info = dict(info)
        info["i"] = info.get("i", 0) + offset
        callback(info)
    return shifted


@torch.no_grad()
def sample_ember_speed_hd(model, x, sigmas, extra_args=None, callback=None, disable=None, *,
                          transform="dct", base_sampler="euler", mode="delta_optimal", scales=None,
                          delta=0.01, spectrum_A=203.615097, spectrum_beta=1.915461,
                          manual_sigmas=None, seed=0):
    solver = getattr(k_sampling, f"sample_{base_sampler}", None)
    if solver is None:
        raise ValueError(f"[Ember Speed HD] unknown base sampler {base_sampler!r}")
    extra_args = {} if extra_args is None else extra_args
    full_h, full_w = x.shape[-2], x.shape[-1]

    if not scales or len(scales) < 2:
        return solver(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable)

    if mode == "delta_optimal":
        thresholds = spectrum_thresholds(scales, delta, spectrum_A, spectrum_beta, full_h, full_w)
    elif mode == "manual":
        thresholds = manual_sigmas or []
        if len(thresholds) != len(scales) - 1:
            raise ValueError(f"[Ember Speed HD] manual_sigmas needs {len(scales) - 1} value(s), one per "
                             f"transition in scales; got {len(thresholds)}.")
    else:
        raise ValueError(f"[Ember Speed HD] mode must be delta_optimal or manual; got {mode!r}")
    plan = plan_transitions(sigmas, scales, thresholds)

    if scales[0] < 1.0:
        x = shrink_to_scale(x, scales[0])

    sigmas = sigmas.clone()  # the aligned sigma is written into this copy at each transition
    starts = [0] + [step for step, _, _ in plan]
    for k, start in enumerate(starts):
        end = plan[k][0] if k < len(plan) else len(sigmas) - 1
        segment = sigmas[start:end + 1]
        if len(segment) >= 2:
            x = solver(model, x, segment, extra_args=extra_args,
                       callback=_shift_callback(callback, start), disable=disable)
        if k >= len(plan):
            break
        step, scale_from, scale_to = plan[k]
        x, aligned = grow_at_transition(x, scale_from, scale_to, float(sigmas[step]), transform,
                                        seed + (k + 1) * 10000, full_h, full_w)
        sigmas[step] = float(aligned)
    return x


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

def _solver_names():
    try:
        names = [name[len("sample_"):] for name in dir(k_sampling) if name.startswith("sample_")]
    except Exception:
        names = ["euler", "euler_ancestral", "heun", "dpmpp_2m", "uni_pc"]
    return sorted(name for name in names if name not in UNSUPPORTED_SOLVERS)


class EmberSpeedHDSampler:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "base_sampler": (_solver_names(), {
                    "default": "euler",
                    "tooltip": "The k-diffusion solver run between transitions. It restarts at every "
                               "transition, so multistep solvers lose their history there.",
                }),
                "transform": (["dct", "dwt", "fft"], {
                    "default": "dct",
                    "tooltip": "Basis used to grow the latent. dct: any scale ratio. dwt (Haar): scales must "
                               "double each time, needs PyWavelets. fft: any ratio.",
                }),
                "mode": (["delta_optimal", "manual"], {
                    "default": "delta_optimal",
                    "tooltip": "delta_optimal: place transitions from scales, delta and the spectrum. "
                               "manual: grow at the sigmas listed in manual_sigmas.",
                }),
                "model_preset": (list(SPECTRUM_PRESETS.keys()), {
                    "default": "flux",
                    "tooltip": "Latent power spectrum for delta_optimal. flux / wan21 are measured values; "
                               "custom uses spectrum_A and spectrum_beta.",
                }),
                "scales": ("STRING", {
                    "default": "0.5,1.0",
                    "tooltip": "Comma-separated resolution fractions, increasing, ending at 1.0 "
                               "(e.g. 0.5,1.0 or 0.25,0.5,1.0).",
                }),
                "delta": ("FLOAT", {
                    "default": 0.01, "min": 1e-4, "max": 0.5, "step": 0.001,
                    "tooltip": "Noise-dominated tolerance for delta_optimal. Smaller grows later.",
                }),
                "manual_sigmas": ("STRING", {
                    "default": "0.85",
                    "tooltip": "manual mode: comma-separated sigmas in (0, 1), decreasing, one per transition "
                               "(e.g. 0.95,0.85 for scales 0.25,0.5,1.0).",
                }),
                "spectrum_A": ("FLOAT", {
                    "default": 203.615097, "min": 0.0, "max": 1e6, "step": 0.001,
                    "tooltip": "Spectrum amplitude A (model_preset = custom).",
                }),
                "spectrum_beta": ("FLOAT", {
                    "default": 1.915461, "min": 0.0, "max": 10.0, "step": 0.001,
                    "tooltip": "Spectrum decay exponent beta (model_preset = custom).",
                }),
                "seed": ("INT", {
                    "default": 0, "min": 0, "max": 2**31 - 1, "step": 1,
                    "tooltip": "Seed for the noise added when the latent grows.",
                }),
            },
        }

    RETURN_TYPES = ("SAMPLER",)
    RETURN_NAMES = ("sampler",)
    FUNCTION = "get_sampler"
    CATEGORY = "Ember/Sampling"
    DESCRIPTION = ("Spectral progressive diffusion: samples small, grows the latent in a spectral basis as "
                   "the noise falls. Connect to SamplerCustomAdvanced.")

    def get_sampler(self, base_sampler, transform, mode, model_preset, scales, delta,
                    manual_sigmas, spectrum_A, spectrum_beta, seed):
        preset = SPECTRUM_PRESETS.get(model_preset)
        if preset is not None:
            A, beta = preset
        else:
            A, beta = float(spectrum_A), float(spectrum_beta)
        parsed_scales = parse_scales(scales)
        thresholds = parse_thresholds(manual_sigmas) if mode == "manual" else []
        sampler = comfy.samplers.KSAMPLER(sample_ember_speed_hd, extra_options={
            "transform": transform,
            "base_sampler": base_sampler,
            "mode": mode,
            "scales": parsed_scales,
            "delta": float(delta),
            "spectrum_A": A,
            "spectrum_beta": beta,
            "manual_sigmas": thresholds,
            "seed": int(seed),
        })
        return (sampler,)


NODE_CLASS_MAPPINGS = {"EmberSpeedHDSampler": EmberSpeedHDSampler}
NODE_DISPLAY_NAME_MAPPINGS = {"EmberSpeedHDSampler": "Ember Speed HD Sampler"}
