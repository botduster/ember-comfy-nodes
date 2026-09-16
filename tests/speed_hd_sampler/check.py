#!/usr/bin/env python3
"""Ember Speed HD Sampler vs Aiorbust Speed HD Sampler: same inputs, zero tolerance.

Both nodes run in one process on the same ComfyUI, through the same path SamplerCustomAdvanced
uses: node.get_sampler(**widgets) -> KSAMPLER.sample(guider, sigmas, extra_args, callback, noise,
latent_image, denoise_mask, disable_pbar). The guider is a small deterministic fake model, so this
runs on CPU in seconds. Every observable result is compared exactly:

  * the sampler object (class, extra_options with value types, inpaint_options)
  * every call into the base k-diffusion solver: the latent it received (after each spectral
    growth, i.e. the transition noise), the sigma slice (incl. the aligned sigma), kwargs
  * every model evaluation (latent, sigma, seed) and every progress callback (step index, latents)
  * the sampled latent, the caller's sigmas afterwards, the global torch and numpy RNG states
  * raised exceptions: where (get_sampler / sample) and type

Tensors must match in dtype, shape, stride, requires_grad, torch.equal AND byte for byte.
Every case also runs the reference twice: that is the noise floor and must be 0 too.

    python3 check.py --comfyui <ComfyUI> --original <public-aiorbust-nodes-pack> \
        [--ours <dir with speed_hd_sampler.py>] [--out results.json] [--fail-fast]

Exit code 0 only when every check matches (and the floor is clean).
"""

import argparse
import hashlib
import importlib
import json
import math
import os
import platform
import struct
import subprocess
import sys
import time
import types

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))

# Node 1059:1184 in run 3933479f (Wyatt's pod, 2026-09-16), widget values as the API graph sent them.
GRAPH_WIDGETS = {
    "base_sampler": "euler_ancestral", "transform": "dwt", "mode": "delta_optimal", "model_preset": "custom",
    "scales": "0.5,1.0", "delta": 0.01, "manual_sigmas": "0.85", "spectrum_A": 203.615097,
    "spectrum_beta": 1.450000000000009, "seed": 1183768150,
}
GRAPH_NOISE_SEED = 109542645839864            # RandomNoise 1059:1057
GRAPH_LATENT = (1, 16, 1, 348, 200)           # 1600x2784 (EmberResolutionMP 4.45 MP 9:16 /32), Wan21 latent
GRAPH_SCHEDULE = ("flow", 6.0, "simple", 8)   # ModelSamplingAuraFlow shift 6, BasicScheduler simple 8, denoise 1
# widgets_values of the same node in the saved UI workflow (Krea 2 Beyond Reality V3.0):
UI_WIDGETS_VALUES = ["euler_ancestral", "dwt", "delta_optimal", "custom", "0.5,1.0", 0.01, "0.85", 203.615097, 1.37,
                     1183768150, "fixed"]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def bootstrap(comfyui):
    import torch
    if not torch.cuda.is_available():
        sys.modules.setdefault("triton", None)  # comfy_kitchen's triton probe raises without a GPU driver
    sys.path.insert(0, comfyui)
    from comfy.cli_args import args
    args.cpu = True
    import comfy.samplers  # noqa: F401
    import comfy.k_diffusion.sampling  # noqa: F401
    import comfy.model_sampling  # noqa: F401


def load_module(package, directory, module):
    """Import directory/module.py as package.module without running any __init__.py there."""
    pkg = types.ModuleType(package)
    pkg.__path__ = [directory]
    sys.modules[package] = pkg
    return importlib.import_module(f"{package}.{module}")


def sha256(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ---------------------------------------------------------------------------
# Exact comparison
# ---------------------------------------------------------------------------

def same(a, b, path="$"):
    """None when identical, else '<path>: <reason>'. No tolerance anywhere."""
    import numpy as np
    import torch

    if isinstance(a, torch.Tensor) or isinstance(b, torch.Tensor):
        if not (isinstance(a, torch.Tensor) and isinstance(b, torch.Tensor)):
            return f"{path}: {type(a).__name__} vs {type(b).__name__}"
        for attr in ("dtype", "shape", "device", "requires_grad"):
            if getattr(a, attr) != getattr(b, attr):
                return f"{path}: {attr} {getattr(a, attr)} != {getattr(b, attr)}"
        if a.stride() != b.stride():
            return f"{path}: stride {a.stride()} != {b.stride()}"
        bytes_a = a.detach().contiguous().reshape(-1).view(torch.uint8)
        bytes_b = b.detach().contiguous().reshape(-1).view(torch.uint8)
        has_nan = a.is_floating_point() and bool(torch.isnan(a).any())
        if not torch.equal(bytes_a, bytes_b) or not (torch.equal(a, b) or has_nan):
            diff = int((a != b).sum()) if a.shape == b.shape else -1
            return f"{path}: values differ ({diff} elements)"
        return None
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        if not (isinstance(a, np.ndarray) and isinstance(b, np.ndarray)):
            return f"{path}: {type(a).__name__} vs {type(b).__name__}"
        if a.dtype != b.dtype or a.shape != b.shape or a.tobytes() != b.tobytes():
            return f"{path}: arrays differ"
        return None
    if type(a) is not type(b):
        return f"{path}: type {type(a).__name__} != {type(b).__name__}"
    if isinstance(a, float):
        return None if struct.pack("<d", a) == struct.pack("<d", b) else f"{path}: {a!r} != {b!r}"
    if isinstance(a, dict):
        if list(a.keys()) != list(b.keys()):
            return f"{path}: keys {list(a.keys())} != {list(b.keys())}"
        for key in a:
            found = same(a[key], b[key], f"{path}.{key}")
            if found:
                return found
        return None
    if isinstance(a, (list, tuple)):
        if len(a) != len(b):
            return f"{path}: length {len(a)} != {len(b)}"
        for i, (x, y) in enumerate(zip(a, b)):
            found = same(x, y, f"{path}[{i}]")
            if found:
                return found
        return None
    return None if a == b else f"{path}: {a!r} != {b!r}"


# ---------------------------------------------------------------------------
# Fake model and schedules
# ---------------------------------------------------------------------------

def model_sampling(kind, shift):
    import comfy.model_sampling as ms
    if kind == "flow":
        class Flow(ms.ModelSamplingDiscreteFlow, ms.CONST):
            pass
        sampling = Flow()
        sampling.set_parameters(shift=shift, multiplier=1.0)  # ModelSamplingAuraFlow
        return sampling

    class Eps(ms.ModelSamplingDiscrete, ms.EPS):
        pass
    return Eps()


def schedule(spec):
    """(kind, shift, scheduler, steps[, first_step]) -> (model_sampling, sigmas)."""
    import comfy.samplers
    kind, shift, scheduler, steps = spec[:4]
    sampling = model_sampling(kind, shift)
    sigmas = comfy.samplers.calculate_sigmas(sampling, scheduler, steps).cpu()
    if len(spec) > 4:
        sigmas = sigmas[spec[4]:].clone()
    return sampling, sigmas


def denoise(x, sigma):
    """Deterministic stand-in for a diffusion model: spatial blur, channel mixing, a position ramp."""
    import torch
    import torch.nn.functional as F
    video = x.ndim == 5
    y = x.movedim(2, 1).reshape(-1, x.shape[1], x.shape[-2], x.shape[-1]) if video else x
    blur = F.avg_pool2d(y, 3, stride=1, padding=1, count_include_pad=False)
    ramp = torch.linspace(-0.2, 0.2, y.shape[-1], dtype=y.dtype)
    mixed = torch.tanh(0.8 * blur + 0.3 * torch.roll(blur, shifts=1, dims=1) - 0.1 * y) + ramp
    frames = x.shape[2] if video else 1
    s = sigma.to(y.dtype).repeat_interleave(frames).view(-1, 1, 1, 1)
    out = (1.0 - s) * mixed + s * 0.5 * blur
    if video:
        out = out.reshape(x.shape[0], x.shape[2], x.shape[1], x.shape[-2], x.shape[-1]).movedim(1, 2)
    return out


class FakeGuider:
    """What KSAMPLER.sample receives as model_wrap (normally a CFGGuider)."""
    cfg = 1.0

    def __init__(self, sampling, log):
        self.inner_model = types.SimpleNamespace(model_sampling=sampling)
        self.model_patcher = types.SimpleNamespace(model=self.inner_model,
                                                   get_model_object=lambda name: getattr(self.inner_model, name))
        self.log = log

    def __call__(self, x, sigma, model_options={}, seed=None):
        self.log.append(("model", x.clone(), sigma.clone(), seed))
        return denoise(x, sigma)


# ---------------------------------------------------------------------------
# One run of one implementation
# ---------------------------------------------------------------------------

def run(node_cls, widgets, latent_shape=(1, 4, 1, 32, 24), sched=("flow", 6.0, "simple", 8), noise_seed=7,
        dtype="float32", with_callback=True, denoise_from_latent=False):
    import numpy as np
    import torch
    import comfy.k_diffusion.sampling as kds

    torch.manual_seed(1234)
    np.random.seed(4321)
    result = {}
    log = []
    try:
        (sampler,) = node_cls().get_sampler(**widgets)
    except Exception as exc:  # noqa: BLE001 - the exception type IS the observable
        result["error"] = ("get_sampler", type(exc).__name__)
        result["torch_rng"] = torch.get_rng_state()
        return result
    result["sampler"] = (type(sampler).__module__, type(sampler).__name__, sampler.extra_options,
                         sampler.inpaint_options)

    sampling, sigmas = schedule(sched)
    dt = getattr(torch, dtype)
    gen = torch.Generator().manual_seed(noise_seed)
    noise = torch.randn(latent_shape, generator=gen, dtype=torch.float32).to(dt)
    latent = (torch.randn(latent_shape, generator=gen, dtype=torch.float32) * 0.5).to(dt) if denoise_from_latent \
        else torch.zeros(latent_shape, dtype=dt)
    sigmas_in = sigmas.clone()

    name = f"sample_{widgets['base_sampler']}"
    original_solver = getattr(kds, name, None)
    if original_solver is not None:
        def logged(model, x, sigmas, extra_args=None, callback=None, disable=None, **kwargs):
            log.append(("solver", x.clone(), sigmas.clone(), sorted(kwargs), disable, callback is None,
                        x.stride(), sorted((extra_args or {}).keys())))
            return original_solver(model, x, sigmas, extra_args=extra_args, callback=callback, disable=disable,
                                   **kwargs)
        setattr(kds, name, logged)

    def callback(step, denoised, x, total):
        log.append(("callback", step, denoised.clone(), x.clone(), total))

    try:
        out = sampler.sample(FakeGuider(sampling, log), sigmas, {"model_options": {}, "seed": noise_seed},
                             callback if with_callback else None, noise, latent, None, True)
        result["out"] = out
    except Exception as exc:  # noqa: BLE001
        result["error"] = ("sample", type(exc).__name__)
    finally:
        if original_solver is not None:
            setattr(kds, name, original_solver)
    result["log"] = log
    result["sigmas_after"] = sigmas
    result["sigmas_unchanged"] = torch.equal(sigmas, sigmas_in)
    result["torch_rng"] = torch.get_rng_state()
    result["numpy_rng"] = np.random.get_state()
    return result


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------

def spec_without_tooltips(types_dict):
    out = {}
    for section, inputs in types_dict.items():
        out[section] = {}
        for name, spec in inputs.items():
            kind = spec[0]
            opts = {k: v for k, v in (spec[1] if len(spec) > 1 else {}).items() if k != "tooltip"}
            out[section][name] = (kind, opts)
    return out


def widget_names(node_cls):
    names = []
    for section in ("required", "optional"):
        for name in node_cls.INPUT_TYPES().get(section, {}):
            names.append(name)
            if name in ("seed", "noise_seed"):
                names.append("control_after_generate")  # inserted by ComfyUI's frontend
    return names


def validates(spec, value):
    kind, opts = spec[0], (spec[1] if len(spec) > 1 else {})
    if isinstance(kind, list):
        return value in kind
    if kind == "STRING":
        return isinstance(value, str)
    if kind in ("FLOAT", "INT"):
        return opts.get("min", -math.inf) <= value <= opts.get("max", math.inf)
    return False


def base_widgets(**overrides):
    widgets = dict(GRAPH_WIDGETS)
    widgets.update(overrides)
    return widgets


def cases(ref_cls, our_cls):
    """Yield (group, label, kwargs for run()) or (group, label, callable returning a comparable)."""
    # 1. Interface: what a UI workflow and the API validator see.
    yield "interface", "INPUT_TYPES (tooltips excluded)", lambda cls: spec_without_tooltips(cls.INPUT_TYPES())
    yield "interface", "RETURN_TYPES / RETURN_NAMES / OUTPUT_NODE", \
        lambda cls: (cls.RETURN_TYPES, getattr(cls, "RETURN_NAMES", None), getattr(cls, "OUTPUT_NODE", False))
    yield "interface", "widgets_values order incl. control_after_generate", widget_names
    yield "interface", "saved V3.0 widgets_values validate", lambda cls: [
        validates(cls.INPUT_TYPES()["required"][name], value)
        for name, value in zip([n for n in widget_names(cls) if n != "control_after_generate"],
                               [v for i, v in enumerate(UI_WIDGETS_VALUES) if i != 10])]

    # 2. The real graph: Wyatt's run 3933479f, full-size latent and schedule.
    yield "graph_3933479f", "API widgets, 1600x2784 latent, shift 6 simple 8", dict(
        widgets=base_widgets(), latent_shape=GRAPH_LATENT, sched=GRAPH_SCHEDULE, noise_seed=GRAPH_NOISE_SEED)
    yield "graph_3933479f", "UI widgets_values (spectrum_beta 1.37)", dict(
        widgets=base_widgets(spectrum_beta=1.37), latent_shape=GRAPH_LATENT, sched=GRAPH_SCHEDULE,
        noise_seed=GRAPH_NOISE_SEED)
    yield "graph_3933479f", "no progress callback", dict(
        widgets=base_widgets(), latent_shape=GRAPH_LATENT, sched=GRAPH_SCHEDULE, noise_seed=GRAPH_NOISE_SEED,
        with_callback=False)
    yield "graph_3933479f", "node defaults", dict(
        widgets=dict(base_sampler="euler", transform="dct", mode="delta_optimal", model_preset="flux",
                     scales="0.5,1.0", delta=0.01, manual_sigmas="0.85", spectrum_A=203.615097,
                     spectrum_beta=1.915461, seed=0),
        latent_shape=GRAPH_LATENT, sched=GRAPH_SCHEDULE, noise_seed=GRAPH_NOISE_SEED)

    # 3. Full grid of every combo option x scale lists x latent shapes (incl. odd sizes and batch).
    manual_for = {2: "0.85", 3: "0.95,0.85", 4: "0.97,0.93,0.85", 1: "0.85"}
    shapes = [(2, 4, 32, 24), (1, 4, 2, 24, 16), (1, 4, 31, 23), (1, 4, 1, 33, 21)]
    for transform in ("dct", "dwt", "fft"):
        for mode in ("delta_optimal", "manual"):
            for preset in ("flux", "wan21", "custom"):
                for scales in ("0.5,1.0", "0.25,0.5,1.0", "0.3,1.0", "1.0", "0.125,0.25,0.5,1.0"):
                    for shape in shapes:
                        widgets = base_widgets(base_sampler="euler", transform=transform, mode=mode,
                                               model_preset=preset, scales=scales,
                                               manual_sigmas=manual_for[scales.count(",") + 1])
                        yield "grid", f"{transform}/{mode}/{preset}/{scales}/{shape}", dict(
                            widgets=widgets, latent_shape=shape)

    # 4. Edge values, one factor at a time around the graph settings (small 5-D latent).
    small = (1, 16, 1, 64, 40)
    edges = [("delta", v) for v in (1e-4, 0.001, 0.2, 0.5, 0.9999, 1.0)]
    edges += [("spectrum_A", v) for v in (0.0, 1e-3, 1.0, 203.615097, 1e6)]
    edges += [("spectrum_beta", v) for v in (0.0, 1.0, 1.915461, 10.0)]
    edges += [("seed", v) for v in (0, 1, 9999, 2**31 - 1)]
    edges += [("scales", v) for v in (" 0.5 , 1.0 ,", "0.5,,1.0", "nan", "0.5", "1.0,0.5", "0,1.0", "", "abc",
                                      "0.5,0.5,1.0", "1.0000001", "0.9999995", "1e-3,1.0", "0.5,1.0,1.0",
                                      "0.75,1.0", "0.5,nan,1.0")]
    edges += [("model_preset", v) for v in ("flux", "wan21", "not-a-preset")]
    edges += [("transform", "bogus"), ("mode", "bogus"), ("base_sampler", "bogus")]
    for transform in ("dct", "dwt", "fft"):
        for key, value in edges:
            yield "edges", f"{transform}: {key}={value!r}", dict(
                widgets=base_widgets(**{"transform": transform, key: value}), latent_shape=small)
        for sigmas_text, scales in (("0.85", "0.5,1.0"), ("0.999", "0.5,1.0"), ("0.001", "0.5,1.0"),
                                    (" 0.5 ", "0.5,1.0"), ("0.9,", "0.5,1.0"), ("0.95,0.85", "0.5,1.0"),
                                    ("", "0.5,1.0"), ("1.0", "0.5,1.0"), ("0", "0.5,1.0"), ("-0.1", "0.5,1.0"),
                                    ("0.5,0.9", "0.25,0.5,1.0"), ("0.9,0.9", "0.25,0.5,1.0"), ("nan", "0.5,1.0"),
                                    ("abc", "0.5,1.0"), ("0.95,0.85", "0.25,0.5,1.0"), ("0.85", "1.0")):
            yield "edges", f"{transform}: manual {sigmas_text!r} scales {scales!r}", dict(
                widgets=base_widgets(transform=transform, mode="manual", manual_sigmas=sigmas_text, scales=scales),
                latent_shape=small)

    # 5. Schedules: many steps, few steps, EPS (sigmas > 1), partial denoise, one step, coinciding transitions.
    schedules = [("flow", 6.0, "simple", 20), ("flow", 1.15, "simple", 4), ("eps", None, "karras", 10),
                 ("flow", 6.0, "simple", 8, 3), ("flow", 6.0, "simple", 1), ("flow", 6.0, "simple", 2),
                 ("flow", 3.0, "beta", 12)]
    for sched in schedules:
        for transform in ("dct", "dwt", "fft"):
            for mode, manual in (("delta_optimal", "0.85"), ("manual", "0.95,0.85"), ("manual", "0.9,0.3")):
                for base in ("euler", "euler_ancestral"):
                    yield "schedules", f"{sched} {transform} {mode} {manual} {base}", dict(
                        widgets=base_widgets(base_sampler=base, transform=transform, mode=mode, scales="0.25,0.5,1.0",
                                             manual_sigmas=manual),
                        latent_shape=(1, 4, 1, 40, 32), sched=sched, denoise_from_latent=len(sched) > 4)

    # 6. Every base solver in the combo, graph settings.
    for base in ref_cls.INPUT_TYPES()["required"]["base_sampler"][0]:
        for transform in ("dwt", "dct"):
            yield "solvers", f"{base} {transform}", dict(
                widgets=base_widgets(base_sampler=base, transform=transform), latent_shape=(1, 4, 1, 32, 24))

    # 7. Latent dtypes (the spectral maths round-trips through float32 numpy).
    for dtype in ("float16", "bfloat16", "float64"):
        for transform in ("dct", "dwt", "fft"):
            yield "dtypes", f"{dtype} {transform}", dict(
                widgets=base_widgets(transform=transform), latent_shape=(1, 16, 1, 64, 40), dtype=dtype)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", required=True)
    parser.add_argument("--original", required=True, help="public-aiorbust-nodes-pack checkout")
    parser.add_argument("--ours", default=os.path.join(REPO, "ember_nodes"))
    parser.add_argument("--out")
    parser.add_argument("--fail-fast", action="store_true")
    parser.add_argument("--threads", type=int, default=2)
    args = parser.parse_args()

    import numpy as np
    import scipy
    import torch
    torch.set_num_threads(args.threads)
    bootstrap(args.comfyui)
    import pywt

    ref = load_module("aiorbust_ref", os.path.join(args.original, "nodes"), "aiorbust_speed_hd_sampler")
    ours = load_module("ember_ref", args.ours, "speed_hd_sampler")
    ref_cls, our_cls = ref.AiorbustSpeedHDSampler, ours.EmberSpeedHDSampler

    started = time.time()
    groups = {}
    failures = []
    facts = {}
    for group, label, work in cases(ref_cls, our_cls):
        tally = groups.setdefault(group, {"checks": 0, "mismatches": 0, "floor_mismatches": 0, "errors_matched": 0})
        if callable(work):
            a, b = work(ref_cls), work(our_cls)
            floor = work(ref_cls)
        else:
            a, b, floor = run(ref_cls, **work), run(our_cls, **work), run(ref_cls, **work)
            if "error" in a and same(a["error"], b.get("error")) is None:
                tally["errors_matched"] += 1
            if group == "graph_3933479f" and "log" in a:
                solver_calls = [entry for entry in a["log"] if entry[0] == "solver"]
                facts[label] = {
                    "solver_segments": [(tuple(e[1].shape), [round(float(s), 6) for s in e[2]]) for e in solver_calls],
                    "output_shape": tuple(a["out"].shape) if "out" in a else None,
                    "error": a.get("error"),
                }
        tally["checks"] += 1
        found = same(a, b)
        floor_found = same(a, floor)
        if floor_found:
            tally["floor_mismatches"] += 1
            failures.append({"group": group, "case": label, "floor": floor_found})
        if found:
            tally["mismatches"] += 1
            failures.append({"group": group, "case": label, "diff": found})
            if args.fail_fast:
                break
        if group == "interface" and label.startswith("saved V3.0") and not all(a):
            failures.append({"group": group, "case": label, "diff": "reference rejects the saved values"})
            tally["mismatches"] += 1

    # A control that cannot fail proves nothing: the graph case must actually grow the latent.
    graph = facts.get("API widgets, 1600x2784 latent, shift 6 simple 8")
    if graph is not None:
        shapes = [seg[0] for seg in graph["solver_segments"]]
        if shapes != [(1, 16, 1, 174, 100), (1, 16, 1, 348, 200)]:
            failures.append({"group": "graph_3933479f", "case": "control", "diff": f"no transition seen: {shapes}"})

    total = sum(g["checks"] for g in groups.values())
    bad = sum(g["mismatches"] + g["floor_mismatches"] for g in groups.values())
    report = {
        "result": "PASS" if not failures else "FAIL",
        "checks": total,
        "mismatches": sum(g["mismatches"] for g in groups.values()),
        "floor_mismatches": sum(g["floor_mismatches"] for g in groups.values()),
        "groups": groups,
        "graph_facts": facts,
        "failures": failures[:20],
        "seconds": round(time.time() - started, 1),
        "environment": {
            "python": platform.python_version(), "platform": platform.platform(),
            "torch": torch.__version__, "numpy": np.__version__, "scipy": scipy.__version__,
            "pywavelets": pywt.__version__,
            "comfyui": open(os.path.join(args.comfyui, "comfyui_version.py")).read().split('"')[-2],
            "reference_commit": subprocess.run(["git", "-C", args.original, "rev-parse", "HEAD"],
                                               capture_output=True, text=True).stdout.strip(),
            "reference_sha256": {name: sha256(os.path.join(args.original, "nodes", name)) for name in
                                 ("aiorbust_speed_hd_sampler.py", "speed_hd_core.py", "speed_hd_spectral_utils.py")},
            "ours_sha256": sha256(os.path.join(args.ours, "speed_hd_sampler.py")),
        },
    }
    text = json.dumps(report, indent=2, default=str)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text + "\n")
    print(json.dumps({k: report[k] for k in ("result", "checks", "mismatches", "floor_mismatches", "seconds")}))
    if failures:
        print(json.dumps(failures[:5], indent=2, default=str))
    sys.exit(0 if not failures and bad == 0 else 1)


if __name__ == "__main__":
    main()
