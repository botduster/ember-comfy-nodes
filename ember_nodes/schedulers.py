"""Registers the `beta57` scheduler in ComfyUI's global scheduler registry.

beta57 is ComfyUI's beta schedule with alpha=0.5, beta=0.7 (core `beta` uses
0.6/0.6). Our Krea 2 workflows use it in BasicScheduler, KSampler and the eye
detailer. Today RES4LYF and the AIORBust pack each register an identical definition
at import time, so this pack registers it too and no longer depends on either.

Idempotent: if another pack already registered beta57, that registration is left
untouched.
"""

import comfy.samplers


def register_beta57():
    if "beta57" in comfy.samplers.SCHEDULER_HANDLERS:
        return False
    comfy.samplers.SCHEDULER_HANDLERS["beta57"] = comfy.samplers.SchedulerHandler(
        lambda model_sampling, steps: comfy.samplers.beta_scheduler(model_sampling, steps, alpha=0.5, beta=0.7)
    )
    comfy.samplers.SCHEDULER_NAMES.append("beta57")
    return True
