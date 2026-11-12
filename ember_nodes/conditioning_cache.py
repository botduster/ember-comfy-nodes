"""Ember Conditioning Cache: encode a FIXED prompt once per worker, then reuse it.

Why: the Realism Refiner's prompt never changes, yet its ~8 GB text encoder is moved onto the GPU
on every run just to encode the same words. This node stores the encoded conditioning on disk
(and in memory) keyed by sha256(cache_key + text). `clip` is a LAZY input: when the cache already
holds the entry, ComfyUI is told it does not need `clip`, so the encode never runs and the encoder
never reaches the GPU.

`cache_key` must change whenever anything that shapes the encoding changes: the text-encoder file,
any LoRA applied to it and its strength. The caller (the Krea 2 handler) builds it from those.

Only the conditioning (a list of [tensor, dict]) is stored, with torch.save; it is read back with
torch.load(weights_only=True), so a cache file can never run code. Reference images are NOT part
of it: attach them after this node (core ReferenceLatent).
"""

import hashlib
import os
import threading

import torch

CACHE_DIR = os.environ.get("EMBER_COND_CACHE_DIR", "/tmp/ember-conditioning-cache")
_MEMORY = {}
_LOCK = threading.Lock()


def cache_path(cache_key, text):
    digest = hashlib.sha256(f"{cache_key}\0{text}".encode("utf-8")).hexdigest()[:40]
    return os.path.join(CACHE_DIR, f"{digest}.pt")


def lookup(cache_key, text):
    path = cache_path(cache_key, text)
    with _LOCK:
        if path in _MEMORY:
            return _MEMORY[path]
    if not os.path.isfile(path):
        return None
    try:
        cond = torch.load(path, map_location="cpu", weights_only=True)
    except Exception:
        return None  # unreadable entry: treated as a miss, re-encoded and overwritten
    with _LOCK:
        _MEMORY[path] = cond
    return cond


def store(cache_key, text, cond):
    path = cache_path(cache_key, text)
    os.makedirs(CACHE_DIR, exist_ok=True)
    tmp = f"{path}.{os.getpid()}.tmp"
    cpu = [[c.detach().cpu(), {k: (v.detach().cpu() if torch.is_tensor(v) else v) for k, v in extra.items()}]
           for c, extra in cond]
    torch.save(cpu, tmp)
    os.replace(tmp, path)
    with _LOCK:
        _MEMORY[path] = cpu


class EmberConditioningCache:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "text": ("STRING", {"multiline": True, "default": ""}),
                "cache_key": ("STRING", {"default": "", "tooltip": "Text encoder + LoRAs on it; change = re-encode."}),
            },
            "optional": {
                "clip": ("CLIP", {"lazy": True}),
            },
        }

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"
    CATEGORY = "Ember/Conditioning"
    DESCRIPTION = "Text conditioning cached per worker; the text encoder only runs on the first use."

    def check_lazy_status(self, text, cache_key, clip=None):
        return [] if lookup(cache_key, text) is not None else ["clip"]

    def encode(self, text, cache_key, clip=None):
        cond = lookup(cache_key, text)
        if cond is not None:
            return (cond,)
        if clip is None:
            raise RuntimeError("Ember Conditioning Cache: no cached entry and no clip connected")
        tokens = clip.tokenize(text)
        cond = clip.encode_from_tokens_scheduled(tokens)
        store(cache_key, text, cond)
        return (lookup(cache_key, text),)


NODE_CLASS_MAPPINGS = {"EmberConditioningCache": EmberConditioningCache}
NODE_DISPLAY_NAME_MAPPINGS = {"EmberConditioningCache": "Ember Conditioning Cache"}
