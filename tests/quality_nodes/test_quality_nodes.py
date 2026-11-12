#!/usr/bin/env python3
"""Spec tests for the 2026-11-12 Krea 2 quality changes (no ComfyUI needed, torch + kornia only).

  * Ember Renoise: "ISO 300" preset; optional grain_opacity. At the default (absent / 1.0) the output
    is bit-identical to the node at b284107 (the commit the Krea 2 image pins), loaded from git.
  * Ember Phone Finish: highlight clip / phone HDR / chroma 4:2:0, each off by default (input returned).
  * Ember Conditioning Cache: the encoder runs once per (key, text); a hit tells ComfyUI it needs no clip.

    python3 tests/quality_nodes/test_quality_nodes.py
"""

import importlib.util
import os
import subprocess
import sys
import tempfile
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))
NODES_DIR = os.environ.get("EMBER_NODES_DIR") or os.path.join(REPO, "ember_nodes")
PINNED = "b284107b68b7fa20c6123b3d79488cb9dce9ec6d"  # ember-comfy-nodes commit in krea2-t2i docker/node-packs.tsv


def _load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


rn = _load(os.path.join(NODES_DIR, "renoise.py"), "q_renoise")
pf = _load(os.path.join(NODES_DIR, "phone_finish.py"), "q_phone_finish")


def _pinned_renoise():
    try:
        src = subprocess.run(["git", "-C", REPO, "show", f"{PINNED}:ember_nodes/renoise.py"], check=True,
                             capture_output=True, text=True).stdout
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None
    d = tempfile.mkdtemp()
    p = os.path.join(d, "renoise_pinned.py")
    with open(p, "w") as f:
        f.write(src)
    return _load(p, "q_renoise_pinned")


def photo(h=64, w=48, seed=3):
    g = torch.Generator().manual_seed(seed)
    return torch.rand(1, h, w, 3, generator=g)


class RenoiseTests(unittest.TestCase):
    def test_iso300_preset_between_200_and_400(self):
        self.assertIn("ISO 300", rn.ISO_PRESETS)
        lo, mid, hi = rn.ISO_PRESETS["ISO 200"], rn.ISO_PRESETS["ISO 300"], rn.ISO_PRESETS["ISO 400"]
        self.assertTrue(lo[0] < mid[0] < hi[0] and lo[1] < mid[1] < hi[1])
        self.assertIn("ISO 300", rn.EmberRenoise.INPUT_TYPES()["required"]["iso_preset"][0])

    def test_existing_presets_unchanged(self):
        old = _pinned_renoise()
        if old is None:
            self.skipTest("git history not available")
        for k, v in old.ISO_PRESETS.items():
            self.assertEqual(rn.ISO_PRESETS[k], v, k)

    def test_opacity_is_an_optional_input_defaulting_to_1(self):
        spec = rn.EmberRenoise.INPUT_TYPES()["optional"]["grain_opacity"]
        self.assertEqual(spec[1]["default"], 1.0)

    def test_default_is_bit_identical_to_pinned_node(self):
        old = _pinned_renoise()
        if old is None:
            self.skipTest("git history not available")
        img = photo()
        for iso in ("ISO 200", "ISO 1600"):
            want = old.EmberRenoise().renoise(img, 1119690152049, iso, 0.5)[0]
            got_absent = rn.EmberRenoise().renoise(img, 1119690152049, iso, 0.5)[0]
            got_one = rn.EmberRenoise().renoise(img, 1119690152049, iso, 0.5, grain_opacity=1.0)[0]
            self.assertTrue(torch.equal(want, got_absent), iso)
            self.assertTrue(torch.equal(want, got_one), iso)

    def test_opacity_scales_the_noise(self):
        img = photo() * 0.5 + 0.25  # away from the clamps
        full = rn.EmberRenoise().renoise(img, 7, "ISO 1600", 1.0)[0] - img
        half = rn.EmberRenoise().renoise(img, 7, "ISO 1600", 1.0, grain_opacity=0.5)[0] - img
        zero = rn.EmberRenoise().renoise(img, 7, "ISO 1600", 1.0, grain_opacity=0.0)[0]
        self.assertTrue(torch.allclose(half, full * 0.5, atol=1e-6))
        self.assertTrue(torch.equal(zero, img))


class PhoneFinishTests(unittest.TestCase):
    node = pf.EmberPhoneFinish()

    def test_all_off_returns_the_input(self):
        img = photo()
        self.assertIs(self.node.execute(img)[0], img)
        req = pf.EmberPhoneFinish.INPUT_TYPES()["required"]
        for k in ("phone_hdr", "highlight_clip", "chroma_420"):
            self.assertIs(req[k][1]["default"], False, k)
        self.assertEqual(req["clip_level"][1]["default"], 0.97)
        self.assertEqual(req["hdr_amount"][1]["default"], 0.05)

    def test_highlight_clip(self):
        img = torch.tensor([0.97, 0.99, 0.485]).view(1, 1, 1, 3)
        out = self.node.execute(img, highlight_clip=True, clip_level=0.97)[0].flatten()
        self.assertAlmostEqual(out[0].item(), 1.0, places=6)
        self.assertAlmostEqual(out[1].item(), 1.0, places=6)
        self.assertAlmostEqual(out[2].item(), 0.5, places=6)

    def test_phone_hdr_lifts_darks_and_lowers_brights(self):
        dark = torch.full((1, 64, 64, 3), 0.1)
        bright = torch.full((1, 64, 64, 3), 0.9)
        d = self.node.execute(dark, phone_hdr=True, hdr_amount=0.05)[0]
        b = self.node.execute(bright, phone_hdr=True, hdr_amount=0.05)[0]
        self.assertTrue(torch.allclose(d, torch.full_like(d, 0.1 + 0.05 * 0.4), atol=1e-5))
        self.assertTrue(torch.allclose(b, torch.full_like(b, 0.9 - 0.05 * 0.4), atol=1e-5))

    def test_chroma_420_keeps_luma_and_halves_colour_resolution(self):
        img = photo(9, 7)  # odd sides on purpose
        out = self.node.execute(img, chroma_420=True)[0]
        self.assertEqual(out.shape, img.shape)
        luma = lambda x: x[..., 0] * 0.299 + x[..., 1] * 0.587 + x[..., 2] * 0.114  # noqa: E731
        inside = (out > 1e-4) & (out < 1 - 1e-4)  # clamped pixels may move luma
        mask = inside.all(dim=-1)
        self.assertTrue(torch.allclose(luma(out)[mask], luma(img)[mask], atol=1e-5))
        cb = lambda x: -0.168736 * x[..., 0] - 0.331264 * x[..., 1] + 0.5 * x[..., 2]  # noqa: E731
        c = cb(pf.apply_chroma_420(photo(8, 8) * 0.4 + 0.3))[0]  # mid-range: no RGB clamping
        self.assertTrue(torch.allclose(c[0::2, 0::2], c[1::2, 1::2], atol=1e-5))

    def test_flat_colour_is_untouched_by_420(self):
        img = torch.tensor([0.6, 0.3, 0.2]).view(1, 1, 1, 3).expand(1, 6, 6, 3).clone()
        self.assertTrue(torch.allclose(pf.apply_chroma_420(img), img, atol=1e-5))


class FakeClip:
    def __init__(self):
        self.calls = 0

    def tokenize(self, text):
        return {"t": text}

    def encode_from_tokens_scheduled(self, tokens):
        self.calls += 1
        return [[torch.ones(1, 4, 8) * len(tokens["t"]), {"pooled_output": torch.zeros(1, 8)}]]


class ConditioningCacheTests(unittest.TestCase):
    def setUp(self):
        self.dir = tempfile.mkdtemp()
        os.environ["EMBER_COND_CACHE_DIR"] = self.dir
        self.cc = _load(os.path.join(NODES_DIR, "conditioning_cache.py"), f"q_cc_{id(self)}")

    def test_encodes_once_then_needs_no_clip(self):
        node, clip = self.cc.EmberConditioningCache(), FakeClip()
        self.assertEqual(node.check_lazy_status("a phone photo", "k1"), ["clip"])
        first = node.encode("a phone photo", "k1", clip)[0]
        self.assertEqual(clip.calls, 1)
        self.assertEqual(node.check_lazy_status("a phone photo", "k1"), [])
        second = node.encode("a phone photo", "k1", None)[0]
        self.assertEqual(clip.calls, 1)
        self.assertTrue(torch.equal(first[0][0], second[0][0]))

    def test_survives_a_new_process_via_disk(self):
        self.cc.EmberConditioningCache().encode("x", "k", FakeClip())
        fresh = _load(os.path.join(NODES_DIR, "conditioning_cache.py"), "q_cc_fresh")
        self.assertEqual(fresh.EmberConditioningCache().check_lazy_status("x", "k"), [])
        cond = fresh.EmberConditioningCache().encode("x", "k", None)[0]
        self.assertEqual(cond[0][0].shape, (1, 4, 8))

    def test_key_or_text_change_is_a_miss(self):
        node, clip = self.cc.EmberConditioningCache(), FakeClip()
        node.encode("x", "k", clip)
        self.assertEqual(node.check_lazy_status("x", "other-lora-stack"), ["clip"])
        self.assertEqual(node.check_lazy_status("y", "k"), ["clip"])

    def test_miss_without_clip_fails_loudly(self):
        with self.assertRaises(RuntimeError):
            self.cc.EmberConditioningCache().encode("never", "k", None)

    def test_corrupt_entry_is_a_miss(self):
        path = self.cc.cache_path("k", "z")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "wb") as f:
            f.write(b"not a tensor")
        self.assertEqual(self.cc.EmberConditioningCache().check_lazy_status("z", "k"), ["clip"])


if __name__ == "__main__":
    sys.exit(0 if unittest.main(exit=False, verbosity=1).result.wasSuccessful() else 1)
