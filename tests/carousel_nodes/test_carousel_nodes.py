#!/usr/bin/env python3
"""Spec tests for the four Krea 2 carousel nodes (clean-room, 2026-10-15).

These nodes have no reference implementation to compare against (the nodes they stand in for are
proprietary and are never run), so every check here is against Ember's own written spec: the module
docstrings and README "Krea 2 carousel nodes". Exact values are pinned where the spec defines them
(widget interface, prompt wording, the maths on constructed colours); behaviour is pinned elsewhere.

    python3 tests/carousel_nodes/test_carousel_nodes.py            # needs torch (any ComfyUI python)
    EMBER_NODES_DIR=/path/to/copy python3 ...                      # used by mutants.py

Modules are loaded by file path, so neither ComfyUI nor the pack's __init__ is imported.
"""

import importlib.util
import json
import math
import os
import sys
import unittest

import torch

HERE = os.path.dirname(os.path.abspath(__file__))
NODES_DIR = os.environ.get("EMBER_NODES_DIR") or os.path.abspath(os.path.join(HERE, "..", "..", "ember_nodes"))


def _load(name):
    spec = importlib.util.spec_from_file_location(f"ember_test_{name}", os.path.join(NODES_DIR, f"{name}.py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


kc = _load("krea2_carousel")
st = _load("skin_tone")
cl = _load("clarity")

# Independent copy of the NTSC/FCC YIQ matrix: the tests must not borrow the module's own constants.
M = torch.tensor([[0.299, 0.587, 0.114], [0.595716, -0.274453, -0.321263], [0.211456, -0.522591, 0.311135]],
                 dtype=torch.float64)
MINV = torch.linalg.inv(M)


def widgets(cls):
    out = []
    for section in ("required", "optional"):
        for name, spec in cls.INPUT_TYPES().get(section, {}).items():
            out.append((section, name, spec))
    return out


def yiq_patch(h, w, y, chroma, hue_deg):
    yiq = torch.tensor([y, chroma * math.cos(math.radians(hue_deg)), chroma * math.sin(math.radians(hue_deg))],
                       dtype=torch.float64)
    rgb = (MINV @ yiq).clamp(0, 1)
    return rgb.view(1, 1, 1, 3).expand(1, h, w, 3).clone()


def hue_chroma_y(rgb_pixel):
    yiq = M @ rgb_pixel.to(torch.float64)
    return (math.degrees(math.atan2(float(yiq[2]), float(yiq[1]))), math.hypot(float(yiq[1]), float(yiq[2])),
            float(yiq[0]))


# ===================================================================================================
# Pose List Prep
# ===================================================================================================

class TestPoseListPrep(unittest.TestCase):
    node = kc.EmberKrea2PoseListPrep()

    def test_interface(self):
        w = widgets(kc.EmberKrea2PoseListPrep)
        self.assertEqual([n for _, n, _ in w], ["pose_list", "slide_count", "if_too_few"])
        self.assertEqual(w[1][2][1], {"default": 0, "min": 0, "max": 8, "step": 1,
                                      "tooltip": w[1][2][1]["tooltip"]})
        self.assertEqual(w[2][2][0], ["error", "use what is there"])
        self.assertEqual(kc.EmberKrea2PoseListPrep.RETURN_TYPES, ("STRING", "INT", "STRING"))

    def test_lines_are_cleaned(self):
        text = ("1. Sitting on the bed,   legs crossed\n"
                "  - standing by the window \n"
                "* Standing by the WINDOW\n"
                "\n"
                "Slide 4: mirror selfie, phone in right hand;\n"
                "(5) \"leaning on the kitchen counter\"\n"
                "• lying on her stomach\n")
        poses, count, report = self.node.prepare(text, 0)
        self.assertEqual(poses.split("\n"), ["Sitting on the bed, legs crossed", "standing by the window",
                                             "mirror selfie, phone in right hand", "leaning on the kitchen counter",
                                             "lying on her stomach"])
        self.assertEqual(count, 5)
        self.assertIn("dropped duplicate: Standing by the WINDOW", report)

    def test_numbers_inside_a_pose_survive(self):
        poses, _, _ = self.node.prepare("holding 2 coffee cups\n3 steps from the door", 0)
        self.assertEqual(poses, "holding 2 coffee cups\n3 steps from the door")

    def test_json_forms(self):
        for text in ('["a", "b"]', '[{"pose": "a"}, {"description": "b"}]', '{"poses": ["a", "b"]}',
                     '{"slides": [{"text": "a"}, {"prompt": "b"}]}', '["a", null, "b", "A"]'):
            self.assertEqual(self.node.prepare(text, 0)[:2], ("a\nb", 2), text)

    def test_bad_json_shapes_raise(self):
        for text in ('{"nope": ["a"]}', '[{"x": "a"}]', '[1, 2]'):
            with self.assertRaises(ValueError, msg=text):
                self.node.prepare(text, 0)

    def test_broken_json_falls_back_to_lines(self):
        poses, count, _ = self.node.prepare('[not json\nsecond line', 0)
        self.assertEqual((poses, count), ("[not json\nsecond line", 2))

    def test_counts(self):
        five = "\n".join(f"pose {c}" for c in "abcde")
        self.assertEqual(self.node.prepare(five, 3)[:2], ("pose a\npose b\npose c", 3))
        self.assertEqual(self.node.prepare(five, 5)[1], 5)
        with self.assertRaises(ValueError):
            self.node.prepare(five, 6)
        poses, count, report = self.node.prepare(five, 7, "use what is there")
        self.assertEqual(count, 5)
        self.assertIn("only 5", report)

    def test_cap_at_eight(self):
        ten = "\n".join(f"pose number {c}" for c in "abcdefghij")
        poses, count, report = self.node.prepare(ten, 0)
        self.assertEqual(count, 8)
        self.assertEqual(poses.split("\n")[-1], "pose number h")
        self.assertIn("first 8 of 10", report)

    def test_empty_raises(self):
        for text in ("", "   \n\n - \n", "[]"):
            with self.assertRaises(ValueError, msg=repr(text)):
                self.node.prepare(text, 0)

    def test_slide_count_range(self):
        with self.assertRaises(ValueError):
            kc.select_poses(["a"], 9)
        with self.assertRaises(ValueError):
            kc.select_poses(["a"], -1)

    def test_long_pose_truncated_on_a_word(self):
        pose = " ".join(["abcdefg"] * 60)  # 479 chars; a hard cut at 300 lands inside a word
        out, _, _ = self.node.prepare(pose, 0)
        self.assertLessEqual(len(out), kc.MAX_POSE_CHARS)
        self.assertTrue(out.endswith("abcdefg"))
        self.assertGreater(len(out), kc.MAX_POSE_CHARS - 6)


# ===================================================================================================
# Carousel Director
# ===================================================================================================

POSES = "sitting on the bed, legs crossed\nmirror selfie\nchange into a different outfit\nwalks to the kitchen"


class TestCarouselDirector(unittest.TestCase):
    node = kc.EmberKrea2CarouselDirector()

    def run_node(self, **kw):
        args = dict(pose_list=POSES, slide_count=4, output_mode="all slides", slide_index=1, camera="auto",
                    continuity="balanced")
        args.update(kw)
        return dict(zip(kc.EmberKrea2CarouselDirector.RETURN_NAMES, self.node.direct(**args)))

    def test_interface(self):
        c = kc.EmberKrea2CarouselDirector
        self.assertEqual([n for _, n, _ in widgets(c)],
                         ["pose_list", "slide_count", "output_mode", "slide_index", "camera", "continuity",
                          "pose_list_in", "slide_count_in", "master_image", "character_trigger",
                          "session_direction", "extra_prompt"])
        self.assertEqual(c.RETURN_NAMES, ("edit_prompts", "negative_prompts", "selected_prompt", "manifest",
                                          "report", "slide_count", "selected_index"))
        self.assertEqual(len(c.OUTPUT_IS_LIST), len(c.RETURN_TYPES))
        self.assertEqual(c.OUTPUT_IS_LIST, (True, True, False, False, False, False, False))
        req = c.INPUT_TYPES()["required"]
        self.assertEqual(req["camera"][0], ["auto", "locked", "free"])
        self.assertEqual(req["continuity"][0], ["balanced", "strong", "maximum"])
        self.assertEqual(req["slide_count"][1]["max"], 8)

    def test_golden_prompt(self):
        out = self.run_node(pose_list="sitting on the bed, legs crossed.", slide_count=1,
                            character_trigger="  k41ajnk ", session_direction="soft morning light.",
                            extra_prompt="Never add extra limbs.")
        self.assertEqual(out["edit_prompts"], [
            "k41ajnk, Show the same person from the reference photo in a new pose. "
            "Keep her face, hair, skin tone, body shape and outfit unchanged. "
            "Pick the camera angle and framing that suit the new pose, as a natural phone photo would. "
            "New pose: sitting on the bed, legs crossed. "
            "Shoot direction: soft morning light. "
            "Never add extra limbs."])

    def test_no_trigger_no_leading_comma(self):
        p = self.run_node(slide_count=1)["edit_prompts"][0]
        self.assertTrue(p.startswith("Show the same person"))

    def test_one_prompt_per_slide_in_order(self):
        out = self.run_node()
        self.assertEqual(len(out["edit_prompts"]), 4)
        self.assertEqual(out["negative_prompts"], ["", "", "", ""])
        for p, pose in zip(out["edit_prompts"], POSES.split("\n")):
            self.assertIn(f"New pose: {pose}.", p)
        self.assertEqual(out["slide_count"], 4)
        self.assertEqual(out["selected_prompt"], out["edit_prompts"][0])

    def test_slide_count_takes_first_n(self):
        out = self.run_node(slide_count=2)
        self.assertEqual(len(out["edit_prompts"]), 2)
        self.assertIn("kept the first 2 of 4 poses", out["report"])

    def test_too_few_poses_raises(self):
        with self.assertRaises(ValueError):
            self.run_node(slide_count=5)

    def test_one_slide_mode(self):
        out = self.run_node(output_mode="one slide", slide_index=3)
        self.assertEqual(len(out["edit_prompts"]), 1)
        self.assertEqual(out["negative_prompts"], [""])
        self.assertIn("New pose: change into a different outfit.", out["edit_prompts"][0])
        self.assertEqual(out["selected_prompt"], out["edit_prompts"][0])
        self.assertEqual(out["selected_index"], 3)
        self.assertEqual(out["slide_count"], 4)
        with self.assertRaises(ValueError):
            self.run_node(output_mode="one slide", slide_index=5)

    def test_out_of_range_index_in_all_mode_selects_first(self):
        out = self.run_node(slide_count=2, slide_index=7)
        self.assertEqual(out["selected_index"], 1)
        self.assertEqual(len(out["edit_prompts"]), 2)

    def test_linked_inputs_override_widgets(self):
        out = self.run_node(pose_list="ignored widget pose", slide_count=1, pose_list_in="a\nb\nc", slide_count_in=3)
        self.assertEqual(len(out["edit_prompts"]), 3)
        self.assertIn("New pose: c.", out["edit_prompts"][2])
        with self.assertRaises(ValueError):
            self.run_node(slide_count_in=0)

    def test_continuity_levels(self):
        bal = self.run_node(slide_count=1, continuity="balanced")["edit_prompts"][0]
        strong = self.run_node(slide_count=1, continuity="strong")["edit_prompts"][0]
        mx = self.run_node(slide_count=1, continuity="maximum")["edit_prompts"][0]
        self.assertNotIn("lighting", bal)
        self.assertIn("location", strong)
        self.assertIn("lighting", strong)
        self.assertNotIn("photo style", strong)
        self.assertIn("photo style", mx)
        self.assertIn("background details", mx)
        for level in ("balanced", "strong", "maximum"):  # each level holds everything the one before does
            self.assertTrue(set(kc._LOCKS["balanced"]) <= set(kc._LOCKS[level]))
        self.assertTrue(set(kc._LOCKS["strong"]) <= set(kc._LOCKS["maximum"]))

    def test_camera_modes(self):
        texts = {c: self.run_node(slide_count=1, camera=c)["edit_prompts"][0] for c in kc.CAMERA_MODES}
        self.assertEqual(len(set(texts.values())), 3)
        self.assertIn("exactly as in the reference photo", texts["locked"])
        self.assertIn("may change", texts["free"])
        with self.assertRaises(ValueError):
            self.run_node(camera="dolly")

    def test_conflict_warnings(self):
        out = self.run_node(camera="locked", continuity="balanced")
        report = out["report"]
        self.assertIn("slide 2 asks for a camera move", report)
        self.assertIn("slide 3 asks for a different outfit", report)
        self.assertNotIn("slide 4", report)  # location is not locked at balanced
        self.assertNotIn("slide 1", report)
        strong = self.run_node(camera="auto", continuity="strong")["report"]
        self.assertIn("slide 4 asks for a different location", strong)
        self.assertNotIn("camera move", strong)  # selfie is fine when the camera is not locked

    def test_manifest(self):
        image = torch.zeros(1, 1344, 768, 3)
        out = self.run_node(master_image=image, character_trigger="trig")
        m = json.loads(out["manifest"])
        self.assertEqual(m["source_size"], {"width": 768, "height": 1344})
        self.assertEqual(m["slide_count"], 4)
        self.assertEqual([s["index"] for s in m["slides"]], [1, 2, 3, 4])
        self.assertEqual([s["prompt"] for s in m["slides"]], out["edit_prompts"])
        self.assertEqual(m["recommended"]["identity_lora"], "krea2_identity_edit_v1_2_r128.safetensors")
        self.assertEqual(m["recommended"]["max_megapixels"], 2.0)
        self.assertEqual(m["character_trigger"], "trig")
        self.assertNotIn("no master_image", out["report"])

    def test_missing_master_is_a_warning(self):
        out = self.run_node()
        self.assertIn("no master_image connected", out["report"])
        self.assertIsNone(json.loads(out["manifest"])["source_size"])

    def test_deterministic(self):
        self.assertEqual(self.run_node(camera="free"), self.run_node(camera="free"))


# ===================================================================================================
# Skin Tone
# ===================================================================================================

class TestSkinTone(unittest.TestCase):
    def test_interface(self):
        w = widgets(st.EmberSkinTone)
        self.assertEqual([(n, s[0], *((s[1].get("default"), s[1].get("min"), s[1].get("max")) if len(s) > 1 else (None,) * 3)) for _, n, s in w],
                         [("image", "IMAGE", None, None, None),
                          ("strength", "FLOAT", 0.8, 0.0, 1.5),
                          ("target_offset_deg", "FLOAT", 0.0, -20.0, 20.0),
                          ("uniformity", "FLOAT", 0.25, 0.0, 1.0),
                          ("saturation_fix", "FLOAT", 0.4, 0.0, 1.0),
                          ("detection_width_deg", "FLOAT", 35.0, 15.0, 60.0),
                          ("mask", "MASK", None, None, None)])
        self.assertEqual(w[-1][0], "optional")
        self.assertEqual(st.EmberSkinTone.RETURN_TYPES, ("IMAGE", "MASK"))

    def test_strength_zero_is_exact(self):
        img = torch.rand(2, 48, 40, 3)
        out, mask = st.EmberSkinTone().apply(img, 0.0, 0.0, 0.25, 0.4, 35.0)
        self.assertTrue(torch.equal(out, img))
        self.assertEqual(tuple(mask.shape), (2, 48, 40))

    def test_strength_zero_is_exact_float64(self):
        img = torch.cat([yiq_patch(30, 30, 0.5, 0.1, -20.0), torch.rand(1, 30, 30, 3, dtype=torch.float64)], dim=2)
        out, _ = st.apply_skin_tone(img, 0.0)
        self.assertTrue(torch.equal(out, img))

    def test_non_skin_pixels_exact_float64(self):
        img = torch.cat([yiq_patch(40, 40, 0.5, 0.1, -20.0), yiq_patch(40, 40, 0.5, 0.1, 120.0)], dim=2)
        out, mask = st.apply_skin_tone(img, 1.0)
        far = mask[0] == 0
        self.assertGreater(int(far.sum()), 500)
        self.assertTrue(torch.equal(out[0][far], img[0][far]))
        self.assertFalse(torch.equal(out[0, 20, 20], img[0, 20, 20]))

    def test_non_skin_pixels_never_clamped(self):
        # out-of-range values outside the skin (an upstream node overshooting) pass through untouched
        img = torch.cat([yiq_patch(40, 40, 0.5, 0.1, -20.0), torch.full((1, 40, 40, 3), 1.05, dtype=torch.float64)], dim=2)
        out, _ = st.apply_skin_tone(img, 1.0)
        self.assertTrue(torch.equal(out[:, :, 60:], img[:, :, 60:]))

    def test_grey_is_untouched(self):
        img = torch.linspace(0, 1, 64).view(1, 1, 64, 1).expand(1, 32, 64, 3).contiguous()
        out, mask = st.apply_skin_tone(img, 1.5, 0, 1, 1, 60)
        self.assertTrue(torch.equal(out, img))
        self.assertEqual(float(mask.max()), 0.0)

    def test_green_cast_rotated_onto_target_luma_kept(self):
        img = yiq_patch(40, 40, 0.55, 0.12, -25.0).float()
        out, mask = st.apply_skin_tone(img, 1.0, 0.0, 0.0, 0.4, 35.0)
        hue, chroma, y = hue_chroma_y(out[0, 20, 20])
        _, chroma0, y0 = hue_chroma_y(img[0, 20, 20])
        self.assertAlmostEqual(hue, 0.0, delta=0.05)
        self.assertAlmostEqual(y, y0, delta=1e-6)
        self.assertAlmostEqual(chroma, chroma0, delta=1e-5)  # uniform patch: mean chroma is its own chroma
        self.assertGreater(float(mask[0, 20, 20]), 0.3)

    def test_target_offset(self):
        img = yiq_patch(40, 40, 0.5, 0.1, -12.0)
        out, _ = st.apply_skin_tone(img, 1.0, 10.0, 0.0, 0.0, 35.0)
        self.assertAlmostEqual(hue_chroma_y(out[0, 20, 20])[0], 10.0, delta=0.05)

    def test_half_strength_moves_half_way(self):
        img = yiq_patch(40, 40, 0.5, 0.1, -16.0)
        out, mask = st.apply_skin_tone(img, 0.5, 0.0, 0.0, 0.0, 35.0)
        w = float(mask[0, 20, 20])
        hue = hue_chroma_y(out[0, 20, 20])[0]
        self.assertAlmostEqual(hue, -16.0 + 0.5 * w * 16.0, delta=0.05)  # d' = d - s*w*m, m = d
        self.assertLess(hue, -1.0)

    def test_outside_detection_width_is_exact(self):
        img = yiq_patch(32, 32, 0.5, 0.1, 90.0).float()  # far from the skin line
        out, mask = st.apply_skin_tone(img, 1.5, 0.0, 1.0, 1.0, 35.0)
        self.assertTrue(torch.equal(out, img))
        self.assertEqual(float(mask.max()), 0.0)

    def test_detection_width_controls_reach(self):
        img = yiq_patch(32, 32, 0.5, 0.1, -30.0)
        narrow = st.apply_skin_tone(img, 1.0, 0.0, 0.0, 0.0, 20.0)[1]
        wide = st.apply_skin_tone(img, 1.0, 0.0, 0.0, 0.0, 60.0)[1]
        self.assertEqual(float(narrow.max()), 0.0)
        self.assertGreater(float(wide.min()), 0.5)

    def _two_patch(self, hue_a, hue_b, chroma_a=0.1, chroma_b=0.1):
        return torch.cat([yiq_patch(60, 60, 0.5, chroma_a, hue_a), yiq_patch(60, 60, 0.5, chroma_b, hue_b)], dim=2)

    def test_uniformity(self):
        img = self._two_patch(-20.0, -10.0)
        keep = st.apply_skin_tone(img, 1.0, 0.0, 0.0, 0.0, 45.0)[0]
        flat = st.apply_skin_tone(img, 1.0, 0.0, 1.0, 0.0, 45.0)[0]
        a, b = hue_chroma_y(keep[0, 30, 30])[0], hue_chroma_y(keep[0, 30, 90])[0]
        self.assertAlmostEqual(a, -5.0, delta=0.3)   # cast removed, the 10 deg spread kept
        self.assertAlmostEqual(b, 5.0, delta=0.3)
        for x in (30, 90):
            self.assertAlmostEqual(hue_chroma_y(flat[0, 30, x])[0], 0.0, delta=0.05)

    def test_saturation_fix(self):
        img = self._two_patch(0.0, 0.0, 0.08, 0.16)
        out = st.apply_skin_tone(img, 1.0, 0.0, 0.0, 1.0, 35.0)[0]
        ca, cb = hue_chroma_y(out[0, 30, 30])[1], hue_chroma_y(out[0, 30, 90])[1]
        self.assertAlmostEqual(ca, 0.12, delta=0.002)
        self.assertAlmostEqual(cb, 0.12, delta=0.002)
        half = st.apply_skin_tone(img, 0.5, 0.0, 0.0, 1.0, 35.0)[0]   # strength scales the fix too
        self.assertAlmostEqual(hue_chroma_y(half[0, 30, 30])[1], 0.10, delta=0.002)
        self.assertAlmostEqual(hue_chroma_y(half[0, 30, 90])[1], 0.14, delta=0.002)
        none = st.apply_skin_tone(img, 1.0, 0.0, 0.0, 0.0, 35.0)[0]
        self.assertAlmostEqual(hue_chroma_y(none[0, 30, 30])[1], 0.08, delta=1e-5)

    def test_dark_and_clipped_pixels_untouched(self):
        dark = yiq_patch(16, 16, 0.02, 0.03, -20.0).float()
        out, mask = st.apply_skin_tone(dark, 1.5, 0.0, 1.0, 1.0, 60.0)
        self.assertTrue(torch.equal(out, dark))

    def test_mask_restricts(self):
        img = yiq_patch(40, 80, 0.5, 0.1, -20.0).float()
        zero = torch.zeros(1, 40, 80)
        self.assertTrue(torch.equal(st.apply_skin_tone(img, 1.0, mask=zero)[0], img))
        half = torch.zeros(40, 80)
        half[:, :40] = 1.0
        out, used = st.apply_skin_tone(img, 1.0, mask=half)
        self.assertTrue(torch.equal(out[:, :, 40:], img[:, :, 40:]))
        self.assertFalse(torch.equal(out[:, :, :40], img[:, :, :40]))
        self.assertEqual(float(used[:, :, 40:].max()), 0.0)
        small = torch.ones(1, 20, 40)  # a mask at another size is resized
        self.assertFalse(torch.equal(st.apply_skin_tone(img, 1.0, mask=small)[0], img))
        with self.assertRaises(ValueError):
            st.apply_skin_tone(img, 1.0, mask=torch.ones(3, 40, 80))

    def test_batch_items_independent(self):
        a = yiq_patch(30, 30, 0.5, 0.1, -20.0).float()
        b = yiq_patch(30, 30, 0.6, 0.12, -5.0).float()
        both = st.apply_skin_tone(torch.cat([a, b]), 1.0)[0]
        self.assertTrue(torch.equal(both[0:1], st.apply_skin_tone(a, 1.0)[0]))
        self.assertTrue(torch.equal(both[1:2], st.apply_skin_tone(b, 1.0)[0]))

    def test_alpha_dtype_range(self):
        img = torch.cat([yiq_patch(20, 20, 0.5, 0.1, -20.0).float(), torch.full((1, 20, 20, 1), 0.3)], dim=-1)
        out, mask = st.apply_skin_tone(img, 1.5, 0.0, 1.0, 1.0, 60.0)
        self.assertEqual(out.dtype, torch.float32)
        self.assertTrue(torch.equal(out[..., 3], img[..., 3]))
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)
        self.assertEqual(mask.dtype, torch.float32)
        with self.assertRaises(ValueError):
            st.apply_skin_tone(torch.rand(20, 20, 3))

    def test_mask_is_soft(self):
        # a sharp skin/grey boundary gives a feathered weight, not a hard step
        img = torch.cat([yiq_patch(50, 250, 0.5, 0.1, 0.0), torch.full((1, 50, 250, 3), 0.5, dtype=torch.float64)], dim=2)
        _, mask = st.apply_skin_tone(img, 1.0)
        row = mask[0, 25]
        self.assertGreater(float(row[250]), 0.0)   # first grey pixel still gets some weight
        self.assertLess(float(row[250]), float(row[240]))

    def test_deterministic(self):
        img = torch.rand(1, 64, 64, 3)
        self.assertTrue(torch.equal(st.apply_skin_tone(img)[0], st.apply_skin_tone(img)[0]))


# ===================================================================================================
# Clarity
# ===================================================================================================

def texture(h=96, w=128, base=0.5, amp=0.08, seed=0):
    g = torch.Generator().manual_seed(seed)
    noise = torch.rand(1, 1, h // 8, w // 8, generator=g)
    smooth = torch.nn.functional.interpolate(noise, size=(h, w), mode="bilinear", align_corners=False)[0, 0]
    lum = base + amp * (smooth - 0.5) * 2
    tint = torch.tensor([1.05, 1.0, 0.9])
    return (lum.unsqueeze(-1) * tint).clamp(0, 1).unsqueeze(0).float()


def highpass_std(img):
    lum = img[0, ..., :3].to(torch.float64) @ torch.tensor(cl._REC709, dtype=torch.float64)
    blur = torch.nn.functional.avg_pool2d(lum[None, None], 9, 1, 4, count_include_pad=False)[0, 0]
    return float((lum - blur).std())


def spec_clarity(img, amount, radius_px, eps, focus):
    """The documented algorithm, written independently of the module (float64)."""
    rgb = img[..., :3].to(torch.float64)
    L = rgb[..., 0] * 0.2126 + rgb[..., 1] * 0.7152 + rgb[..., 2] * 0.0722
    def box(x):
        return torch.nn.functional.avg_pool2d(x[:, None], 2 * radius_px + 1, 1, radius_px, count_include_pad=False)[:, 0]
    mean = box(L)
    var = box(L * L) - mean * mean
    a = var / (var + eps)
    b = mean - a * mean
    base = box(a) * L + box(b)
    w = (4 * L.clamp(0, 1) * (1 - L.clamp(0, 1))).clamp(0, 1) ** focus
    L2 = (L + amount * (L - base) * w).clamp(0, 1)
    out = torch.where((L > 1e-3)[..., None], rgb * (L2 / L.clamp_min(1e-3))[..., None], rgb + (L2 - L)[..., None])
    return out.clamp(0, 1)


class TestClarity(unittest.TestCase):
    def test_matches_written_spec(self):
        g = torch.Generator().manual_seed(5)
        for h, w, pct, want_r in ((40, 200, 5.0, 2), (100, 100, 2.5, 3), (64, 48, 10.0, 5), (30, 30, 0.2, 1)):
            img = (texture(h=h, w=w, seed=h) * 0.8 + 0.1 * torch.rand(1, h, w, 3, generator=g)).to(torch.float64)
            for amount, eps, focus in ((0.3, 0.01, 1.0), (1.7, 0.0001, 2.5), (-0.8, 0.2, 0.0)):
                got = cl.apply_clarity(img, amount, pct, eps, focus)
                want = spec_clarity(img, amount, want_r, eps, focus)
                self.assertTrue(torch.allclose(got, want, atol=1e-9), (h, w, pct, amount))

    def test_interface(self):
        w = widgets(cl.EmberClarity)
        self.assertEqual([(n, s[0], *((s[1].get("default"), s[1].get("min"), s[1].get("max")) if len(s) > 1 else (None,) * 3)) for _, n, s in w],
                         [("image", "IMAGE", None, None, None),
                          ("amount", "FLOAT", 0.35, -1.0, 2.0),
                          ("radius_pct", "FLOAT", 2.5, 0.2, 10.0),
                          ("edge_preserve", "FLOAT", 0.01, 0.0001, 0.2),
                          ("midtone_focus", "FLOAT", 1.0, 0.0, 4.0)])

    def test_amount_zero_is_exact(self):
        img = torch.rand(2, 40, 50, 3)
        self.assertTrue(torch.equal(cl.EmberClarity().apply(img, 0.0, 2.5, 0.01, 1.0)[0], img))

    def test_flat_image_is_exact(self):
        img = torch.full((1, 40, 50, 3), 0.37)
        img[..., 2] = 0.2
        self.assertTrue(torch.equal(cl.apply_clarity(img, 2.0, 5.0, 0.2, 0.0), img))

    def test_box_mean_matches_naive(self):
        g = torch.Generator().manual_seed(1)
        x = torch.rand(1, 1, 9, 13, generator=g, dtype=torch.float64)
        for r in (1, 2, 5, 20):
            got = cl._box_mean(x, r)
            want = torch.empty_like(x)
            for i in range(9):
                for j in range(13):
                    want[0, 0, i, j] = x[0, 0, max(0, i - r):i + r + 1, max(0, j - r):j + r + 1].mean()
            self.assertTrue(torch.allclose(got, want, atol=1e-12), r)

    def test_guided_filter_limits(self):
        g = torch.Generator().manual_seed(2)
        p = torch.rand(1, 1, 30, 30, generator=g, dtype=torch.float64)
        self.assertTrue(torch.allclose(cl.guided_filter_self(p, 2, 1e-12), p, atol=1e-6))   # eps -> 0: identity
        flat = cl.guided_filter_self(p, 2, 1e6)                                             # eps -> inf: box mean
        self.assertTrue(torch.allclose(flat, cl._box_mean(cl._box_mean(p, 2), 2), atol=1e-6))

    def test_positive_adds_local_contrast_negative_removes(self):
        img = texture()
        base = highpass_std(img)
        self.assertGreater(highpass_std(cl.apply_clarity(img, 1.0, 5.0, 0.01, 1.0)), base * 1.2)
        self.assertLess(highpass_std(cl.apply_clarity(img, -1.0, 5.0, 0.01, 1.0)), base * 0.9)

    def test_amount_scales_effect(self):
        img = texture()
        d1 = (cl.apply_clarity(img, 0.5, 5.0, 0.01, 1.0) - img).abs().mean()
        d2 = (cl.apply_clarity(img, 1.0, 5.0, 0.01, 1.0) - img).abs().mean()
        self.assertAlmostEqual(float(d2 / d1), 2.0, delta=0.05)

    def test_edge_preserve(self):
        # a hard edge: a small eps keeps it out of the base layer, so the boost leaves it alone
        img = torch.full((1, 64, 64, 3), 0.3)
        img[:, :, 32:] = 0.7
        small = (cl.apply_clarity(img, 1.0, 5.0, 0.0001, 0.0) - img).abs().max()
        large = (cl.apply_clarity(img, 1.0, 5.0, 0.2, 0.0) - img).abs().max()
        self.assertLess(float(small), 0.01)
        self.assertGreater(float(large), 0.05)

    def test_midtone_focus(self):
        bright = texture(base=0.93, amp=0.04)
        mid = texture(base=0.5, amp=0.04)
        def effect(img, focus):
            return float((cl.apply_clarity(img, 1.0, 5.0, 0.01, focus) - img).abs().mean())
        self.assertLess(effect(bright, 3.0), effect(bright, 0.0) * 0.4)   # highlights protected
        self.assertGreater(effect(mid, 3.0), effect(mid, 0.0) * 0.9)      # midtones barely affected

    def test_radius(self):
        self.assertEqual(max(1, int(math.floor(2.5 / 100.0 * 100 + 0.5))), 3)  # half rounds up
        img = texture()
        a = cl.apply_clarity(img, 1.0, 1.0, 0.01, 1.0)
        b = cl.apply_clarity(img, 1.0, 8.0, 0.01, 1.0)
        self.assertFalse(torch.equal(a, b))

    def test_colour_ratios_kept(self):
        img = texture()
        out = cl.apply_clarity(img, 1.0, 5.0, 0.01, 1.0)
        ok = (out[..., :3] < 0.999).all(-1) & (out[..., :3] > 0.001).all(-1)
        r_in = img[..., 0] / img[..., 1]
        r_out = out[..., 0] / out[..., 1]
        self.assertTrue(torch.allclose(r_in[ok], r_out[ok], atol=1e-5))

    def test_black_pixels_shift(self):
        img = torch.zeros(1, 32, 32, 3)
        img[:, 8:24, 8:24] = 0.6
        out = cl.apply_clarity(img, -1.0, 10.0, 0.2, 0.0)  # softening lifts black next to the square
        self.assertGreater(float(out[0, 7, 16, 0]), 0.0)
        self.assertAlmostEqual(float(out[0, 7, 16, 0]), float(out[0, 7, 16, 2]), delta=1e-6)  # neutral lift

    def test_alpha_batch_dtype_range(self):
        img = torch.cat([texture(), texture(seed=3)])
        img = torch.cat([img, torch.full(img.shape[:3] + (1,), 0.25)], dim=-1)
        out = cl.apply_clarity(img, 2.0, 3.0, 0.05, 0.5)
        self.assertTrue(torch.equal(out[..., 3], img[..., 3]))
        self.assertTrue(torch.equal(out[1:2], cl.apply_clarity(img[1:2], 2.0, 3.0, 0.05, 0.5)))
        self.assertEqual(out.dtype, torch.float32)
        self.assertGreaterEqual(float(out.min()), 0.0)
        self.assertLessEqual(float(out.max()), 1.0)
        with self.assertRaises(ValueError):
            cl.apply_clarity(torch.rand(1, 10, 10))


if __name__ == "__main__":
    result = unittest.main(exit=False, verbosity=int(os.environ.get("VERBOSITY", "1"))).result
    sys.exit(0 if result.wasSuccessful() else 1)
