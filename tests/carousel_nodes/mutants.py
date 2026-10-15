#!/usr/bin/env python3
"""Plant single-point bugs in copies of the carousel nodes; test_carousel_nodes.py must catch every one.

A test that cannot fail proves nothing. Each mutant is one textual edit whose anchor must occur exactly
once (otherwise the mutant is reported as BROKEN, never silently skipped). Each runs the whole test file
in a fresh interpreter against its own copy of ember_nodes/, with bytecode writing off. The unmutated
copy runs first through the same path and must pass.

    python3 tests/carousel_nodes/mutants.py [--out results/mutants.json]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
NODES = os.path.abspath(os.path.join(HERE, "..", "..", "ember_nodes"))
TEST = os.path.join(HERE, "test_carousel_nodes.py")
FILES = ("krea2_carousel.py", "skin_tone.py", "clarity.py")

MUTANTS = [
    # ---- Pose List Prep / shared parsing
    ("krea2_carousel.py", "bullets not stripped", 'text = _BULLET.sub("", text, count=1)', "text = text"),
    ("krea2_carousel.py", "slide labels not stripped", 'text = _SLIDE_LABEL.sub("", text, count=1)', "text = text"),
    ("krea2_carousel.py", "dedupe is case-sensitive", "key = pose.casefold()", "key = pose"),
    ("krea2_carousel.py", "cap is 7 not 8", "MAX_SLIDES = 8", "MAX_SLIDES = 7"),
    ("krea2_carousel.py", "keeps the last N poses", "return poses[:want], notes", "return poses[-want:], notes"),
    ("krea2_carousel.py", "too-few error swallowed",
     'if if_too_few == "error":', 'if if_too_few == "never":'),
    ("krea2_carousel.py", "truncation not on a word", "space >= MAX_POSE_CHARS // 2", "space >= MAX_POSE_CHARS * 2"),
    ("krea2_carousel.py", "JSON 'poses' key ignored", 'for key in ("poses", "pose_list", "slides"):',
     'for key in ("pose_list", "slides"):'),
    # ---- Director
    ("krea2_carousel.py", "widget wins over linked pose list",
     "text = pose_list_in if pose_list_in is not None else pose_list", "text = pose_list"),
    ("krea2_carousel.py", "one-slide mode returns all", 'chosen = slides if output_mode == "all slides" else',
     'chosen = slides if output_mode != "x" else'),
    ("krea2_carousel.py", "negatives one short", '[""] * len(prompts)', '[""] * (len(prompts) - 1)'),
    ("krea2_carousel.py", "manifest width/height swapped",
     '{"width": int(master_image.shape[2]), "height": int(master_image.shape[1])}',
     '{"width": int(master_image.shape[1]), "height": int(master_image.shape[2])}'),
    ("krea2_carousel.py", "outfit dropped from balanced locks",
     '"balanced": ["face", "hair", "skin tone", "body shape", "outfit"],',
     '"balanced": ["face", "hair", "skin tone", "body shape"],'),
    ("krea2_carousel.py", "conflicts ignore which locks are on", "if lock in locked and pattern.search(pose):",
     "if pattern.search(pose):"),
    ("krea2_carousel.py", "camera warning regardless of mode",
     'if camera == "locked" and _CAMERA_MOVES.search(pose):', "if _CAMERA_MOVES.search(pose):"),
    ("krea2_carousel.py", "trigger not stripped", 'f"{trigger.strip()}, {lead}"', 'f"{trigger}, {lead}"'),
    ("krea2_carousel.py", "pose punctuation doubled", 'pose_text = pose.rstrip(".!? ")', "pose_text = pose"),
    ("krea2_carousel.py", "cleaning not repeated (not idempotent)", "        if text == before:\n            break",
     "        break"),
    ("krea2_carousel.py", "slide seeds not offset per slide", '"seed": (int(seed) + i - 1)', '"seed": (int(seed) + 0 * i)'),
    ("krea2_carousel.py", "list output flags wrong", "OUTPUT_IS_LIST = (True, True, False",
     "OUTPUT_IS_LIST = (True, False, False"),
    # ---- Skin Tone
    ("skin_tone.py", "YIQ I row typo", "[0.595716, -0.274453, -0.321263]", "[0.595716, -0.274453, -0.311263]"),
    ("skin_tone.py", "strength-0 shortcut removed", "if strength == 0 or float(weight.max()) <= 0.0:",
     "if float(weight.max()) <= 0.0:"),
    ("skin_tone.py", "single-pass detection (window stays on the target)",
     "second = hue_window(_wrap(err - centre), width_rad) * region", "second = first"),
    ("skin_tone.py", "uniformity applied inverted", "float(uniformity) * _wrap(err - mean_err)",
     "(1.0 - float(uniformity)) * _wrap(err - mean_err)"),
    ("skin_tone.py", "rotation sign flipped", "new_err = err - k *", "new_err = err + k *"),
    ("skin_tone.py", "saturation fix ignores strength", "float(saturation_fix) * k *",
     "float(saturation_fix) * weight *"),
    ("skin_tone.py", "chroma gate removed (greys detected)",
     "region = _smoothstep(*CHROMA_LOW, chroma) *", "region = 1.0 *"),
    ("skin_tone.py", "luma gate removed", "* _smoothstep(*LUMA_LOW, y) *", "* 1.0 *"),
    ("skin_tone.py", "mask adds instead of restricting", "weight = weight * m.clamp(0.0, 1.0)",
     "weight = torch.maximum(weight, m.clamp(0.0, 1.0))"),
    ("skin_tone.py", "no exact copy-back of untouched pixels",
     "out[..., :3] = torch.where(keep, image[..., :3], out[..., :3])", "pass"),
    ("skin_tone.py", "no mask softening", "weight = _gaussian_blur(weight, BLUR_FRACTION * min(h, w))", "pass"),
    ("skin_tone.py", "target offset ignored", "target = math.radians(float(target_offset_deg))", "target = 0.0"),
    ("skin_tone.py", "hue window has no plateau", "half = 0.5 * width_rad", "half = 0.0 * width_rad + 1e-9"),
    ("skin_tone.py", "luma not preserved (Y row touched)",
     "d_iq = torch.stack([torch.zeros_like(i), new_i - i, new_q - q], dim=-1)",
     "d_iq = torch.stack([0.01 * torch.ones_like(i), new_i - i, new_q - q], dim=-1)"),
    # ---- Clarity
    ("clarity.py", "amount-0 shortcut removed and ratio broken", "    if amount == 0:\n        return image.clone()\n",
     "    image = image * 0.999\n"),
    ("clarity.py", "box window off by one", "(ys + r + 1).clamp(0, h)", "(ys + r).clamp(0, h)"),
    ("clarity.py", "box mean divides by full window at borders",
     "area = ((y1 - y0)[:, None] * (x1 - x0)[None, :]).to(x.dtype)",
     "area = torch.full((h, w), float((2 * r + 1) ** 2), dtype=x.dtype)"),
    ("clarity.py", "guided filter eps dropped", "a = var_p / (var_p + eps)", "a = var_p / (var_p + 1e-12)"),
    ("clarity.py", "guided filter b term wrong", "b = mean_p - a * mean_p", "b = mean_p"),
    ("clarity.py", "midtone weight ignored", "weight = (4.0 * lc * (1.0 - lc)).clamp(0.0, 1.0)",
     "weight = torch.ones_like(lc)"),
    ("clarity.py", "midtone focus ignored", "weight = weight ** float(midtone_focus) if midtone_focus != 0 else torch.ones_like(weight)",
     "weight = weight"),
    ("clarity.py", "luma weights Rec.601 instead of 709", "_REC709 = (0.2126, 0.7152, 0.0722)",
     "_REC709 = (0.299, 0.587, 0.114)"),
    ("clarity.py", "additive instead of colour-preserving",
     "out_rgb = torch.where((luma > LUMA_FLOOR).unsqueeze(-1), scaled, shifted)", "out_rgb = (shifted"),
    ("clarity.py", "sign flipped", "new_luma = (luma + float(amount) * detail * weight)",
     "new_luma = (luma - float(amount) * detail * weight)"),
    ("clarity.py", "radius uses the longer side", "radius_pct) / 100.0 * min(h, w)", "radius_pct) / 100.0 * max(h, w)"),
    ("clarity.py", "radius rounds down", "min(h, w) + 0.5)))", "min(h, w))))"),
]


def run_tests(nodes_dir):
    env = dict(os.environ, EMBER_NODES_DIR=nodes_dir, PYTHONDONTWRITEBYTECODE="1", VERBOSITY="0")
    proc = subprocess.run([sys.executable, TEST], env=env, capture_output=True, text=True, timeout=600)
    tail = (proc.stderr or proc.stdout).strip().splitlines()[-1:] or [""]
    return proc.returncode, tail[0]


def copy_nodes(dst):
    for name in FILES:
        shutil.copy2(os.path.join(NODES, name), os.path.join(dst, name))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=os.path.join(HERE, "results", "mutants.json"))
    args = parser.parse_args()
    results, ok = [], True
    with tempfile.TemporaryDirectory() as tmp:
        clean = os.path.join(tmp, "clean")
        os.makedirs(clean)
        copy_nodes(clean)
        code, tail = run_tests(clean)
        print(f"unmutated copy: {'PASS' if code == 0 else 'FAIL'} ({tail})")
        if code != 0:
            sys.exit("the unmutated copy must pass first")
        for idx, (fname, desc, anchor, repl) in enumerate(MUTANTS):
            d = os.path.join(tmp, f"m{idx}")
            os.makedirs(d)
            copy_nodes(d)
            path = os.path.join(d, fname)
            src = open(path).read()
            n = src.count(anchor)
            if n != 1:
                status = f"BROKEN (anchor found {n}x)"
                ok = False
            else:
                open(path, "w").write(src.replace(anchor, repl))
                code, tail = run_tests(d)
                status = "caught" if code != 0 else "MISSED"
                ok &= code != 0
            print(f"[{status}] {fname}: {desc}")
            results.append({"file": fname, "mutant": desc, "status": status, "tail": tail if n == 1 else ""})
    caught = sum(r["status"] == "caught" for r in results)
    print(f"{caught}/{len(results)} mutants caught")
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    with open(args.out, "w") as fh:
        json.dump({"caught": caught, "total": len(results), "mutants": results}, fh, indent=2)
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
