#!/usr/bin/env python3
"""Expected preview lines, computed by the Python node itself.

Writes one JSON object per line: {"mp", "aspect_ratio", "multiple_of", "expected"}.
width/height come from EmberResolutionMP.compute(); the line uses the reference
node's own format string ("{w} x {h}  ·  {mp:.3f} MP ({drift:+.1f}%)  ·  {w/h:.4f}").

    python3 reference.py > cases.jsonl
"""
import contextlib
import importlib.util
import io
import json
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("ember_resolution_mp", os.path.join(HERE, "..", "..", "ember_nodes", "resolution_mp.py"))
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
node = module.EmberResolutionMP()

LABELS = node.INPUT_TYPES()["required"]["aspect_ratio"][0]
MULTIPLES = (1, 3, 8, 16, 32, 64, 100, 128)


def expected(mp, label, multiple):
    if label == module.FROM_IMAGE:
        return "from image — computed at run time"
    with contextlib.redirect_stdout(io.StringIO()):
        w, h, _ = node.compute(mp, label, multiple)
    actual = (w * h) / 1_000_000.0
    drift = (actual - mp) / mp * 100.0 if mp else 0.0
    return f"{w} x {h}  ·  {actual:.3f} MP ({drift:+.1f}%)  ·  {w / h:.4f}"


out = sys.stdout
for multiple in MULTIPLES:
    for label in LABELS:
        seen = set()
        for i in range(5, 801):
            # Both spellings of "i hundredths" a frontend can produce; they are not always the same double.
            for mp in (i / 100, i * 0.01):
                if mp in seen:
                    continue
                seen.add(mp)
                out.write(json.dumps({"mp": mp, "aspect_ratio": label, "multiple_of": multiple,
                                      "expected": expected(mp, label, multiple)}, ensure_ascii=False) + "\n")
