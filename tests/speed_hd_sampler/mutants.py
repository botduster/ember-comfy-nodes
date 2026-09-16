#!/usr/bin/env python3
"""Plant single-point bugs in a copy of ember_nodes/speed_hd_sampler.py; check.py must catch every one.

A test that cannot fail proves nothing. Each mutant is one textual edit (the anchor must occur
exactly once, or the mutant is reported as broken rather than silently skipped). Each runs in a
fresh interpreter from its own directory with bytecode writing off, so no stale .pyc can stand in
for the edit. An unmutated copy runs first through the same path and must pass.

    python3 mutants.py --comfyui <ComfyUI> --original <public-aiorbust-nodes-pack> [--out mutants.json]
"""

import argparse
import json
import os
import shutil
import subprocess
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
SOURCE = os.path.abspath(os.path.join(HERE, "..", "..", "ember_nodes", "speed_hd_sampler.py"))

MUTANTS = [
    ("seed offset: k*10000 instead of (k+1)*10000",
     "seed + (k + 1) * 10000", "seed + k * 10000"),
    ("scales order: transitions zip (to, from) instead of (from, to)",
     "zip(scales[:-1], scales[1:], thresholds)", "zip(scales[1:], scales[:-1], thresholds)"),
    ("delta off by one step: transition one step late",
     "next((j for j in range(steps)", "next((j + 1 for j in range(steps)"),
    ("aligned sigma not written back",
     "sigmas[step] = float(aligned)", "sigmas[step] = float(sigmas[step])"),
    ("kappa uses (1 - t)",
     "return r / (1.0 + (r - 1.0) * t)", "return r / (1.0 + (r - 1.0) * (1.0 - t))"),
    ("nyquist from the longer side",
     "nyquist = min(height, width) / 2.0", "nyquist = max(height, width) / 2.0"),
    ("FFT noise scaled by math.sqrt (keeps complex64) instead of np.sqrt",
     "/ np.sqrt(2.0))", "/ math.sqrt(2.0))"),
    ("DWT detail bands swapped (LH <-> HL)",
     "[approx, (horizontal, vertical, diagonal)]", "[approx, (vertical, horizontal, diagonal)]"),
    ("DCT noise left in float64",
     "coeffs = t * rng.standard_normal((height, width)).astype(np.float32)",
     "coeffs = t * rng.standard_normal((height, width))"),
    ("grown size truncates instead of round()",
     "height, width = round(scale_to * full_h), round(scale_to * full_w)",
     "height, width = int(scale_to * full_h), int(scale_to * full_w)"),
    ("initial shrink truncates instead of round()",
     "height, width = round(x.shape[-2] * scale), round(x.shape[-1] * scale)",
     "height, width = int(x.shape[-2] * scale), int(x.shape[-1] * scale)"),
    ("video fold without the frame/channel permute",
     "return x.permute(0, 2, 1, 3, 4).reshape(b * frames, c, h, w)", "return x.reshape(b * frames, c, h, w)"),
    ("progress callback not offset per segment",
     "callback=_shift_callback(callback, start)", "callback=_shift_callback(callback, 0)"),
    ("NaN scale rejected (reference lets it through)",
     "any(s <= 0.0 or s > 1.0 for s in scales)", "any(not (0.0 < s <= 1.0) for s in scales)"),
    ("manual_sigmas length check allows extra values",
     "if len(thresholds) != len(scales) - 1:", "if len(thresholds) < len(scales) - 1:"),
    ("wan21 preset beta rounded",
     '"wan21": (219.484718, 2.422687)', '"wan21": (219.484718, 2.42269)'),
    ("seed widget max 2**32-1",
     '"default": 0, "min": 0, "max": 2**31 - 1, "step": 1', '"default": 0, "min": 0, "max": 2**32 - 1, "step": 1'),
    ("lcm offered as a base sampler",
     'UNSUPPORTED_SOLVERS = {"dpm_fast", "dpm_adaptive", "lcm"}', 'UNSUPPORTED_SOLVERS = {"dpm_fast", "dpm_adaptive"}'),
]


def run_copy(label, text, args):
    workdir = tempfile.mkdtemp(prefix="speedhd-mutant-")
    try:
        with open(os.path.join(workdir, "speed_hd_sampler.py"), "w") as fh:
            fh.write(text)
        env = dict(os.environ, PYTHONDONTWRITEBYTECODE="1")
        proc = subprocess.run([sys.executable, os.path.join(HERE, "check.py"), "--comfyui", args.comfyui,
                               "--original", args.original, "--ours", workdir, "--fail-fast"],
                              capture_output=True, text=True, env=env)
        summary = None
        for line in proc.stdout.splitlines():
            if line.startswith("{"):
                summary = json.loads(line)
                break
        detail = None
        tail = proc.stdout[proc.stdout.find("["):] if "[" in proc.stdout else ""
        if tail:
            try:
                detail = json.loads(tail)[0]
            except ValueError:
                detail = None
        return {"label": label, "exit": proc.returncode, "summary": summary, "first_failure": detail,
                "stderr_tail": proc.stderr[-400:] if summary is None else None}
    finally:
        shutil.rmtree(workdir, ignore_errors=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--comfyui", required=True)
    parser.add_argument("--original", required=True)
    parser.add_argument("--out")
    args = parser.parse_args()

    source = open(SOURCE).read()
    control = run_copy("unmutated copy", source, args)
    control_ok = control["exit"] == 0 and control["summary"] and control["summary"]["result"] == "PASS"
    print(f"control (unmutated copy): {'PASS' if control_ok else 'FAIL'}", flush=True)

    results = []
    for label, old, new in MUTANTS:
        count = source.count(old)
        if count != 1:
            results.append({"label": label, "status": f"BROKEN (anchor found {count} times)"})
            print(f"BROKEN   {label}: anchor found {count} times", flush=True)
            continue
        outcome = run_copy(label, source.replace(old, new), args)
        summary = outcome["summary"]
        caught = outcome["exit"] != 0 and summary is not None and summary["result"] == "FAIL" \
            and summary["mismatches"] > 0
        outcome["status"] = "CAUGHT" if caught else ("HARNESS ERROR" if summary is None else "MISSED")
        results.append(outcome)
        where = (outcome.get("first_failure") or {})
        print(f"{outcome['status']:<8} {label}  <- {where.get('group')}: {where.get('case')}", flush=True)

    caught = sum(1 for r in results if r.get("status") == "CAUGHT")
    report = {"control_pass": bool(control_ok), "mutants": len(MUTANTS), "caught": caught, "results": results}
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(json.dumps(report, indent=2, default=str) + "\n")
    print(json.dumps({"control_pass": report["control_pass"], "mutants": len(MUTANTS), "caught": caught}))
    sys.exit(0 if control_ok and caught == len(MUTANTS) else 1)


if __name__ == "__main__":
    main()
