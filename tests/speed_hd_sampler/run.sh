#!/usr/bin/env bash
# Ember Speed HD Sampler vs Aiorbust Speed HD Sampler on CPU (zero tolerance), then planted mutants.
#
#   tests/speed_hd_sampler/run.sh <ComfyUI dir> <public-aiorbust-nodes-pack dir> [--mutants]
#
# PYTHON (default python3) needs ComfyUI's own requirements plus PyWavelets. Results go to
# $OUT (default tests/speed_hd_sampler/results/). Exit code is non-zero on any mismatch or missed mutant.
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
py="${PYTHON:-python3}"
out="${OUT:-$here/results}"
mkdir -p "$out"
export PYTHONDONTWRITEBYTECODE=1 OMP_NUM_THREADS="${THREADS:-2}"
"$py" "$here/check.py" --comfyui "$1" --original "$2" --threads "${THREADS:-2}" --out "$out/cpu.json"
if [ "${3:-}" = "--mutants" ]; then
  "$py" "$here/mutants.py" --comfyui "$1" --original "$2" --out "$out/mutants.json"
fi
