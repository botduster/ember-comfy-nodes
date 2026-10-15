#!/usr/bin/env bash
# Spec tests + planted mutants for the four Krea 2 carousel nodes (no ComfyUI needed, only torch).
#
#   PYTHON=/path/to/comfyui/python tests/carousel_nodes/run.sh [--mutants]
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
py="${PYTHON:-python3}"
export PYTHONDONTWRITEBYTECODE=1
"$py" "$here/test_carousel_nodes.py"
if [ "${1:-}" = "--mutants" ]; then
  "$py" "$here/mutants.py" --out "$here/results/mutants.json"
fi
