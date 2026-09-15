#!/usr/bin/env bash
# JS preview vs Python node, whole widget grid, plus planted mutants. Needs python3 and node (>= 18).
set -euo pipefail
here="$(cd "$(dirname "$0")" && pwd)"
cases="$(mktemp)"
trap 'rm -f "$cases"' EXIT
python3 -B "$here/reference.py" > "$cases"
node "$here/check.mjs" "$cases" --mutants
