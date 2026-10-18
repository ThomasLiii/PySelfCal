#!/usr/bin/env bash
# Launch the single-pass multi-line PAH fit of the expanded SEP D4 field (runs/multiline_sep_expansion.py: the NEP
# expansion's recipe on eight tiles of ~3,350 frames, then the stitch). Edit that run script to change it.
# --dry-run prints the plan.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../runs/multiline_sep_expansion.py" "$@"
