#!/usr/bin/env bash
# Launch the multi-line PAH fit of the expanded SEP D4 field (runs/multiline_sep_expansion_slope.py: the whole field in
# one solve, one pass, with a slope sky term). Edit that run script to change it.
# --dry-run prints the plan; --snapshots N writes the solution every N iterations.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../runs/multiline_sep_expansion_slope.py" "$@"
