#!/usr/bin/env bash
# Launch the multi-line PAH fit of the expanded NEP D4 field (runs/multiline_nep_expansion.py: the whole field in one
# solve, the flag-split recipe in two passes). Edit that run script to change it.
# --dry-run prints the plan; --first-pass-only [--snapshots N] runs pass 1 alone, writing snapshots.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../runs/multiline_nep_expansion.py" "$@"
