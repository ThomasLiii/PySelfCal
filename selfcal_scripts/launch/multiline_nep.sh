#!/usr/bin/env bash
# Launch the NEP multi-line PAH fit (runs/multiline_nep.py: Detector 4, the three PAH features on the 16
# adaptive-overlap tiles, stitched). Edit runs/multiline_nep.py to change it. --dry-run prints the plan.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../runs/multiline_nep.py" "$@"
