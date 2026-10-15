#!/usr/bin/env bash
# Launch the SEP multi-line PAH fit (runs/sep_d4_npass_multiline_of.py: Detector 4, the three PAH features on
# six tiles, then 5 N-pass passes, offset first). Edit that run script to change it. --dry-run prints the plan.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../runs/sep_d4_npass_multiline_of.py" "$@"
