#!/usr/bin/env bash
# Launch the NEP Detector 6 expansion reprojection (runs/reproject_nep_expansion_d6.py: the deep data through 2026W29
# and the all-sky ring, on an expanded canvas of the W17 D6 projection). Move the W17 frames onto the canvas first
# (selfcal_scripts/runs/expand_field.py). Edit the run script to change it. --dry-run prints the plan.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../runs/reproject_nep_expansion_d6.py" "$@"
