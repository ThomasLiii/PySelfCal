#!/usr/bin/env bash
# Launch the 'reproject_d4' run. Edit runs/reproject_d4.py to change it (the TOML form, configs/reproject_d4.toml, still runs:
# ../run.sh ../configs/reproject_d4.toml). --dry-run prints the plan.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../runs/reproject_d4.py" "$@"
