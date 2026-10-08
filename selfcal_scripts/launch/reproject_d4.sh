#!/usr/bin/env bash
# Launch the 'reproject_d4' run. Edit runs/reproject_d4.py to change it. --dry-run prints the plan.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../runs/reproject_d4.py" "$@"
