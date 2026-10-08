#!/usr/bin/env bash
# Launch the 'damp_offset' run. Edit runs/damp_offset.py to change it. --dry-run prints the plan.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../runs/damp_offset.py" "$@"
