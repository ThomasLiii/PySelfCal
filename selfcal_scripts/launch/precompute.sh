#!/usr/bin/env bash
# Launch the 'precompute' run. Edit runs/precompute.py to change it. --dry-run prints the plan.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../runs/precompute.py" "$@"
