#!/usr/bin/env bash
# Launch the 'd4_aromatic' run. Edit runs/d4_aromatic.py to change it. --dry-run prints the plan.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../runs/d4_aromatic.py" "$@"
