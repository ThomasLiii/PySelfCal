#!/usr/bin/env bash
# Launch the 'k2_readout' run. Edit runs/k2_readout.py to change it. --dry-run prints the plan.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../runs/k2_readout.py" "$@"
