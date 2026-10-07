#!/usr/bin/env bash
# Launch the 'k2_readout' run. Edit runs/k2_readout.py to change it (the TOML form, configs/k2_readout.toml, still runs:
# ../run.sh ../configs/k2_readout.toml). --dry-run prints the plan.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../runs/k2_readout.py" "$@"
