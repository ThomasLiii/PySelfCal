#!/usr/bin/env bash
# Launch the 'tiled_nep' run. Edit runs/tiled_nep.py to change it (the TOML form, configs/tiled_nep.toml, still runs:
# ../run.sh ../configs/tiled_nep.toml). --dry-run prints the plan.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../runs/tiled_nep.py" "$@"
