#!/usr/bin/env bash
# Launch the 'damp0p5' run. Edit runs/damp0p5.py to change it (the TOML form, configs/damp0p5.toml, still runs:
# ../run.sh ../configs/damp0p5.toml). --dry-run prints the plan.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../runs/damp0p5.py" "$@"
