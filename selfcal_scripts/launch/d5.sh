#!/usr/bin/env bash
# Launch the 'd5' run. Edit runs/d5.py to change it (the TOML form, configs/d5.toml, still runs:
# ../run.sh ../configs/d5.toml). --dry-run prints the plan.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../runs/d5.py" "$@"
