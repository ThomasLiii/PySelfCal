#!/usr/bin/env bash
# Launch the 'reproject_d4' run. Edit configs/reproject_d4.toml to change knobs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../configs/reproject_d4.toml" "$@"
