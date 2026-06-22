#!/usr/bin/env bash
# Launch the 'tiled_nep' run. Edit configs/tiled_nep.toml to change knobs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../configs/tiled_nep.toml" "$@"
