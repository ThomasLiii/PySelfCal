#!/usr/bin/env bash
# Launch the 'k2_readout' run. Edit configs/k2_readout.toml to change knobs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../configs/k2_readout.toml" "$@"
