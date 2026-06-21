#!/usr/bin/env bash
# Launch the 'damp0p5' run. Edit configs/damp0p5.toml to change knobs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../configs/damp0p5.toml" "$@"
