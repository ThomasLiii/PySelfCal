#!/usr/bin/env bash
# Launch the 'd4_aromatic' run. Edit configs/d4_aromatic.toml to change knobs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../configs/d4_aromatic.toml" "$@"
