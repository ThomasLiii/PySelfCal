#!/usr/bin/env bash
# Launch the 'damp_offset' run. Edit configs/damp_offset.toml to change knobs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../configs/damp_offset.toml" "$@"
