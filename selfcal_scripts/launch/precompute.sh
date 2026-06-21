#!/usr/bin/env bash
# Launch the 'precompute' run. Edit configs/precompute.toml to change knobs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../configs/precompute.toml" "$@"
