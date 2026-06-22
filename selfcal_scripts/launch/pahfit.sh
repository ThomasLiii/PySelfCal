#!/usr/bin/env bash
# Launch the 'pahfit' run. Edit configs/pahfit.toml to change knobs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../configs/pahfit.toml" "$@"
