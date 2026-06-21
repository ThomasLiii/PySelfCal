#!/usr/bin/env bash
# Launch the 'd5' run. Edit configs/d5.toml to change knobs.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../configs/d5.toml" "$@"
