#!/usr/bin/env bash
# Launch the fiducial channel maps (runs/channel_maps.py): Detectors 1 to 6, or the detectors given
# (channel_maps.sh 5 6). Edit runs/channel_maps.py to change it. --dry-run prints the plan.
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
exec "${HERE}/../run.sh" "${HERE}/../runs/channel_maps.py" "$@"
