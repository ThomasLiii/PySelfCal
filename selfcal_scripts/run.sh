#!/usr/bin/env bash
# Generic launcher: ./selfcal_scripts/run.sh <run.py> [--dry-run] [args...]
#
# A Python run script (selfcal_scripts/runs/*.py) runs through `python -m selfcal run`
# (`--dry-run`: `python -m selfcal plan`, its plan, nothing run). Resolves the repo root from
# this script's location and puts it on PYTHONPATH so both `selfcal` and `selfcal_scripts`
# import. The thread envs are pinned before numpy is imported.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <run.py> [--dry-run] [args...]" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SCRIPT="$1"; shift

if [[ "${SCRIPT}" == *.toml ]]; then
    echo "TOML configs are no longer run; convert it: selfcal convert ${SCRIPT}" >&2
    exit 2
fi
export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
for arg in "$@"; do                      # --dry-run anywhere: the plan, nothing run
    if [[ "${arg}" == "--dry-run" ]]; then
        exec python -u -m selfcal plan "${SCRIPT}"
    fi
done
exec python -u -m selfcal run "${SCRIPT}" "$@"
