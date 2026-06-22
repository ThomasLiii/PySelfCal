#!/usr/bin/env bash
# Generic launcher: ./selfcal_scripts/run.sh <config.toml> [--dry-run]
#
# Resolves the repo root from this script's location, puts it on PYTHONPATH so
# `python -m selfcal_scripts.run` imports both `selfcal` (installed editable) and
# `selfcal_scripts` (the operational layer), and forwards all args. Thread envs
# are pinned inside run.py before numpy import.
set -euo pipefail

if [[ $# -lt 1 ]]; then
    echo "usage: $0 <config.toml> [--dry-run]" >&2
    exit 2
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONFIG="$1"; shift

export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
exec python -u -m selfcal_scripts.run --config "${CONFIG}" "$@"
