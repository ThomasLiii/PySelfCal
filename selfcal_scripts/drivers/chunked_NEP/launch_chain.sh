#!/bin/bash
# DEPRECATED: superseded by run_cal_tiled_NEP.py — TiledCalibration.run handles
# the sequential per-tile dispatch + abort-on-failure (these launch_chain
# semantics), then stitches. Kept for reference.
# Run the 4 full-dataset chunks sequentially (NW -> NE -> SW -> SE).
# Each chunk takes ~3-4 hr at iter=200 with ~4400-5200 frames (center-only filter).
# Total ETA: ~14 hr. Per-chunk peak RSS ~350-495 GB (well under 642 GB guardrail).

set -e

DRIVER_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$DRIVER_DIR/../../.." && pwd)"
LOG_DIR="${LOG_DIR:-$DRIVER_DIR/logs}"
mkdir -p "$LOG_DIR"

cd "$REPO_ROOT"
source /home/thomasli/anaconda3/etc/profile.d/conda.sh
conda activate selfcal

for chunk in NW NE SW SE; do
  echo "================================================================"
  echo "[$(date)] Starting $chunk chunk run"
  echo "================================================================"
  python -u "$DRIVER_DIR/run_cal_NEP_${chunk}.py" \
    > "$LOG_DIR/run_${chunk}.log" 2>&1
  EXIT=$?
  if [ $EXIT -ne 0 ]; then
    echo "[$(date)] $chunk EXITED WITH STATUS $EXIT -- aborting chain"
    exit $EXIT
  fi
  echo "[$(date)] $chunk chunk done; cal saved"
done

echo "================================================================"
echo "[$(date)] ALL 4 CHUNKS DONE"
echo "================================================================"
