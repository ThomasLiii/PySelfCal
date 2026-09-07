#!/usr/bin/env bash
# ==========================================================================
#  SelfCal transfer-function run. Three equivalent ways to set the 6 inputs:
#
#   1) Command-line flags (any subset; the rest fall back to the defaults below):
#        ./run_transfer_function.sh \
#            --detector 3 --channel 17 \
#            --frames  /scratch/tf/D3_fakesky_frames \
#            --ref     /mnt/.../D3/ref.fits \
#            --output-dir /mnt/md124/thomasli/selfcal/outputs \
#            --run-name TF_D3
#
#   2) Env vars:  DETECTOR=3 CHANNEL=17 REPROJ_FRAME_DIR=... ./run_transfer_function.sh
#
#   3) Just edit the 6 defaults below and run  ./run_transfer_function.sh
#
#  Precedence: flag > env var > default. Everything else (the frozen fiducial
#  recipe and the ref.fits placement) is handled for you.
# ==========================================================================
DETECTOR="${DETECTOR:-3}"                 # 1..6
CHANNEL="${CHANNEL:-17}"                  # 1..34 (single LVF channel)
REPROJ_FRAME_DIR="${REPROJ_FRAME_DIR:-/scratch/tf/D3_fakesky_frames}"  # fake-sky frames
REF_FITS="${REF_FITS:-/mnt/md124/thomasli/selfcal/outputs/SPHEREx_NEP_2026W17_D3_6p2arcsec/ref.fits}"
OUTPUT_DIR="${OUTPUT_DIR:-/mnt/md124/thomasli/selfcal/outputs}"        # cal+mosaic land under here
RUN_NAME="${RUN_NAME:-TF_D3}"            # under OUTPUT_DIR; the run's folder name
# ==========================================================================

set -euo pipefail

usage() { sed -n '2,19p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit "${1:-0}"; }

# --- command-line flags override the defaults/env above ---
while [[ $# -gt 0 ]]; do
  case "$1" in
    -d|--detector)          DETECTOR="$2"; shift 2;;
    -c|--channel)           CHANNEL="$2"; shift 2;;
    -f|--frames|--reproj-frame-dir) REPROJ_FRAME_DIR="$2"; shift 2;;
    -r|--ref|--ref-fits)    REF_FITS="$2"; shift 2;;
    -o|--output-dir)        OUTPUT_DIR="$2"; shift 2;;
    -n|--run-name)          RUN_NAME="$2"; shift 2;;
    -h|--help)              usage 0;;
    *) echo "unknown option: $1" >&2; usage 1;;
  esac
done

# Run from the repo root so `import selfcal` resolves; set SELFCAL_PY to the
# selfcal env python if `python` isn't it (e.g. ~/anaconda3/envs/selfcal/bin/python).
REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PY="${SELFCAL_PY:-python}"

# The runner reads the reference grid from {OUTPUT_DIR}/{RUN_NAME}/ref.fits;
# point it at your ref.fits (a symlink — the file is not copied or modified).
mkdir -p "$OUTPUT_DIR/$RUN_NAME"
ln -sf "$REF_FITS" "$OUTPUT_DIR/$RUN_NAME/ref.fits"

# Fill the 6 inputs into the frozen fiducial config (targets keys, so the
# example values in the template don't matter).
TMPL="$(dirname "${BASH_SOURCE[0]}")/transfer_function.toml"
CFG="$(mktemp --suffix=.toml)"
sed -E \
  -e "s|^detector[[:space:]]*=.*|detector = $DETECTOR|" \
  -e "s|^channels[[:space:]]*=.*|channels = [[$CHANNEL]]|" \
  -e "s|^reproj_override[[:space:]]*=.*|reproj_override = \"$REPROJ_FRAME_DIR\"|" \
  -e "s|^output_dir[[:space:]]*=.*|output_dir = \"$OUTPUT_DIR\"|" \
  -e "s|^run_name[[:space:]]*=.*|run_name = \"$RUN_NAME\"|" \
  "$TMPL" > "$CFG"

echo "[tf] D$DETECTOR Ch$CHANNEL  frames=$REPROJ_FRAME_DIR"
echo "[tf] ref.fits -> $OUTPUT_DIR/$RUN_NAME/ref.fits  (-> $REF_FITS)"
echo "[tf] outputs  -> $OUTPUT_DIR/$RUN_NAME/{calibration,mosaic}/"

cd "$REPO"
exec "$PY" -m selfcal_scripts.run --config "$CFG"
