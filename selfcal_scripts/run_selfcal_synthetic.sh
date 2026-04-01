#!/bin/bash

OUTPUT_DIR="${1:-/data3/caoye/selfcal/outputs}"
RUN_NAME="${2:-nep_det2_6p2arcsec}"
NODE=0

# Create the logs directory if it doesn't already exist
mkdir -p selfcal_logs

echo "Starting SelfCal Pipeline..."
echo "Output Directory: $OUTPUT_DIR"
echo "Run Name: $RUN_NAME"

# Execute the Python script with numactl
numactl --cpunodebind=$NODE --membind=$NODE python run_selfcal_synthetic.py \
    --output_dir "$OUTPUT_DIR" \
    --run_name "$RUN_NAME" \
    >> selfcal_logs/run_selfcal_synthetic_output.txt

echo "Pipeline finished. Check selfcal_logs/run_selfcal_synthetic_output.txt for details."