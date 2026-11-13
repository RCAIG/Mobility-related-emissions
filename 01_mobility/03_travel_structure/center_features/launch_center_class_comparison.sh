#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${SERVER_DATA_ROOT}/prediction_travel_mode/model/code
PYTHON=/apps/miniconda3/envs/ce_trace/bin/python
BANDS=$CODE_DIR/mode5_experiments/v5_alphaearth_only/selected_bands_task_aware.json
LOG_DIR=$CODE_DIR/mode_class_comparison_logs
mkdir -p "$LOG_DIR"
cd "$CODE_DIR"

run_training() {
  local run_name="$1"
  local classes="$2"
  local use_center="$3"
  local gpu="$4"
  MODE5_RUN_NAME="$run_name" \
  MODE5_N_CLUSTERS="$classes" \
  MODE5_USE_CENTER="$use_center" \
  MODE5_REQUIRE_CENTER_MATCH=1 \
  MODE5_SELECTED_BANDS_JSON="$BANDS" \
  MODE5_LR=3e-4 \
  MODE5_CUDA_DEVICE="$gpu" \
  "$PYTHON" -u model_image_landUse_centerInformation_I_v5.py \
    > "$LOG_DIR/${run_name}.log" 2>&1
}

run_training v5_center_matched_with_center 5 1 0 &
run_training v5_center_matched_no_center 5 0 1 &
run_training v4_center_matched_with_center 4 1 2 &
run_training v4_center_matched_no_center 4 0 3 &
wait

echo "CENTER_CLASS_COMPARISON_COMPLETE"
