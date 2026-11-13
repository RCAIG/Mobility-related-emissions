#!/usr/bin/env bash
set -euo pipefail

CODE_DIR=${SERVER_DATA_ROOT}/prediction_travel_mode/model/code
PYTHON=/apps/miniconda3/envs/ce_trace/bin/python
SAMPLE_PER_CHUNK=${MODE_SAMPLE_PER_CHUNK:-5000}
cd "$CODE_DIR"

run_job() {
  local run_name="$1"
  local classes="$2"
  local city_id="$3"
  local year="$4"
  local income="$5"
  local gpu="$6"
  local run_dir="$CODE_DIR/mode5_experiments/$run_name"
  local log_dir="$run_dir/sampled_city_logs"
  mkdir -p "$log_dir"
  MODE5_RUN_DIR="$run_dir" \
  MODE5_USE_CENTER=1 \
  MODE5_NUM_CLASSES="$classes" \
  MODE5_CITY_ID="$city_id" \
  MODE5_YEAR="$year" \
  MODE5_INCOME="$income" \
  MODE5_CUDA_DEVICE="$gpu" \
  MODE5_SAMPLE_PER_CHUNK="$SAMPLE_PER_CHUNK" \
  MODE5_OUTPUT_SUBDIR=sampled_city_inference \
  "$PYTHON" -u infer_six_cities_v5.py \
    > "$log_dir/${city_id}_${year}.log" 2>&1
}

run_batch() {
  local run_name="$1"
  local classes="$2"
  run_job "$run_name" "$classes" 1262 2017 high_income 0 &
  run_job "$run_name" "$classes" 2335 2017 upper_middle 1 &
  run_job "$run_name" "$classes" 1186 2017 high_income 2 &
  run_job "$run_name" "$classes" 1301 2017 lower_middle 3 &
  wait

  run_job "$run_name" "$classes" 906 2017 high_income 0 &
  run_job "$run_name" "$classes" 353 2017 high_income 1 &
  run_job "$run_name" "$classes" 1262 2024 high_income 2 &
  run_job "$run_name" "$classes" 2335 2024 upper_middle 3 &
  wait

  run_job "$run_name" "$classes" 1186 2024 high_income 0 &
  run_job "$run_name" "$classes" 1301 2024 lower_middle 1 &
  run_job "$run_name" "$classes" 906 2024 high_income 2 &
  run_job "$run_name" "$classes" 353 2024 high_income 3 &
  wait
}

run_batch v5_center_matched_with_center 5
run_batch v4_center_matched_with_center 4
echo SAMPLED_CENTER_INFERENCE_COMPLETE
