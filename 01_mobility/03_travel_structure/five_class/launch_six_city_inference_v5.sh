#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 ]]; then
  echo "Usage: $0 RUN_NAME" >&2
  exit 2
fi

CODE_DIR=${SERVER_DATA_ROOT}/prediction_travel_mode/model/code
PYTHON=/apps/miniconda3/envs/ce_trace/bin/python
RUN_NAME="$1"
RUN_DIR="$CODE_DIR/mode5_experiments/$RUN_NAME"
LOG_DIR="$RUN_DIR/six_city_logs"
mkdir -p "$LOG_DIR"
cd "$CODE_DIR"

run_job() {
  local city_id="$1"
  local year="$2"
  local income="$3"
  local gpu="$4"
  MODE5_RUN_DIR="$RUN_DIR" \
  MODE5_USE_CENTER=0 \
  MODE5_CITY_ID="$city_id" \
  MODE5_YEAR="$year" \
  MODE5_INCOME="$income" \
  MODE5_CUDA_DEVICE="$gpu" \
  "$PYTHON" -u infer_six_cities_v5_fast.py \
    > "$LOG_DIR/${city_id}_${year}.log" 2>&1
}

run_job 1262 2017 high_income 0 &
run_job 2335 2017 upper_middle 1 &
run_job 1186 2017 high_income 2 &
run_job 1301 2017 lower_middle 3 &
wait

run_job 906 2017 high_income 0 &
run_job 353 2017 high_income 1 &
run_job 1262 2024 high_income 2 &
run_job 2335 2024 upper_middle 3 &
wait

run_job 1186 2024 high_income 0 &
run_job 1301 2024 lower_middle 1 &
run_job 906 2024 high_income 2 &
run_job 353 2024 high_income 3 &
wait

echo "All six-city inference jobs completed for $RUN_NAME"
