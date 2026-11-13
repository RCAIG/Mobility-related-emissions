#!/usr/bin/env bash
set -euo pipefail

cd ${SERVER_DATA_ROOT}/prediction_travel_mode/model/code
BANDS="$PWD/mode5_experiments/v5_grouped_baseline/selected_bands_task_aware.json"
PYTHON=/apps/miniconda3/envs/ce_trace/bin/python

MODE5_RUN_NAME=v5_lr1e4 \
MODE5_CUDA_DEVICE=0 \
MODE5_USE_CENTER=0 \
MODE5_SELECTED_BANDS_JSON="$BANDS" \
MODE5_BATCH_SIZE=48 \
MODE5_LR=0.0001 \
MODE5_MAX_EPOCHS=60 \
MODE5_PATIENCE=10 \
nohup "$PYTHON" -u model_image_landUse_centerInformation_I_v5.py \
  > mode5_v5_lr1e4.log 2>&1 &
echo $! > mode5_v5_lr1e4.pid

MODE5_RUN_NAME=v5_lr6e4 \
MODE5_CUDA_DEVICE=2 \
MODE5_USE_CENTER=0 \
MODE5_SELECTED_BANDS_JSON="$BANDS" \
MODE5_BATCH_SIZE=48 \
MODE5_LR=0.0006 \
MODE5_MAX_EPOCHS=60 \
MODE5_PATIENCE=10 \
nohup "$PYTHON" -u model_image_landUse_centerInformation_I_v5.py \
  > mode5_v5_lr6e4.log 2>&1 &
echo $! > mode5_v5_lr6e4.pid
