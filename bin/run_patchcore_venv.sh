#!/bin/bash
# venv equivalent of the Docker command:
#   docker run --rm --gpus all --user ... patchcore bin/run_patchcore.py ...
#
# Paths:
#   /data (Docker mount of ./dataset) -> ./dataset
#   /app/results (Docker mount of ./results) -> ./results

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$(readlink -f "${BASH_SOURCE[0]}")")" && cd .. && pwd)"
cd "$SCRIPT_DIR"

SITE_PACKAGES="$SCRIPT_DIR/.venv/lib/python3.10/site-packages"

env -u PYTHONHOME \
  HOME=/home2/t-hori \
  PYTHONPATH="$SCRIPT_DIR/src:$SITE_PACKAGES" \
  /usr/bin/python3.10 bin/run_patchcore.py \
  --gpu 1 --seed 0 --save_patchcore_model \
  --log_group IM224_WR50_L2-3_P01_D1024-1024_PS-3_AN-1_S0 \
  --log_project MyDataset_Results \
  ./results \
  patch_core -b wideresnet50 -le layer2 -le layer3 --faiss_on_gpu \
    --pretrain_embed_dimension 1024 --target_embed_dimension 1024 \
    --anomaly_scorer_num_nn 1 --patchsize 3 \
  sampler -p 0.1 approx_greedy_coreset \
  dataset -d anomaly_dataset --resize 256 --imagesize 224 \
    mvtec ./dataset 2>&1
