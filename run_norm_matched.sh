#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
export HF_HUB_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export USE_TF=0
export TRANSFORMERS_NO_TF=1
export TF_CPP_MIN_LOG_LEVEL=3
mkdir -p norm_matched_outputs/logs
python -u run_norm_matched.py \
  --source-root pci_h100_outputs \
  --root norm_matched_outputs \
  2>&1 | tee -a norm_matched_outputs/logs/train_norm_matched.log
