#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p norm_matched_outputs/logs
python -u analyze_hidden_cancellation.py \
  2>&1 | tee -a norm_matched_outputs/logs/hidden_cancellation.log
python -u analyze_norm_matched.py \
  2>&1 | tee -a norm_matched_outputs/logs/analysis_norm_matched.log
