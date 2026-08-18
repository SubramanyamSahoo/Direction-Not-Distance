#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
bash run_train_8b.sh
bash run_train_14b.sh
