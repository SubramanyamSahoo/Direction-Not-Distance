#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"

python3 -m venv --system-site-packages .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
python -m pip install --no-cache-dir -r requirements_pcie.txt
python verify_python_stack.py
python verify_protocol.py
python verify_h100_pcie.py
python verify_method_math.py
python -m pip freeze | sort > environment.lock.txt

echo "SETUP COMPLETE"
echo "Activate later with: source .venv/bin/activate"
