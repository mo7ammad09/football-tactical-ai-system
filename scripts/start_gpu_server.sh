#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r server/requirements.txt

mkdir -p uploads outputs stubs uploaded_models

# You can override MODEL_PATH when running this script.
# Example: MODEL_PATH=models/abdullah_yolov5.pt bash scripts/start_gpu_server.sh
export MODEL_PATH="${MODEL_PATH:-models/abdullah_yolov5.pt}"

uvicorn server.main:app --host 0.0.0.0 --port 8000
