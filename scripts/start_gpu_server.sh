#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
REQ_HASH_FILE=".venv/.server_requirements.sha256"
CURRENT_REQ_HASH="$(sha256sum server/requirements.txt | awk '{print $1}')"
SAVED_REQ_HASH=""
if [[ -f "$REQ_HASH_FILE" ]]; then
  SAVED_REQ_HASH="$(cat "$REQ_HASH_FILE")"
fi

if [[ "$CURRENT_REQ_HASH" != "$SAVED_REQ_HASH" ]]; then
  pip install -q --upgrade pip
  pip install -q -r server/requirements.txt
  echo "$CURRENT_REQ_HASH" > "$REQ_HASH_FILE"
fi

# RunPod images can miss shared libs required by OpenCV (cv2) and ffmpeg.
# Install once if needed and if apt-get is available.
if command -v apt-get >/dev/null 2>&1; then
  packages=()

  if ! python -c "import cv2" >/dev/null 2>&1; then
    echo "Installing missing system libraries for OpenCV..."
    packages+=(
      libgl1
      libglib2.0-0
      libsm6
      libxrender1
      libxext6
      libxcb1
    )
  fi

  if ! command -v ffmpeg >/dev/null 2>&1; then
    echo "Installing ffmpeg for broader video codec support..."
    packages+=(ffmpeg)
  fi

  if [[ ${#packages[@]} -gt 0 ]]; then
    apt-get update -y
    apt-get install -y "${packages[@]}"
  fi
fi

mkdir -p uploads outputs stubs uploaded_models

# You can override MODEL_PATH when running this script.
# Example: MODEL_PATH=models/abdullah_yolov5.pt bash scripts/start_gpu_server.sh
export MODEL_PATH="${MODEL_PATH:-models/abdullah_yolov5.pt}"
export YOLO_CONFIG_DIR="${YOLO_CONFIG_DIR:-/tmp/Ultralytics}"

uvicorn server.main:app --host 0.0.0.0 --port 8000
