#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

if [[ ! -d .venv ]]; then
  python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q --upgrade pip
pip install -q -r requirements.txt

# Optional defaults for remote mode
export REMOTE_GPU_URL="${REMOTE_GPU_URL:-http://localhost:8000}"
export REMOTE_GPU_API_KEY="${REMOTE_GPU_API_KEY:-}"

streamlit run Home.py
