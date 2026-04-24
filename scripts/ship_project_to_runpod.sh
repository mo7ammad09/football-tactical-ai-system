#!/usr/bin/env bash
set -euo pipefail

# Required:
# RUNPOD_HOST, RUNPOD_USER
# Optional:
# SSH_KEY_PATH (default ~/.ssh/id_ed25519)
# REMOTE_DIR (default /workspace/Football-Tactical-AI-System)

if [[ -z "${RUNPOD_HOST:-}" || -z "${RUNPOD_USER:-}" ]]; then
  echo "[ERROR] Set RUNPOD_HOST and RUNPOD_USER"
  echo "Example: RUNPOD_HOST=ssh.runpod.io RUNPOD_USER=<user> bash scripts/ship_project_to_runpod.sh"
  exit 1
fi

SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_ed25519}"
REMOTE_DIR="${REMOTE_DIR:-/workspace/Football-Tactical-AI-System}"
ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

SSH_BASE=(ssh -tt -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=accept-new "${RUNPOD_USER}@${RUNPOD_HOST}")


echo "[1/4] Checking SSH connection..."
"${SSH_BASE[@]}" "echo connected"

echo "[2/4] Preparing remote tools/directories..."
"${SSH_BASE[@]}" "apt-get update && apt-get install -y python3-venv && mkdir -p /workspace"

echo "[3/4] Uploading project with SCP (this may take a few minutes)..."
scp -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=accept-new -r \
  "$ROOT_DIR" "${RUNPOD_USER}@${RUNPOD_HOST}:/workspace/Football-Tactical-AI-System"

echo "[4/4] Starting GPU API server in background..."
"${SSH_BASE[@]}" "cd '$REMOTE_DIR' && nohup bash scripts/start_gpu_server.sh > /workspace/football_server.log 2>&1 < /dev/null & sleep 4 && tail -n 30 /workspace/football_server.log || true"

echo "Done."
echo "Next: find the HTTP 8000 public endpoint from RunPod (not ssh.runpod.io) and use it as REMOTE_GPU_URL."
