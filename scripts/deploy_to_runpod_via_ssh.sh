#!/usr/bin/env bash
set -euo pipefail

# Required env vars:
# RUNPOD_HOST   -> Pod public IP/hostname
# RUNPOD_PORT   -> SSH port shown in RunPod
# REPO_URL      -> Git repository URL
# Optional:
# RUNPOD_USER   -> default: root
# SSH_KEY_PATH  -> default: ~/.ssh/id_ed25519
# PROJECT_DIR   -> default: /workspace/Football-Tactical-AI-System

if [[ -z "${RUNPOD_HOST:-}" || -z "${RUNPOD_PORT:-}" || -z "${REPO_URL:-}" ]]; then
  echo "[ERROR] Missing required vars."
  echo "Usage:"
  echo "RUNPOD_HOST=<host> RUNPOD_PORT=<port> REPO_URL=<git-url> bash scripts/deploy_to_runpod_via_ssh.sh"
  exit 1
fi

RUNPOD_USER="${RUNPOD_USER:-root}"
SSH_KEY_PATH="${SSH_KEY_PATH:-$HOME/.ssh/id_ed25519}"
PROJECT_DIR="${PROJECT_DIR:-/workspace/Football-Tactical-AI-System}"

SSH_BASE=(ssh -i "$SSH_KEY_PATH" -o StrictHostKeyChecking=accept-new -p "$RUNPOD_PORT" "${RUNPOD_USER}@${RUNPOD_HOST}")

echo "[1/4] Checking SSH connection..."
"${SSH_BASE[@]}" "echo connected"

echo "[2/4] Installing base tools on pod..."
"${SSH_BASE[@]}" "apt-get update && apt-get install -y git python3-venv"

echo "[3/4] Cloning or updating project..."
"${SSH_BASE[@]}" "if [ -d '$PROJECT_DIR/.git' ]; then cd '$PROJECT_DIR' && git pull; else git clone '$REPO_URL' '$PROJECT_DIR'; fi"

echo "[4/4] Starting GPU API server in background..."
"${SSH_BASE[@]}" "cd '$PROJECT_DIR' && nohup bash scripts/start_gpu_server.sh > /workspace/football_server.log 2>&1 & sleep 3 && tail -n 30 /workspace/football_server.log || true"

echo "Done. Next: test from your machine:"
echo "curl http://${RUNPOD_HOST}:8000/health"
