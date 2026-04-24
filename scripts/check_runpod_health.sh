#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${RUNPOD_HOST:-}" ]]; then
  echo "[ERROR] Set RUNPOD_HOST"
  echo "Example: RUNPOD_HOST=1.2.3.4 bash scripts/check_runpod_health.sh"
  exit 1
fi

curl -sS "http://${RUNPOD_HOST}:8000/health" || true
echo
