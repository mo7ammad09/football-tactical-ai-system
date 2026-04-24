#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
SRC_DIR="${ROOT_DIR}/src"

for target in "${ROOT_DIR}/runpod/src" "${ROOT_DIR}/replicate_abdullah/src"; do
  if [[ -d "${target}" ]]; then
    echo "Syncing ${SRC_DIR} -> ${target}"
    rsync -a --delete --exclude='__pycache__' "${SRC_DIR}/" "${target}/"
  else
    echo "Skipping missing target: ${target}"
  fi
done

echo "Done."
