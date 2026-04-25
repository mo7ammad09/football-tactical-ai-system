#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DOCKERHUB_USER:-}" ]]; then
  echo "[ERROR] Set DOCKERHUB_USER first"
  exit 1
fi

IMAGE_NAME="${IMAGE_NAME:-football-tactical-ai}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_REF="${DOCKERHUB_USER}/${IMAGE_NAME}:${IMAGE_TAG}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

echo "[1/3] Building image: ${IMAGE_REF}"
docker build --platform linux/amd64 -f runpod/Dockerfile -t "${IMAGE_REF}" .

echo "[2/3] Pushing image"
docker push "${IMAGE_REF}"

echo "[3/3] Done"
echo "Use this image in RunPod endpoint: ${IMAGE_REF}"
