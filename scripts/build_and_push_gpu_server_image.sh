#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DOCKERHUB_USER:-}" ]]; then
  echo "[ERROR] Set DOCKERHUB_USER first"
  echo "Example: DOCKERHUB_USER=myuser bash scripts/build_and_push_gpu_server_image.sh"
  exit 1
fi

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

IMAGE_NAME="${IMAGE_NAME:-football-tactical-ai-server}"
IMAGE_TAG="${IMAGE_TAG:-latest}"
IMAGE_REF="${DOCKERHUB_USER}/${IMAGE_NAME}:${IMAGE_TAG}"

echo "[1/3] Building ${IMAGE_REF}"
docker build -f server/Dockerfile -t "$IMAGE_REF" .

echo "[2/3] Pushing ${IMAGE_REF}"
docker push "$IMAGE_REF"

echo "[3/3] Done"
echo "Use this image in your RunPod Pod template: $IMAGE_REF"
