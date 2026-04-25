#!/usr/bin/env bash
set -euo pipefail

echo "🔨 Building Docker image..."

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT_DIR"

docker build --platform linux/amd64 -f runpod/Dockerfile -t football-analysis-runpod .

echo "✅ Build complete!"
echo "📦 Image: football-analysis-runpod"
