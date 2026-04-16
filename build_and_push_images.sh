#!/bin/sh
set -eu

REGISTRY="${REGISTRY:-ghcr.io/pranavnbapat}"
APP_IMAGE="${APP_IMAGE:-$REGISTRY/ko-classifier}"
AGRI_EMBEDDING_MODEL="${AGRI_EMBEDDING_MODEL:-intfloat/multilingual-e5-small}"
PRELOAD_AGRI_MODEL="${PRELOAD_AGRI_MODEL:-false}"
TORCH_VERSION="${TORCH_VERSION:-2.11.0}"
TORCH_INDEX_URL="${TORCH_INDEX_URL:-https://download.pytorch.org/whl/cpu}"
TAG="${1:-latest}"

echo "Building app image: ${APP_IMAGE}:${TAG}"
echo "Embedding model: ${AGRI_EMBEDDING_MODEL}"
echo "Preload agriculture model: ${PRELOAD_AGRI_MODEL}"
export DOCKER_BUILDKIT=1
docker build \
  --build-arg AGRI_EMBEDDING_MODEL="${AGRI_EMBEDDING_MODEL}" \
  --build-arg PRELOAD_AGRI_MODEL="${PRELOAD_AGRI_MODEL}" \
  --build-arg TORCH_VERSION="${TORCH_VERSION}" \
  --build-arg TORCH_INDEX_URL="${TORCH_INDEX_URL}" \
  -t "${APP_IMAGE}:${TAG}" .

echo "Pushing app image: ${APP_IMAGE}:${TAG}"
docker push "${APP_IMAGE}:${TAG}"

echo "Done."
echo "App image: ${APP_IMAGE}:${TAG}"
