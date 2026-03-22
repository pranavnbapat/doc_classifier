#!/bin/sh
set -eu

REGISTRY="${REGISTRY:-ghcr.io/pranavnbapat}"
APP_IMAGE="${APP_IMAGE:-$REGISTRY/ko-classifier}"
AGRI_EMBEDDING_MODEL="${AGRI_EMBEDDING_MODEL:-intfloat/multilingual-e5-small}"
TAG="${1:-latest}"

echo "Building app image: ${APP_IMAGE}:${TAG}"
echo "Embedding model: ${AGRI_EMBEDDING_MODEL}"
docker build \
  --build-arg AGRI_EMBEDDING_MODEL="${AGRI_EMBEDDING_MODEL}" \
  -t "${APP_IMAGE}:${TAG}" .

echo "Pushing app image: ${APP_IMAGE}:${TAG}"
docker push "${APP_IMAGE}:${TAG}"

echo "Done."
echo "App image: ${APP_IMAGE}:${TAG}"
