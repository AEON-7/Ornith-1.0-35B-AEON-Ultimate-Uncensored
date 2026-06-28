#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
IMAGE_TAG="${IMAGE_TAG:-aeon-vllm-ornith-dflash-kvfix:local}"
BASE_IMAGE="${BASE_IMAGE:-ghcr.io/aeon-7/aeon-vllm-ultimate:latest}"

docker build \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  -f "${ROOT}/Dockerfile" \
  -t "${IMAGE_TAG}" \
  "${ROOT}"

echo "Built ${IMAGE_TAG} from ${BASE_IMAGE}"
