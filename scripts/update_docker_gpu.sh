#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/update_docker_gpu.sh [OPTIONS]

Build (and optionally push) the GPU Docker image.

Options:
  --image IMAGE_URI   Image URI to tag (default derived from gcloud/env)
  --tag TAG           Image tag when deriving the default image URI
  --no-cache          Disable Docker build cache
  --cache             Enable Docker build cache (default)
  --pull              Always attempt to pull a newer base image
  --push              Push the image after building
  --no-push           Do not push (default)
  --platform PLATFORM Set target platform (e.g. linux/amd64)
  -h, --help          Show this help

Environment fallbacks for default --image:
  VERTEX_PROJECT or GCP_PROJECT or `gcloud config get-value project`
  VERTEX_REGION or `gcloud config get-value ai/region` (fallback us-central1)
  AR_REPOSITORY (default: tfm-forecasting)
  IMAGE_NAME (default: trainer-gpu)
  IMAGE_TAG (default: latest)
USAGE
}

if ! command -v docker >/dev/null 2>&1; then
  echo "Error: docker is not installed or not on PATH" >&2
  exit 1
fi

IMAGE_URI="${IMAGE_URI:-}"
IMAGE_TAG="${IMAGE_TAG:-}"
NO_CACHE=0
PULL=0
PUSH=0
PLATFORM="${DOCKER_PLATFORM:-}"

require_value() {
  local opt="$1"
  local val="${2-}"
  if [[ -z "$val" || "$val" == --* ]]; then
    echo "Error: ${opt} requires a non-empty value." >&2
    exit 2
  fi
  printf '%s\n' "$val"
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --image)
      IMAGE_URI="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --tag)
      IMAGE_TAG="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --no-cache)
      NO_CACHE=1
      shift
      ;;
    --cache)
      NO_CACHE=0
      shift
      ;;
    --pull)
      PULL=1
      shift
      ;;
    --push)
      PUSH=1
      shift
      ;;
    --no-push)
      PUSH=0
      shift
      ;;
    --platform)
      PLATFORM="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Error: unknown argument: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -n "${PLATFORM}" && "${PLATFORM}" == -* ]]; then
  echo "Error: --platform value looks like an option: ${PLATFORM}" >&2
  exit 2
fi

_gcloud_value() {
  local key="$1"
  if ! command -v gcloud >/dev/null 2>&1; then
    return 1
  fi
  local out
  out="$(gcloud config get-value "$key" 2>/dev/null || true)"
  if [[ -z "$out" || "$out" == "(unset)" ]]; then
    return 1
  fi
  printf '%s\n' "$out"
}

tag_to_latest() {
  local image_uri="$1"
  if [[ "$image_uri" == *@* ]]; then
    return 1
  fi
  if [[ "$image_uri" != *:* ]]; then
    printf '%s:latest\n' "$image_uri"
    return 0
  fi
  printf '%s:latest\n' "${image_uri%:*}"
}

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

if [[ -z "${IMAGE_URI}" ]]; then
  PROJECT="${VERTEX_PROJECT:-${GCP_PROJECT:-$(_gcloud_value project || true)}}"
  REGION="${VERTEX_REGION:-$(_gcloud_value ai/region || true)}"
  REPOSITORY="${AR_REPOSITORY:-tfm-forecasting}"
  IMAGE_NAME="${IMAGE_NAME:-trainer-gpu}"
  if [[ -z "${IMAGE_TAG}" ]]; then
    IMAGE_TAG="latest"
  fi

  if [[ -z "${PROJECT}" ]]; then
    echo "Error: could not determine project. Set --image, VERTEX_PROJECT, GCP_PROJECT, or gcloud project config." >&2
    exit 2
  fi
  if [[ -z "${REGION}" ]]; then
    REGION="us-central1"
  fi

  IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT}/${REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"
fi

DOCKERFILE="${ROOT_DIR}/Dockerfile.gpu"

if [[ ! -f "${DOCKERFILE}" ]]; then
  echo "Error: Dockerfile not found: ${DOCKERFILE}" >&2
  exit 1
fi

BUILD_OPTS=(--file "${DOCKERFILE}" --tag "${IMAGE_URI}")
if [[ "${NO_CACHE}" -eq 1 ]]; then
  BUILD_OPTS+=(--no-cache)
fi
if [[ "${PULL}" -eq 1 ]]; then
  BUILD_OPTS+=(--pull)
fi
if [[ -n "${PLATFORM}" ]]; then
  BUILD_OPTS+=(--platform "${PLATFORM}")
fi

echo "[gpu] building: ${IMAGE_URI}"
docker build "${BUILD_OPTS[@]}" "${ROOT_DIR}"

LATEST_IMAGE_URI=""
if ! LATEST_IMAGE_URI="$(tag_to_latest "${IMAGE_URI}")"; then
  echo "Error: cannot derive :latest tag from digest-based image URI: ${IMAGE_URI}" >&2
  exit 2
fi
if [[ "${LATEST_IMAGE_URI}" != "${IMAGE_URI}" ]]; then
  echo "[gpu] tagging: ${LATEST_IMAGE_URI}"
  docker tag "${IMAGE_URI}" "${LATEST_IMAGE_URI}"
fi

if [[ "${PUSH}" -eq 1 ]]; then
  echo "[gpu] pushing: ${IMAGE_URI}"
  docker push "${IMAGE_URI}"
  if [[ "${LATEST_IMAGE_URI}" != "${IMAGE_URI}" ]]; then
    echo "[gpu] pushing: ${LATEST_IMAGE_URI}"
    docker push "${LATEST_IMAGE_URI}"
  fi
fi
