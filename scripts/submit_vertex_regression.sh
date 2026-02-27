#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/submit_vertex_regression.sh --priordump <path|gs://...> [OPTIONS] [-- <extra pretrain args>]

Submit a Vertex AI custom training job for regression pretraining.

Options:
  --priordump PATH         Local or gs:// prior dump (required)
  --run-name NAME          Logical run name (default: vertex-reg-<timestamp>)
  --display-name NAME      Vertex job display name (default derived from run name)
  --project PROJECT        GCP project (fallback env/gcloud config)
  --region REGION          Vertex region (fallback env/gcloud ai/region or us-central1)
  --bucket BUCKET          Bucket or gs://bucket[/prefix] for canonical storage root
  --image IMAGE_URI        Training container image URI
  --machine-type TYPE      Vertex machine type (default: n1-standard-8)
  --accelerator-type TYPE  GPU type (default: NVIDIA_TESLA_T4)
  --accelerator-count N    GPU count (default: 1)
  --service-account EMAIL  Optional service account for Vertex job
  --dry-run                Print resolved commands/config without submitting
  -h, --help               Show help

Canonical layout:
  gs://<bucket>/tfm_forecasting/priors/<name>.h5
  gs://<bucket>/tfm_forecasting/runs/<run_name>/weights/*
  gs://<bucket>/tfm_forecasting/runs/<run_name>/checkpoints/*
  gs://<bucket>/tfm_forecasting/runs/<run_name>/metadata/*
USAGE
}

if ! command -v gcloud >/dev/null 2>&1; then
  echo "Error: gcloud is required." >&2
  exit 1
fi

PRIORDUMP=""
RUN_NAME=""
DISPLAY_NAME=""
PROJECT="${VERTEX_PROJECT:-${GCP_PROJECT:-}}"
REGION="${VERTEX_REGION:-}"
BUCKET_INPUT="${VERTEX_BUCKET:-}"
IMAGE_URI="${VERTEX_IMAGE_URI:-}"
MACHINE_TYPE="${VERTEX_MACHINE_TYPE:-n1-standard-8}"
ACCELERATOR_TYPE="${VERTEX_ACCELERATOR_TYPE:-NVIDIA_TESLA_T4}"
ACCELERATOR_COUNT="${VERTEX_ACCELERATOR_COUNT:-1}"
SERVICE_ACCOUNT="${VERTEX_SERVICE_ACCOUNT:-}"
DRY_RUN=0
EXTRA_ARGS=()

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
    --priordump)
      PRIORDUMP="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --priordump=*)
      PRIORDUMP="$(require_value "--priordump" "${1#*=}")"
      shift
      ;;
    --run-name)
      RUN_NAME="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --run-name=*)
      RUN_NAME="$(require_value "--run-name" "${1#*=}")"
      shift
      ;;
    --display-name)
      DISPLAY_NAME="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --display-name=*)
      DISPLAY_NAME="$(require_value "--display-name" "${1#*=}")"
      shift
      ;;
    --project)
      PROJECT="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --project=*)
      PROJECT="$(require_value "--project" "${1#*=}")"
      shift
      ;;
    --region)
      REGION="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --region=*)
      REGION="$(require_value "--region" "${1#*=}")"
      shift
      ;;
    --bucket)
      BUCKET_INPUT="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --bucket=*)
      BUCKET_INPUT="$(require_value "--bucket" "${1#*=}")"
      shift
      ;;
    --image)
      IMAGE_URI="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --image=*)
      IMAGE_URI="$(require_value "--image" "${1#*=}")"
      shift
      ;;
    --machine-type)
      MACHINE_TYPE="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --machine-type=*)
      MACHINE_TYPE="$(require_value "--machine-type" "${1#*=}")"
      shift
      ;;
    --accelerator-type)
      ACCELERATOR_TYPE="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --accelerator-type=*)
      ACCELERATOR_TYPE="$(require_value "--accelerator-type" "${1#*=}")"
      shift
      ;;
    --accelerator-count)
      ACCELERATOR_COUNT="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --accelerator-count=*)
      ACCELERATOR_COUNT="$(require_value "--accelerator-count" "${1#*=}")"
      shift
      ;;
    --service-account)
      SERVICE_ACCOUNT="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --service-account=*)
      SERVICE_ACCOUNT="$(require_value "--service-account" "${1#*=}")"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    --)
      shift
      EXTRA_ARGS+=("$@")
      break
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

if [[ -z "$PRIORDUMP" ]]; then
  echo "Error: --priordump is required." >&2
  exit 2
fi

_gcloud_value() {
  local key="$1"
  local out
  out="$(gcloud config get-value "$key" 2>/dev/null || true)"
  if [[ -z "$out" || "$out" == "(unset)" ]]; then
    return 1
  fi
  printf '%s\n' "$out"
}

normalize_gcs_prefix() {
  local raw="$1"
  if [[ -z "$raw" ]]; then
    return 1
  fi
  if [[ "$raw" != gs://* ]]; then
    raw="gs://${raw}"
  fi
  raw="${raw%/}"
  printf '%s\n' "$raw"
}

yaml_quote() {
  local value="$1"
  value="${value//\'/\'\'}"
  printf "'%s'" "$value"
}

if [[ -z "$PROJECT" ]]; then
  PROJECT="$(_gcloud_value project || true)"
fi
if [[ -z "$PROJECT" ]]; then
  echo "Error: could not determine GCP project. Provide --project or set VERTEX_PROJECT/GCP_PROJECT." >&2
  exit 2
fi

if [[ -z "$REGION" ]]; then
  REGION="$(_gcloud_value ai/region || true)"
fi
if [[ -z "$REGION" ]]; then
  REGION="us-central1"
fi

if [[ -z "$BUCKET_INPUT" ]]; then
  BUCKET_INPUT="$(_gcloud_value ai/staging_bucket || true)"
fi
if [[ -z "$BUCKET_INPUT" ]]; then
  echo "Error: could not determine bucket. Provide --bucket, set VERTEX_BUCKET, or configure gcloud ai/staging_bucket." >&2
  exit 2
fi
BUCKET_PREFIX="$(normalize_gcs_prefix "$BUCKET_INPUT")"

if [[ -z "$RUN_NAME" ]]; then
  RUN_NAME="vertex-reg-$(date +%Y%m%d-%H%M%S)"
fi
if [[ -z "$DISPLAY_NAME" ]]; then
  DISPLAY_NAME="tfm-regression-${RUN_NAME}"
fi

if [[ -z "$IMAGE_URI" ]]; then
  AR_REPOSITORY="${AR_REPOSITORY:-tfm-forecasting}"
  IMAGE_NAME="${IMAGE_NAME:-trainer-gpu}"
  IMAGE_TAG="${IMAGE_TAG:-latest}"
  IMAGE_URI="${REGION}-docker.pkg.dev/${PROJECT}/${AR_REPOSITORY}/${IMAGE_NAME}:${IMAGE_TAG}"
fi

BASE_PREFIX="${BUCKET_PREFIX}/tfm_forecasting"
RUN_PREFIX="${BASE_PREFIX}/runs/${RUN_NAME}"
WEIGHTS_URI="${RUN_PREFIX}/weights/nanotabpfn_dynscm_weights.pth"
BUCKETS_URI="${RUN_PREFIX}/weights/nanotabpfn_dynscm_buckets.pth"
BEST_WEIGHTS_URI="${RUN_PREFIX}/weights/nanotabpfn_dynscm_weights.best.pth"
CHECKPOINT_DIR_URI="${RUN_PREFIX}/checkpoints"
METADATA_URI="${RUN_PREFIX}/metadata"

if [[ "$PRIORDUMP" == gs://* ]]; then
  PRIOR_URI="${PRIORDUMP%/}"
  UPLOAD_CMD=()
else
  if [[ ! -f "$PRIORDUMP" ]]; then
    echo "Error: local prior dump not found: $PRIORDUMP" >&2
    exit 1
  fi
  PRIOR_URI="${BASE_PREFIX}/priors/$(basename "$PRIORDUMP")"
  UPLOAD_CMD=(gcloud storage cp "$PRIORDUMP" "$PRIOR_URI")
fi

TRAIN_ARGS=(
  /app/pretrain_regression.py
  "--priordump=${PRIOR_URI}"
  "--saveweights=${WEIGHTS_URI}"
  "--savebuckets=${BUCKETS_URI}"
  "--save_best_weights=${BEST_WEIGHTS_URI}"
  "--runname=${RUN_NAME}"
  "--checkpoint_gcs_dir=${CHECKPOINT_DIR_URI}"
)
if [[ ${#EXTRA_ARGS[@]} -gt 0 ]]; then
  TRAIN_ARGS+=("${EXTRA_ARGS[@]}")
fi

CONFIG_FILE="$(mktemp -t tfm_vertex_job_XXXX.yaml)"
{
  echo "workerPoolSpecs:"
  echo "- machineSpec:"
  echo "    machineType: $(yaml_quote "$MACHINE_TYPE")"
  echo "    acceleratorType: $(yaml_quote "$ACCELERATOR_TYPE")"
  echo "    acceleratorCount: ${ACCELERATOR_COUNT}"
  echo "  replicaCount: 1"
  echo "  containerSpec:"
  echo "    imageUri: $(yaml_quote "$IMAGE_URI")"
  echo "    command:"
  echo "    - 'python'"
  echo "    args:"
  for arg in "${TRAIN_ARGS[@]}"; do
    echo "    - $(yaml_quote "$arg")"
  done
  echo "baseOutputDirectory:"
  echo "  outputUriPrefix: $(yaml_quote "$METADATA_URI")"
  if [[ -n "$SERVICE_ACCOUNT" ]]; then
    echo "serviceAccount: $(yaml_quote "$SERVICE_ACCOUNT")"
  fi
} > "$CONFIG_FILE"

SUBMIT_CMD=(
  gcloud ai custom-jobs create
  --project "$PROJECT"
  --region "$REGION"
  --display-name "$DISPLAY_NAME"
  --config "$CONFIG_FILE"
)

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] project: ${PROJECT}"
  echo "[dry-run] region: ${REGION}"
  echo "[dry-run] image: ${IMAGE_URI}"
  echo "[dry-run] run_prefix: ${RUN_PREFIX}"
  echo "[dry-run] prior_uri: ${PRIOR_URI}"
  if [[ ${#UPLOAD_CMD[@]} -gt 0 ]]; then
    echo "[dry-run] upload command: ${UPLOAD_CMD[*]}"
  else
    echo "[dry-run] upload command: <none, gs:// priordump already accessible>"
  fi
  echo "[dry-run] submit command: ${SUBMIT_CMD[*]}"
  echo "[dry-run] generated config (${CONFIG_FILE}):"
  cat "$CONFIG_FILE"
  rm -f "$CONFIG_FILE"
  exit 0
fi

if [[ ${#UPLOAD_CMD[@]} -gt 0 ]]; then
  echo "Uploading local prior dump to ${PRIOR_URI}"
  "${UPLOAD_CMD[@]}"
fi

echo "Submitting Vertex custom job: ${DISPLAY_NAME}"
"${SUBMIT_CMD[@]}"
rm -f "$CONFIG_FILE"
