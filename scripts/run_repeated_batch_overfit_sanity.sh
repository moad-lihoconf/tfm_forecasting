#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run_repeated_batch_overfit_sanity.sh [OPTIONS]

Run a deterministic repeated-batch overfit sanity check against the ultra-overfit
DynSCM dump using the dump-based trainer.

Options:
  --vertex               Submit to Vertex instead of running locally
  --dump PATH            Prior dump path (default: workdir/forecast_research/dynscm_prior_ultra_overfit_512.h5)
  --run-name NAME        Run name / workdir stem (default: repeated-batch-overfit-<timestamp>)
  --subset-size N        Train subset size (default: 1)
  --val-subset-size N    Validation subset size (default: 1)
  --batch-size N         Batch size (default: 1)
  --steps N              Steps per epoch (default: 200)
  --epochs N             Epoch count (default: 20)
  --lr X                 Learning rate (default: 1e-4)
  --split-seed N         Deterministic split seed (default: 2402)
  --grad-clip-norm X     Gradient clipping threshold (default: 1.0)
  --target-normalization MODE
                         none | per_function_zscore | per_function_clamped (default: none)
  --feature-normalization MODE
                         per_function_zscore | none (default: per_function_zscore)
  --project NAME         Vertex project override
  --region NAME          Vertex region override
  --bucket BUCKET        Vertex bucket override
  --image URI            Vertex image override
  --machine-type TYPE    Vertex machine type (default: n1-standard-8)
  --accelerator-type T   Vertex GPU type (default: NVIDIA_TESLA_T4)
  --accelerator-count N  Vertex GPU count (default: 1)
  --service-account SA   Vertex service account override
  --dry-run              Print resolved command without running it
  -h, --help             Show help
USAGE
}

require_value() {
  local opt="$1"
  local val="${2-}"
  if [[ -z "$val" || "$val" == --* ]]; then
    echo "Error: ${opt} requires a non-empty value." >&2
    exit 2
  fi
  printf '%s\n' "$val"
}

DUMP_PATH="workdir/forecast_research/dynscm_prior_ultra_overfit_512.h5"
RUN_NAME="repeated-batch-overfit-$(date +%Y%m%d-%H%M%S)"
SUBSET_SIZE=1
VAL_SUBSET_SIZE=1
BATCH_SIZE=1
STEPS=200
EPOCHS=20
LR="1e-4"
SPLIT_SEED=2402
GRAD_CLIP_NORM="1.0"
TARGET_NORMALIZATION="none"
FEATURE_NORMALIZATION="per_function_zscore"
VERTEX_MODE=0
PROJECT="${VERTEX_PROJECT:-${GCP_PROJECT:-}}"
REGION="${VERTEX_REGION:-}"
BUCKET_INPUT="${VERTEX_BUCKET:-}"
IMAGE_URI="${VERTEX_IMAGE_URI:-}"
MACHINE_TYPE="${VERTEX_MACHINE_TYPE:-n1-standard-8}"
ACCELERATOR_TYPE="${VERTEX_ACCELERATOR_TYPE:-NVIDIA_TESLA_T4}"
ACCELERATOR_COUNT="${VERTEX_ACCELERATOR_COUNT:-1}"
SERVICE_ACCOUNT="${VERTEX_SERVICE_ACCOUNT:-}"
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --vertex)
      VERTEX_MODE=1
      shift
      ;;
    --dump)
      DUMP_PATH="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --dump=*)
      DUMP_PATH="$(require_value --dump "${1#*=}")"
      shift
      ;;
    --run-name)
      RUN_NAME="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --run-name=*)
      RUN_NAME="$(require_value --run-name "${1#*=}")"
      shift
      ;;
    --subset-size)
      SUBSET_SIZE="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --subset-size=*)
      SUBSET_SIZE="$(require_value --subset-size "${1#*=}")"
      shift
      ;;
    --val-subset-size)
      VAL_SUBSET_SIZE="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --val-subset-size=*)
      VAL_SUBSET_SIZE="$(require_value --val-subset-size "${1#*=}")"
      shift
      ;;
    --batch-size)
      BATCH_SIZE="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --batch-size=*)
      BATCH_SIZE="$(require_value --batch-size "${1#*=}")"
      shift
      ;;
    --steps)
      STEPS="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --steps=*)
      STEPS="$(require_value --steps "${1#*=}")"
      shift
      ;;
    --epochs)
      EPOCHS="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --epochs=*)
      EPOCHS="$(require_value --epochs "${1#*=}")"
      shift
      ;;
    --lr)
      LR="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --lr=*)
      LR="$(require_value --lr "${1#*=}")"
      shift
      ;;
    --split-seed)
      SPLIT_SEED="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --split-seed=*)
      SPLIT_SEED="$(require_value --split-seed "${1#*=}")"
      shift
      ;;
    --grad-clip-norm)
      GRAD_CLIP_NORM="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --grad-clip-norm=*)
      GRAD_CLIP_NORM="$(require_value --grad-clip-norm "${1#*=}")"
      shift
      ;;
    --target-normalization)
      TARGET_NORMALIZATION="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --target-normalization=*)
      TARGET_NORMALIZATION="$(require_value --target-normalization "${1#*=}")"
      shift
      ;;
    --feature-normalization)
      FEATURE_NORMALIZATION="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --feature-normalization=*)
      FEATURE_NORMALIZATION="$(require_value --feature-normalization "${1#*=}")"
      shift
      ;;
    --project)
      PROJECT="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --project=*)
      PROJECT="$(require_value --project "${1#*=}")"
      shift
      ;;
    --region)
      REGION="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --region=*)
      REGION="$(require_value --region "${1#*=}")"
      shift
      ;;
    --bucket)
      BUCKET_INPUT="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --bucket=*)
      BUCKET_INPUT="$(require_value --bucket "${1#*=}")"
      shift
      ;;
    --image)
      IMAGE_URI="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --image=*)
      IMAGE_URI="$(require_value --image "${1#*=}")"
      shift
      ;;
    --machine-type)
      MACHINE_TYPE="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --machine-type=*)
      MACHINE_TYPE="$(require_value --machine-type "${1#*=}")"
      shift
      ;;
    --accelerator-type)
      ACCELERATOR_TYPE="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --accelerator-type=*)
      ACCELERATOR_TYPE="$(require_value --accelerator-type "${1#*=}")"
      shift
      ;;
    --accelerator-count)
      ACCELERATOR_COUNT="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --accelerator-count=*)
      ACCELERATOR_COUNT="$(require_value --accelerator-count "${1#*=}")"
      shift
      ;;
    --service-account)
      SERVICE_ACCOUNT="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --service-account=*)
      SERVICE_ACCOUNT="$(require_value --service-account "${1#*=}")"
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
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

printf '[overfit-sanity] mode=%s run_name=%s subset_size=%s val_subset_size=%s batch_size=%s steps=%s epochs=%s grad_clip_norm=%s\n' \
  "$([[ "$VERTEX_MODE" -eq 1 ]] && printf 'vertex' || printf 'local')" \
  "$RUN_NAME" "$SUBSET_SIZE" "$VAL_SUBSET_SIZE" "$BATCH_SIZE" "$STEPS" "$EPOCHS" "$GRAD_CLIP_NORM"
printf '[overfit-sanity] dump=%s\n' "$DUMP_PATH"

if [[ "$VERTEX_MODE" -eq 1 ]]; then
  CMD=(
    bash scripts/submit_vertex_regression.sh
    --priordump "$DUMP_PATH"
    --run-name "$RUN_NAME"
    --machine-type "$MACHINE_TYPE"
    --accelerator-type "$ACCELERATOR_TYPE"
    --accelerator-count "$ACCELERATOR_COUNT"
  )
  if [[ -n "$PROJECT" ]]; then
    CMD+=(--project "$PROJECT")
  fi
  if [[ -n "$REGION" ]]; then
    CMD+=(--region "$REGION")
  fi
  if [[ -n "$BUCKET_INPUT" ]]; then
    CMD+=(--bucket "$BUCKET_INPUT")
  fi
  if [[ -n "$IMAGE_URI" ]]; then
    CMD+=(--image "$IMAGE_URI")
  fi
  if [[ -n "$SERVICE_ACCOUNT" ]]; then
    CMD+=(--service-account "$SERVICE_ACCOUNT")
  fi
  if [[ "$DRY_RUN" -eq 1 ]]; then
    CMD+=(--dry-run)
  fi
  CMD+=(
    --
    --epochs "$EPOCHS"
    --steps "$STEPS"
    --batchsize "$BATCH_SIZE"
    --accumulate 1
    --lr "$LR"
    --optimizer adamw
    --regression_loss mse
    --target_normalization "$TARGET_NORMALIZATION"
    --feature_normalization "$FEATURE_NORMALIZATION"
    --train_subset_size "$SUBSET_SIZE"
    --val_subset_size "$VAL_SUBSET_SIZE"
    --split_seed "$SPLIT_SEED"
    --eval_every_epochs 1
    --val_steps auto_full
    --early_stopping_patience 10
    --early_stopping_min_delta 1e-5
    --grad_clip_norm "$GRAD_CLIP_NORM"
    --debug_trace_first_n_batches "$STEPS"
    --debug_trace_every_n_batches 0
  )
else
  WEIGHTS_PATH="workdir/${RUN_NAME}/weights.pth"
  BUCKETS_PATH="workdir/${RUN_NAME}/buckets.pth"
  TRACE_PATH="workdir/${RUN_NAME}/train_trace.json"

  CMD=(
    poetry run python pretrain_regression.py
    --priordump "$DUMP_PATH"
    --saveweights "$WEIGHTS_PATH"
    --savebuckets "$BUCKETS_PATH"
    --runname "$RUN_NAME"
    --epochs "$EPOCHS"
    --steps "$STEPS"
    --batchsize "$BATCH_SIZE"
    --accumulate 1
    --lr "$LR"
    --optimizer adamw
    --regression_loss mse
    --target_normalization "$TARGET_NORMALIZATION"
    --feature_normalization "$FEATURE_NORMALIZATION"
    --train_subset_size "$SUBSET_SIZE"
    --val_subset_size "$VAL_SUBSET_SIZE"
    --split_seed "$SPLIT_SEED"
    --eval_every_epochs 1
    --val_steps auto_full
    --early_stopping_patience 10
    --early_stopping_min_delta 1e-5
    --grad_clip_norm "$GRAD_CLIP_NORM"
    --debug_train_trace_json "$TRACE_PATH"
    --debug_trace_first_n_batches "$STEPS"
    --debug_trace_every_n_batches 0
  )
  if [[ "$DRY_RUN" -eq 1 ]]; then
    printf '[dry-run] '
    printf '%q ' "${CMD[@]}"
    printf '\n'
    exit 0
  fi
fi

"${CMD[@]}"
