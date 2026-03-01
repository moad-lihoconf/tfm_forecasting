#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run_vertex_curriculum.sh [OPTIONS]

Generate priors, audit them, upload them to GCS, submit each Vertex stage,
and wait for completion before warm-starting the next stage.

Options:
  --run-prefix PREFIX       Prefix for all run names (default: dynscm-curriculum-<timestamp>)
  --project PROJECT         GCP project (fallback env/gcloud config)
  --region REGION           Vertex region (fallback env/gcloud ai/region or us-central1)
  --bucket BUCKET           Bucket or gs://bucket[/prefix] for canonical storage root
  --machine-type TYPE       Vertex machine type (default: g2-standard-12)
  --accelerator-type TYPE   GPU type (default: NVIDIA_L4)
  --accelerator-count N     GPU count (default: 1)
  --python BIN              Python executable to use locally (default: auto-detect)
  --workdir DIR             Local work directory (default: workdir/curriculum/<run-prefix>)
  --from-stage NAME         First stage to execute (default: benchmark_aligned_easy_16k)
  --to-stage NAME           Last stage to execute (default: benchmark_aligned_full_32k)
  --poll-seconds N          Seconds between Vertex job polls (default: 60)
  --stream-logs             Stream Vertex logs for each stage (default: enabled)
  --no-stream-logs          Disable log streaming and only poll job state
  --benchmark-slice         Run a fixed benchmark slice after each stage (default: enabled)
  --no-benchmark-slice      Disable benchmark slice evaluation
  --benchmark-device DEV    Benchmark device: cpu|cuda (default: cpu)
  --benchmark-max-series N  Max series per dataset for the slice (default: 64)
  --benchmark-threshold P   Max allowed relative degradation vs prev stage (default: 0.03)
  --benchmark-datasets CSV  Comma-separated dataset names (default: exchange_rate,ettm1)
  --dry-run                 Print the full plan without running commands
  -h, --help                Show help

Stages:
  benchmark_aligned_easy_16k
  benchmark_aligned_easy_plus_16k
  benchmark_aligned_medium_graph_16k
  benchmark_aligned_medium_noise_16k
  benchmark_aligned_medium_missing_16k
  benchmark_aligned_medium_32k
  benchmark_aligned_full_32k
USAGE
}

if ! command -v gcloud >/dev/null 2>&1; then
  echo "Error: gcloud is required." >&2
  exit 1
fi

require_value() {
  local opt="$1"
  local val="${2-}"
  if [[ -z "$val" || "$val" == --* ]]; then
    echo "Error: ${opt} requires a non-empty value." >&2
    exit 2
  fi
  printf '%s\n' "$val"
}

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

resolve_python_cmd() {
  if [[ -n "$PYTHON_BIN_OVERRIDE" ]]; then
    if ! command -v "$PYTHON_BIN_OVERRIDE" >/dev/null 2>&1; then
      echo "Error: requested python executable not found: $PYTHON_BIN_OVERRIDE" >&2
      exit 2
    fi
    PYTHON_CMD=("$PYTHON_BIN_OVERRIDE")
    return 0
  fi

  if command -v python >/dev/null 2>&1; then
    PYTHON_CMD=(python)
    return 0
  fi
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=(python3)
    return 0
  fi
  if command -v poetry >/dev/null 2>&1; then
    PYTHON_CMD=(poetry run python)
    return 0
  fi

  echo "Error: could not find a Python interpreter. Install python/python3 or pass --python." >&2
  exit 2
}

RUN_PREFIX="dynscm-curriculum-$(date +%Y%m%d-%H%M%S)"
PROJECT="${VERTEX_PROJECT:-${GCP_PROJECT:-}}"
REGION="${VERTEX_REGION:-}"
BUCKET_INPUT="${VERTEX_BUCKET:-}"
MACHINE_TYPE="${VERTEX_MACHINE_TYPE:-g2-standard-12}"
ACCELERATOR_TYPE="${VERTEX_ACCELERATOR_TYPE:-NVIDIA_L4}"
ACCELERATOR_COUNT="${VERTEX_ACCELERATOR_COUNT:-1}"
PYTHON_BIN_OVERRIDE="${PYTHON_BIN:-}"
PYTHON_CMD=()
WORK_ROOT=""
FROM_STAGE="benchmark_aligned_easy_16k"
TO_STAGE="benchmark_aligned_full_32k"
POLL_SECONDS=60
STREAM_LOGS=1
DRY_RUN=0
BENCHMARK_SLICE=1
BENCHMARK_DEVICE="cpu"
BENCHMARK_MAX_SERIES=64
BENCHMARK_THRESHOLD=0.03
BENCHMARK_DATASETS="exchange_rate,ettm1"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-prefix)
      RUN_PREFIX="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --run-prefix=*)
      RUN_PREFIX="$(require_value --run-prefix "${1#*=}")"
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
    --python)
      PYTHON_BIN_OVERRIDE="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --python=*)
      PYTHON_BIN_OVERRIDE="$(require_value --python "${1#*=}")"
      shift
      ;;
    --workdir)
      WORK_ROOT="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --workdir=*)
      WORK_ROOT="$(require_value --workdir "${1#*=}")"
      shift
      ;;
    --from-stage)
      FROM_STAGE="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --from-stage=*)
      FROM_STAGE="$(require_value --from-stage "${1#*=}")"
      shift
      ;;
    --to-stage)
      TO_STAGE="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --to-stage=*)
      TO_STAGE="$(require_value --to-stage "${1#*=}")"
      shift
      ;;
    --poll-seconds)
      POLL_SECONDS="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --poll-seconds=*)
      POLL_SECONDS="$(require_value --poll-seconds "${1#*=}")"
      shift
      ;;
    --benchmark-slice)
      BENCHMARK_SLICE=1
      shift
      ;;
    --no-benchmark-slice)
      BENCHMARK_SLICE=0
      shift
      ;;
    --benchmark-device)
      BENCHMARK_DEVICE="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --benchmark-device=*)
      BENCHMARK_DEVICE="$(require_value --benchmark-device "${1#*=}")"
      shift
      ;;
    --benchmark-max-series)
      BENCHMARK_MAX_SERIES="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --benchmark-max-series=*)
      BENCHMARK_MAX_SERIES="$(require_value --benchmark-max-series "${1#*=}")"
      shift
      ;;
    --benchmark-threshold)
      BENCHMARK_THRESHOLD="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --benchmark-threshold=*)
      BENCHMARK_THRESHOLD="$(require_value --benchmark-threshold "${1#*=}")"
      shift
      ;;
    --benchmark-datasets)
      BENCHMARK_DATASETS="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --benchmark-datasets=*)
      BENCHMARK_DATASETS="$(require_value --benchmark-datasets "${1#*=}")"
      shift
      ;;
    --stream-logs)
      STREAM_LOGS=1
      shift
      ;;
    --no-stream-logs)
      STREAM_LOGS=0
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
BASE_PREFIX="${BUCKET_PREFIX}/tfm_forecasting"

if [[ -z "$WORK_ROOT" ]]; then
  WORK_ROOT="workdir/curriculum/${RUN_PREFIX}"
fi
resolve_python_cmd
PRIOR_DIR="${WORK_ROOT}/priors"
AUDIT_DIR="${WORK_ROOT}/audits"
LOG_DIR="${WORK_ROOT}/logs"
SUMMARY_FILE="${WORK_ROOT}/curriculum_summary.tsv"
EVAL_DIR="${WORK_ROOT}/eval"
SCORECARD_FILE="${WORK_ROOT}/curriculum_scorecard.tsv"
mkdir -p "$PRIOR_DIR" "$AUDIT_DIR" "$LOG_DIR" "$EVAL_DIR"

# The automated ladder follows the current evidence-backed diversity-axis
# curriculum; legacy exploratory stages remain available only as manual priors.
STAGES=(
  benchmark_aligned_easy_16k
  benchmark_aligned_easy_plus_16k
  benchmark_aligned_medium_graph_16k
  benchmark_aligned_medium_noise_16k
  benchmark_aligned_medium_missing_16k
  benchmark_aligned_medium_32k
  benchmark_aligned_full_32k
)

stage_index() {
  local target="$1"
  local i
  for i in "${!STAGES[@]}"; do
    if [[ "${STAGES[$i]}" == "$target" ]]; then
      printf '%s\n' "$i"
      return 0
    fi
  done
  return 1
}

stage_run_suffix() {
  case "$1" in
    benchmark_aligned_easy_16k) echo "easy-16k" ;;
    benchmark_aligned_easy_plus_16k) echo "easy-plus-16k" ;;
    benchmark_aligned_medium_graph_16k) echo "medium-graph-16k" ;;
    benchmark_aligned_medium_noise_16k) echo "medium-noise-16k" ;;
    benchmark_aligned_medium_missing_16k) echo "medium-missing-16k" ;;
    benchmark_aligned_medium_32k) echo "medium-32k" ;;
    benchmark_aligned_full_32k) echo "full-32k" ;;
    *)
      echo "Error: unknown stage: $1" >&2
      exit 2
      ;;
  esac
}

stage_epochs() {
  case "$1" in
    benchmark_aligned_easy_16k) echo "10" ;;
    benchmark_aligned_easy_plus_16k) echo "20" ;;
    benchmark_aligned_medium_32k) echo "60" ;;
    benchmark_aligned_full_32k) echo "80" ;;
    *) echo "40" ;;
  esac
}

stage_patience() {
  case "$1" in
    benchmark_aligned_easy_16k) echo "3" ;;
    benchmark_aligned_easy_plus_16k) echo "5" ;;
    *) echo "10" ;;
  esac
}

stage_min_delta() {
  case "$1" in
    benchmark_aligned_easy_16k) echo "1e-4" ;;
    *) echo "1e-5" ;;
  esac
}

common_train_args() {
  cat <<'ARGS'
--steps=400
--batchsize=16
--accumulate=2
--optimizer=adamw
--lr=5e-4
--weight_decay=0.0
--dropout=0.0
--amp
--amp_dtype=float16
--loss_weighting=per_target
--regression_loss=mse
--target_normalization=none
--feature_normalization=per_function_zscore
--min_train_target_std=1e-3
--val_split=0.125
--split_seed=2402
--val_steps=32
--eval_every_epochs=1
--openml_eval_every_epochs=0
--no-strict_prior_integrity
ARGS
}

FROM_INDEX="$(stage_index "$FROM_STAGE" || true)"
TO_INDEX="$(stage_index "$TO_STAGE" || true)"
if [[ -z "$FROM_INDEX" ]]; then
  echo "Error: unknown --from-stage: $FROM_STAGE" >&2
  exit 2
fi
if [[ -z "$TO_INDEX" ]]; then
  echo "Error: unknown --to-stage: $TO_STAGE" >&2
  exit 2
fi
if (( FROM_INDEX > TO_INDEX )); then
  echo "Error: --from-stage must come before --to-stage." >&2
  exit 2
fi

run_name_for_stage() {
  local stage="$1"
  echo "${RUN_PREFIX}-$(stage_run_suffix "$stage")"
}

prior_uri_for_stage() {
  local stage="$1"
  echo "${BASE_PREFIX}/priors/${stage}.h5"
}

local_prior_for_stage() {
  local stage="$1"
  echo "${PRIOR_DIR}/${stage}.h5"
}

audit_json_for_stage() {
  local stage="$1"
  echo "${AUDIT_DIR}/${stage}.audit.json"
}

checkpoint_uri_for_stage() {
  local stage="$1"
  local run_name
  run_name="$(run_name_for_stage "$stage")"
  echo "${BASE_PREFIX}/runs/${run_name}/checkpoints/best_checkpoint.pth"
}

wait_for_job() {
  local job_id="$1"
  local stage="$2"
  local job_resource="projects/${PROJECT}/locations/${REGION}/customJobs/${job_id}"
  local stream_pid=""

  if [[ "$STREAM_LOGS" -eq 1 ]]; then
    echo "[curriculum] streaming logs for stage ${stage}: ${job_resource}"
    gcloud ai custom-jobs stream-logs "$job_resource" &
    stream_pid="$!"
  fi

  while true; do
    local state
    state="$(gcloud ai custom-jobs describe "$job_id" \
      --project "$PROJECT" \
      --region "$REGION" \
      --verbosity=error \
      --format='value(state)')"
    printf '[curriculum] stage=%s job=%s state=%s\n' "$stage" "$job_id" "$state"
    case "$state" in
      JOB_STATE_SUCCEEDED)
        if [[ -n "$stream_pid" ]]; then
          kill "$stream_pid" 2>/dev/null || true
          wait "$stream_pid" 2>/dev/null || true
        fi
        return 0
        ;;
      JOB_STATE_FAILED|JOB_STATE_CANCELLED|JOB_STATE_EXPIRED|JOB_STATE_PAUSED)
        if [[ -n "$stream_pid" ]]; then
          kill "$stream_pid" 2>/dev/null || true
          wait "$stream_pid" 2>/dev/null || true
        fi
        echo "[curriculum] stage ${stage} did not succeed." >&2
        echo "[curriculum] inspect logs with: gcloud ai custom-jobs stream-logs projects/${PROJECT_NUMBER:-$PROJECT}/locations/${REGION}/customJobs/${job_id}" >&2
        return 1
        ;;
      *)
        sleep "$POLL_SECONDS"
        ;;
    esac
  done
}

print_stage_plan() {
  local stage="$1"
  local run_name="$2"
  local local_prior="$3"
  local prior_uri="$4"
  local audit_json="$5"
  local epochs="$6"
  local patience="$7"
  local min_delta="$8"
  local load_checkpoint="$9"

  printf '[dry-run] stage=%s\n' "$stage"
  printf '  run_name=%s\n' "$run_name"
  printf '  local_prior=%s\n' "$local_prior"
  printf '  prior_uri=%s\n' "$prior_uri"
  printf '  audit_json=%s\n' "$audit_json"
  printf '  epochs=%s\n' "$epochs"
  printf '  early_stopping_patience=%s\n' "$patience"
  printf '  early_stopping_min_delta=%s\n' "$min_delta"
  printf '  python_cmd=%s\n' "${PYTHON_CMD[*]}"
  if [[ "$STREAM_LOGS" -eq 1 ]]; then
    printf '  stream_logs=enabled\n'
  else
    printf '  stream_logs=disabled\n'
  fi
  if [[ -n "$load_checkpoint" ]]; then
    printf '  warm_start_checkpoint=%s\n' "$load_checkpoint"
  else
    printf '  warm_start_checkpoint=<none>\n'
  fi
  if [[ "$BENCHMARK_SLICE" -eq 1 ]]; then
    printf '  benchmark_slice=enabled (%s max_series=%s device=%s threshold=%s)\n' \
      "$BENCHMARK_DATASETS" "$BENCHMARK_MAX_SERIES" "$BENCHMARK_DEVICE" "$BENCHMARK_THRESHOLD"
  else
    printf '  benchmark_slice=disabled\n'
  fi
}

printf 'stage\trun_name\tprofile\tjob_id\tcheckpoint_uri\tstatus\n' > "$SUMMARY_FILE"
printf 'stage\trun_name\tjob_id\ttarget_y_std_mean\ttrain_loss\tval_loss\tval_rmse\tval_nrmse\tslice_dynscm_rmse\tslice_dynscm_mase\tslice_standard_rmse\tslice_standard_mase\tgate\n' > "$SCORECARD_FILE"

_read_audit_field() {
  local json_path="$1"
  local key="$2"
  "${PYTHON_CMD[@]}" - <<PY
import json
with open(${json_path!r}, "r", encoding="utf-8") as f:
    payload = json.load(f)
value = payload.get(${key!r})
print("nan" if value is None else str(value))
PY
}

_read_ckpt_metric_triplet() {
  local ckpt_path="$1"
  "${PYTHON_CMD[@]}" - <<PY
import math
import torch
ckpt = torch.load(${ckpt_path!r}, map_location="cpu")
metrics = ckpt.get("metrics", {}) if isinstance(ckpt, dict) else {}
train_loss = float(metrics.get("train_loss", float("nan")))
val_loss = float(metrics.get("val_loss", float("nan")))
val_rmse = float(
    metrics.get(
        "val_rmse",
        math.sqrt(val_loss) if math.isfinite(val_loss) and val_loss >= 0 else float("nan"),
    )
)
print(f"{train_loss}\t{val_loss}\t{val_rmse}")
PY
}

_bench_means_for_model() {
  local rows_csv="$1"
  local model_name="$2"
  "${PYTHON_CMD[@]}" - <<PY
import pandas as pd
df = pd.read_csv(${rows_csv!r})
ok = df[(df["status"] == "ok") & (df["model"] == ${model_name!r})]
if ok.empty:
    print("nan\tnan")
else:
    rmse = float(ok["rmse"].mean())
    mase = float(ok["mase"].mean())
    print(f"{rmse}\t{mase}")
PY
}

PREV_STAGE=""
PREV_DYNSCM_RMSE=""
PREV_DYNSCM_MASE=""
GATED_STOP=0

for i in "${!STAGES[@]}"; do
  if (( i < FROM_INDEX || i > TO_INDEX )); then
    continue
  fi

  stage="${STAGES[$i]}"
  run_name="$(run_name_for_stage "$stage")"
  local_prior="$(local_prior_for_stage "$stage")"
  prior_uri="$(prior_uri_for_stage "$stage")"
  audit_json="$(audit_json_for_stage "$stage")"
  epochs="$(stage_epochs "$stage")"
  patience="$(stage_patience "$stage")"
  min_delta="$(stage_min_delta "$stage")"

  load_checkpoint=""
  if (( i > 0 )); then
    prev_stage="${STAGES[$((i - 1))]}"
    load_checkpoint="$(checkpoint_uri_for_stage "$prev_stage")"
  fi

  if [[ "$DRY_RUN" -eq 1 ]]; then
    print_stage_plan "$stage" "$run_name" "$local_prior" "$prior_uri" "$audit_json" "$epochs" "$patience" "$min_delta" "$load_checkpoint"
    continue
  fi

  echo "[curriculum] generating prior for stage ${stage}"
  "${PYTHON_CMD[@]}" -m tfmplayground.priors \
    --lib dynscm \
    --dynscm_profile "$stage" \
    --np_seed 42 \
    --torch_seed 42 \
    --dynscm_seed 42 \
    --dynscm_workers 4 \
    --dynscm_worker_blas_threads 1 \
    --no_dynscm_compute_spectral_diagnostics \
    --save_path "$local_prior"

  echo "[curriculum] auditing prior for stage ${stage}"
  "${PYTHON_CMD[@]}" scripts/audit_dynscm_prior.py \
    --priordump "$local_prior" \
    --no-fail_on_issues \
    --json_out "$audit_json"

  echo "[curriculum] uploading prior for stage ${stage} to ${prior_uri}"
  gcloud storage cp "$local_prior" "$prior_uri"

  submit_cmd=(
    bash scripts/submit_vertex_regression.sh
    --project "$PROJECT"
    --region "$REGION"
    --bucket "$BUCKET_PREFIX"
    --machine-type "$MACHINE_TYPE"
    --accelerator-type "$ACCELERATOR_TYPE"
    --accelerator-count "$ACCELERATOR_COUNT"
    --priordump "$prior_uri"
    --run-name "$run_name"
    --
  )

  if [[ -n "$load_checkpoint" ]]; then
    submit_cmd+=("--loadcheckpoint=${load_checkpoint}" "--warm_start")
  fi
  submit_cmd+=("--epochs=${epochs}")
  submit_cmd+=("--early_stopping_patience=${patience}")
  submit_cmd+=("--early_stopping_min_delta=${min_delta}")
  while IFS= read -r arg; do
    [[ -z "$arg" ]] && continue
    submit_cmd+=("$arg")
  done < <(common_train_args)

  echo "[curriculum] submitting stage ${stage}"
  submit_output="$(${submit_cmd[@]} 2>&1)"
  printf '%s\n' "$submit_output" | tee "${LOG_DIR}/${stage}.submit.log"
  job_resource="$(printf '%s\n' "$submit_output" | grep -o 'projects/[^ ]*/locations/[^ ]*/customJobs/[0-9]\+' | tail -n 1 || true)"
  if [[ -z "$job_resource" ]]; then
    echo "Error: failed to parse Vertex job id for stage ${stage}." >&2
    exit 1
  fi
  job_id="${job_resource##*/}"

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$stage" "$run_name" "$stage" "$job_id" "$(checkpoint_uri_for_stage "$stage")" "submitted" >> "$SUMMARY_FILE"

  wait_for_job "$job_id" "$stage"

  printf '%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$stage" "$run_name" "$stage" "$job_id" "$(checkpoint_uri_for_stage "$stage")" "succeeded" >> "$SUMMARY_FILE"
  echo "[curriculum] stage ${stage} completed"

  if [[ "$BENCHMARK_SLICE" -eq 1 ]]; then
    stage_eval_dir="${EVAL_DIR}/${stage}"
    mkdir -p "$stage_eval_dir"
    local_ckpt="${stage_eval_dir}/best_checkpoint.pth"
    echo "[curriculum] downloading checkpoint for stage ${stage}"
    gcloud storage cp "$(checkpoint_uri_for_stage "$stage")" "$local_ckpt"

    target_std_mean="$(_read_audit_field "$audit_json" "target_y_std_mean")"
    IFS=$'\t' read -r train_loss val_loss val_rmse < <(_read_ckpt_metric_triplet "$local_ckpt")
    val_nrmse="$("${PYTHON_CMD[@]}" - <<PY
import math
try:
    val_rmse = float(${val_rmse!r})
    target_std = float(${target_std_mean!r})
except Exception:
    print("nan")
    raise SystemExit(0)
if not (math.isfinite(val_rmse) and math.isfinite(target_std) and target_std > 0):
    print("nan")
else:
    print(str(val_rmse / target_std))
PY
)"

    bench_out="${stage_eval_dir}/benchmark_slice"
    echo "[curriculum] running benchmark slice for stage ${stage}"
    bash scripts/run_benchmark_slice.sh \
      --ckpt "$local_ckpt" \
      --output_dir "$bench_out" \
      --device "$BENCHMARK_DEVICE" \
      --max-series "$BENCHMARK_MAX_SERIES" \
      --datasets "$BENCHMARK_DATASETS"

    rows_csv="${bench_out}/regression_rows.csv"
    if [[ ! -f "$rows_csv" ]]; then
      echo "[curriculum] benchmark did not produce regression_rows.csv; stopping." >&2
      exit 1
    fi

    IFS=$'\t' read -r dynscm_rmse dynscm_mase < <(_bench_means_for_model "$rows_csv" "nanotabpfn_dynscm")
    IFS=$'\t' read -r std_rmse std_mase < <(_bench_means_for_model "$rows_csv" "nanotabpfn_standard")

    gate="na"
    if [[ -n "$PREV_DYNSCM_RMSE" && -n "$PREV_DYNSCM_MASE" ]]; then
      gate="$("${PYTHON_CMD[@]}" - <<PY
import math
prev_rmse = float(${PREV_DYNSCM_RMSE!r})
prev_mase = float(${PREV_DYNSCM_MASE!r})
cur_rmse = float(${dynscm_rmse!r})
cur_mase = float(${dynscm_mase!r})
thr = float(${BENCHMARK_THRESHOLD!r})
if not all(math.isfinite(v) for v in (prev_rmse, prev_mase, cur_rmse, cur_mase)):
    print("fail_nonfinite")
elif prev_rmse <= 0 or prev_mase <= 0:
    print("na")
else:
    rmse_rel = (cur_rmse - prev_rmse) / prev_rmse
    mase_rel = (cur_mase - prev_mase) / prev_mase
    if rmse_rel > thr or mase_rel > thr:
        print("fail_degrade")
    else:
        print("pass")
PY
)"
    fi

    printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
      "$stage" "$run_name" "$job_id" "$target_std_mean" "$train_loss" "$val_loss" "$val_rmse" "$val_nrmse" \
      "$dynscm_rmse" "$dynscm_mase" "$std_rmse" "$std_mase" "$gate" >> "$SCORECARD_FILE"

    if [[ "$gate" == "fail_degrade" || "$gate" == "fail_nonfinite" ]]; then
      echo "[curriculum] gate failed at stage ${stage} (prev=${PREV_STAGE}); stopping curriculum here." >&2
      GATED_STOP=1
      break
    fi

    PREV_STAGE="$stage"
    PREV_DYNSCM_RMSE="$dynscm_rmse"
    PREV_DYNSCM_MASE="$dynscm_mase"
  fi
done

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "[dry-run] summary_file=${SUMMARY_FILE}"
  echo "[dry-run] scorecard_file=${SCORECARD_FILE}"
else
  if [[ "$GATED_STOP" -eq 1 ]]; then
    echo "[curriculum] stopped early due to benchmark gate"
  else
    echo "[curriculum] all requested stages completed"
  fi
  echo "[curriculum] summary_file=${SUMMARY_FILE}"
  echo "[curriculum] scorecard_file=${SCORECARD_FILE}"
fi
