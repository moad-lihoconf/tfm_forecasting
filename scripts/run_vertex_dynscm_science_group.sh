#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run_vertex_dynscm_science_group.sh [OPTIONS]

Run one ML-science experiment group on Vertex AI, then compute synthetic eval,
live prior audit, benchmark comparison, and a machine-readable scorecard.

Options:
  --group NAME              Experiment group: temporal_ablation | benchmark_contract
  --run-prefix PREFIX       Prefix for all run names (default: dynscm-science-<timestamp>)
  --project PROJECT         GCP project (fallback env/gcloud config)
  --region REGION           Vertex region (fallback env/gcloud ai/region or us-central1)
  --bucket BUCKET           Bucket or gs://bucket[/prefix] for canonical storage root
  --machine-type TYPE       Vertex machine type (default: g2-standard-12)
  --accelerator-type TYPE   GPU type (default: NVIDIA_L4)
  --accelerator-count N     GPU count (default: 1)
  --python BIN              Python executable to use locally (default: auto-detect)
  --workdir DIR             Local work directory (default: workdir/research/<run-prefix>)
  --poll-seconds N          Seconds between Vertex job polls (default: 60)
  --stream-logs             Stream Vertex logs for each job (default: enabled)
  --no-stream-logs          Disable log streaming
  --warm-start-checkpoint   Common warm-start checkpoint for all runs
  --dry-run                 Print the plan without running it
  -h, --help                Show help
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
  if [[ "$raw" != gs://* ]]; then
    raw="gs://${raw}"
  fi
  printf '%s\n' "${raw%/}"
}

resolve_python_cmd() {
  if [[ -n "$PYTHON_BIN_OVERRIDE" ]]; then
    PYTHON_CMD=("$PYTHON_BIN_OVERRIDE")
    return
  fi
  if command -v python >/dev/null 2>&1; then
    PYTHON_CMD=(python)
    return
  fi
  if command -v python3 >/dev/null 2>&1; then
    PYTHON_CMD=(python3)
    return
  fi
  echo "Error: could not find python/python3." >&2
  exit 2
}

run_name_for_profile() {
  local prefix="$1"
  local profile="$2"
  printf '%s-%s\n' "$prefix" "${profile//_/-}"
}

append_scorecard_json() {
  local row_json="$1"
  "${PYTHON_CMD[@]}" - <<PY
import json
from pathlib import Path

path = Path(${SCORECARD_JSON@Q})
payload = json.loads(path.read_text(encoding="utf-8"))
payload.append(json.loads(${row_json@Q}))
path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
PY
}

submit_profile() {
  local profile="$1"
  local run_name="$2"
  bash scripts/submit_vertex_regression_dynscm_live.sh \
    --project "$PROJECT" \
    --region "$REGION" \
    --bucket "$BUCKET_PREFIX" \
    --run-name "$run_name" \
    --machine-type "$MACHINE_TYPE" \
    --accelerator-type "$ACCELERATOR_TYPE" \
    --accelerator-count "$ACCELERATOR_COUNT" \
    --research-profile "$profile" \
    -- \
    --loadcheckpoint="$COMMON_WARM_START" \
    --warm_start
}

wait_for_job() {
  local profile="$1"
  local run_name="$2"
  local job_id="$3"
  while true; do
    local state
    state="$(gcloud ai custom-jobs describe "$job_id" \
      --project "$PROJECT" \
      --region "$REGION" \
      --format='value(state)')"
    printf '[science] group=%s profile=%s run=%s job=%s state=%s\n' \
      "$GROUP" "$profile" "$run_name" "$job_id" "$state"
    case "$state" in
      JOB_STATE_SUCCEEDED)
        return 0
        ;;
      JOB_STATE_FAILED|JOB_STATE_CANCELLED|JOB_STATE_EXPIRED|JOB_STATE_PAUSED)
        echo "Job failed for ${profile}: ${job_id}" >&2
        return 1
        ;;
      *)
        sleep "$POLL_SECONDS"
        ;;
    esac
  done
}

GROUP=""
RUN_PREFIX="dynscm-science-$(date +%Y%m%d-%H%M%S)"
PROJECT="${VERTEX_PROJECT:-${GCP_PROJECT:-}}"
REGION="${VERTEX_REGION:-}"
BUCKET_INPUT="${VERTEX_BUCKET:-}"
MACHINE_TYPE="${VERTEX_MACHINE_TYPE:-g2-standard-12}"
ACCELERATOR_TYPE="${VERTEX_ACCELERATOR_TYPE:-NVIDIA_L4}"
ACCELERATOR_COUNT="${VERTEX_ACCELERATOR_COUNT:-1}"
PYTHON_BIN_OVERRIDE="${PYTHON_BIN:-}"
PYTHON_CMD=()
WORK_ROOT=""
POLL_SECONDS=60
STREAM_LOGS=1
DRY_RUN=0
COMMON_WARM_START="gs://tfm-forecasting-vertex-artifacts/tfm_forecasting/runs/dynscm-train-only-20260228-212512-medium-missing-16k/checkpoints/best_checkpoint.pth"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --group)
      GROUP="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --group=*)
      GROUP="$(require_value --group "${1#*=}")"
      shift
      ;;
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
    --poll-seconds)
      POLL_SECONDS="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --poll-seconds=*)
      POLL_SECONDS="$(require_value --poll-seconds "${1#*=}")"
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
    --warm-start-checkpoint)
      COMMON_WARM_START="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --warm-start-checkpoint=*)
      COMMON_WARM_START="$(require_value --warm-start-checkpoint "${1#*=}")"
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

if [[ -z "$GROUP" ]]; then
  echo "Error: --group is required." >&2
  exit 2
fi
if [[ -z "$PROJECT" ]]; then
  PROJECT="$(_gcloud_value project || true)"
fi
if [[ -z "$PROJECT" ]]; then
  echo "Error: could not determine project." >&2
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
  echo "Error: could not determine bucket." >&2
  exit 2
fi

case "$GROUP" in
  temporal_ablation)
    PROFILES=(
      temporal_length_only_16k
      temporal_regimes_only_16k
      temporal_drift_only_16k
      temporal_regimes_plus_drift_16k
      temporal_length_plus_regimes_16k
      temporal_full_medium32k_reference
    )
    ;;
  benchmark_contract)
    PROFILES=(
      benchmark_contract_observed_easy
      benchmark_contract_observed_temporal
    )
    ;;
  *)
    echo "Error: unsupported group: $GROUP" >&2
    exit 2
    ;;
esac

BUCKET_PREFIX="$(normalize_gcs_prefix "$BUCKET_INPUT")"
if [[ -z "$WORK_ROOT" ]]; then
  WORK_ROOT="workdir/research/${RUN_PREFIX}"
fi
resolve_python_cmd

mkdir -p "$WORK_ROOT"/{logs,artifacts,eval,analysis}
SCORECARD_TSV="${WORK_ROOT}/scorecard.tsv"
SCORECARD_JSON="${WORK_ROOT}/scorecard.json"
printf 'experiment_group\tprofile\trun_name\twarm_start_checkpoint\ttrain_best_val_loss\ttarget_eval_loss\tstable_eval_loss\ttemporal_eval_loss\ttarget_skipped_fraction\tstable_skipped_fraction\tbenchmark_mismatch_summary\tdecision\tnotes\n' > "$SCORECARD_TSV"
printf '[]\n' > "$SCORECARD_JSON"

if [[ "$DRY_RUN" -eq 1 ]]; then
  for profile in "${PROFILES[@]}"; do
    echo "[dry-run] group=${GROUP} profile=${profile} run_name=$(run_name_for_profile "$RUN_PREFIX" "$profile")"
    echo "[dry-run] warm_start_checkpoint=${COMMON_WARM_START}"
    echo "[dry-run] synthetic_eval=enabled"
    echo "[dry-run] prior_audit=enabled"
    echo "[dry-run] benchmark_compare=enabled"
  done
  echo "[dry-run] scorecard_tsv=${SCORECARD_TSV}"
  echo "[dry-run] scorecard_json=${SCORECARD_JSON}"
  exit 0
fi

for profile in "${PROFILES[@]}"; do
  run_name="$(run_name_for_profile "$RUN_PREFIX" "$profile")"
  submit_log="${WORK_ROOT}/logs/${profile}.submit.log"
  submit_output="$(submit_profile "$profile" "$run_name" | tee "$submit_log")"
  job_resource="$(printf '%s\n' "$submit_output" | tail -n 1)"
  job_id="${job_resource##*/}"

  if [[ "$STREAM_LOGS" -eq 1 ]]; then
    gcloud ai custom-jobs stream-logs "$job_id" \
      --project "$PROJECT" \
      --region "$REGION" || true
  fi
  wait_for_job "$profile" "$run_name" "$job_id"

  checkpoint_uri="${BUCKET_PREFIX}/tfm_forecasting/runs/${run_name}/checkpoints/best_checkpoint.pth"
  local_checkpoint="${WORK_ROOT}/artifacts/${profile}.best_checkpoint.pth"
  gcloud storage cp "$checkpoint_uri" "$local_checkpoint" >/dev/null

  eval_json="${WORK_ROOT}/eval/${profile}.synthetic_eval.json"
  "${PYTHON_CMD[@]}" scripts/eval_dynscm_synthetic_suite.py \
    --checkpoint_path "$local_checkpoint" \
    --research_profile "$profile" \
    --output_json "$eval_json"

  prior_audit_json="${WORK_ROOT}/analysis/${profile}.prior_audit.json"
  benchmark_compare_json="${WORK_ROOT}/analysis/${profile}.benchmark_compare.json"
  benchmark_compare_md="${WORK_ROOT}/analysis/${profile}.benchmark_compare.md"
  "${PYTHON_CMD[@]}" scripts/compare_live_dynscm_profile_to_forecast_benchmark.py \
    --research_profile "$profile" \
    --source train \
    --json-out "$benchmark_compare_json" \
    --markdown-out "$benchmark_compare_md" \
    --prior-audit-json "$prior_audit_json"

  readarray -t metrics < <("${PYTHON_CMD[@]}" - <<PY
import json
from pathlib import Path
import torch

eval_payload = json.loads(Path(${eval_json@Q}).read_text(encoding="utf-8"))
compare_payload = json.loads(Path(${benchmark_compare_json@Q}).read_text(encoding="utf-8"))
ckpt = torch.load(${local_checkpoint@Q}, map_location="cpu")
training = ckpt.get("training", {})
best_val = float(training.get("best_metric", float("nan")))
top = compare_payload["mismatches"][:3]
summary = ";".join(f"{item['dimension']}={float(item['score']):.3f}" for item in top)
print(best_val)
print(float(eval_payload["suites"]["target_eval"]["loss"]))
print(float(eval_payload["suites"]["stable_eval"]["loss"]))
print(float(eval_payload["suites"].get("temporal_eval_hard", eval_payload["suites"].get("temporal_eval", {})).get("loss", float("nan"))))
print(float(eval_payload["suites"]["target_eval"]["skipped_fraction"]))
print(float(eval_payload["suites"]["stable_eval"]["skipped_fraction"]))
print(summary)
PY
  )

  train_best_val_loss="${metrics[0]}"
  target_eval_loss="${metrics[1]}"
  stable_eval_loss="${metrics[2]}"
  temporal_eval_loss="${metrics[3]}"
  target_skipped_fraction="${metrics[4]}"
  stable_skipped_fraction="${metrics[5]}"
  benchmark_mismatch_summary="${metrics[6]}"
  decision="pending_review"
  notes="eval+benchmark_compare complete"

  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$GROUP" "$profile" "$run_name" "$COMMON_WARM_START" "$train_best_val_loss" \
    "$target_eval_loss" "$stable_eval_loss" "$temporal_eval_loss" \
    "$target_skipped_fraction" "$stable_skipped_fraction" \
    "$benchmark_mismatch_summary" "$decision" "$notes" >> "$SCORECARD_TSV"

  row_json="$("${PYTHON_CMD[@]}" - <<PY
import json
print(json.dumps({
    "experiment_group": ${GROUP@Q},
    "profile": ${profile@Q},
    "run_name": ${run_name@Q},
    "warm_start_checkpoint": ${COMMON_WARM_START@Q},
    "train_best_val_loss": float(${train_best_val_loss@Q}),
    "target_eval_loss": float(${target_eval_loss@Q}),
    "stable_eval_loss": float(${stable_eval_loss@Q}),
    "temporal_eval_loss": float(${temporal_eval_loss@Q}),
    "target_skipped_fraction": float(${target_skipped_fraction@Q}),
    "stable_skipped_fraction": float(${stable_skipped_fraction@Q}),
    "benchmark_mismatch_summary": ${benchmark_mismatch_summary@Q},
    "decision": ${decision@Q},
    "notes": ${notes@Q},
}))
PY
  )"
  append_scorecard_json "$row_json"
done

echo "[science] scorecard_tsv=${SCORECARD_TSV}"
echo "[science] scorecard_json=${SCORECARD_JSON}"
