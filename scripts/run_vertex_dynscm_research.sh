#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run_vertex_dynscm_research.sh [OPTIONS]

Run the live DynSCM research branch on Vertex AI, evaluate each checkpoint on the
synthetic suite, and build a scorecard across the five planned experiments.

Options:
  --run-prefix PREFIX       Prefix for all run names (default: dynscm-research-<timestamp>)
  --project PROJECT         GCP project (fallback env/gcloud config)
  --region REGION           Vertex region (fallback env/gcloud ai/region or us-central1)
  --bucket BUCKET           Bucket or gs://bucket[/prefix] for canonical storage root
  --machine-type TYPE       Vertex machine type (default: g2-standard-12)
  --accelerator-type TYPE   GPU type (default: NVIDIA_L4)
  --accelerator-count N     GPU count (default: 1)
  --python BIN              Python executable to use locally (default: auto-detect)
  --workdir DIR             Local work directory (default: workdir/research/<run-prefix>)
  --poll-seconds N          Seconds between Vertex job polls (default: 60)
  --stream-logs             Stream Vertex logs for each stage (default: enabled)
  --no-stream-logs          Disable log streaming
  --stop-on-win             Stop once an experiment satisfies the win rule
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

RUN_PREFIX="dynscm-research-$(date +%Y%m%d-%H%M%S)"
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
STOP_ON_WIN=0
DRY_RUN=0
COMMON_WARM_START="gs://tfm-forecasting-vertex-artifacts/tfm_forecasting/runs/dynscm-train-only-20260228-212512-medium-missing-16k/checkpoints/best_checkpoint.pth"

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
    --stop-on-win)
      STOP_ON_WIN=1
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

BUCKET_PREFIX="$(normalize_gcs_prefix "$BUCKET_INPUT")"
if [[ -z "$WORK_ROOT" ]]; then
  WORK_ROOT="workdir/research/${RUN_PREFIX}"
fi
resolve_python_cmd

EXPERIMENTS=(
  medium32k_live_baseline
  medium32k_live_guardrails
  medium32k_live_batch_homogeneous
  medium32k_live_mode_ladder
  medium32k_live_mixture
)

mkdir -p "$WORK_ROOT"/{logs,artifacts,eval}
SCORECARD_TSV="${WORK_ROOT}/scorecard.tsv"
SCORECARD_JSON="${WORK_ROOT}/scorecard.json"
printf 'profile\trun_name\tjob_id\tcheckpoint\ttarget_eval_loss\tstable_eval_loss\tstable_degradation\tdecision\n' > "$SCORECARD_TSV"
printf '[]\n' > "$SCORECARD_JSON"

run_name_for_profile() {
  local profile="$1"
  case "$profile" in
    medium32k_live_baseline) echo "${RUN_PREFIX}-baseline" ;;
    medium32k_live_guardrails) echo "${RUN_PREFIX}-guardrails" ;;
    medium32k_live_batch_homogeneous) echo "${RUN_PREFIX}-batch-homogeneous" ;;
    medium32k_live_mode_ladder) echo "${RUN_PREFIX}-mode-ladder" ;;
    medium32k_live_mixture) echo "${RUN_PREFIX}-mixture" ;;
    *)
      echo "Error: unknown profile $profile" >&2
      exit 2
      ;;
  esac
}

promotion_run_name_for_profile() {
  local profile="$1"
  echo "$(run_name_for_profile "$profile")-full-promotion"
}

checkpoint_uri_for_run() {
  local run_name="$1"
  echo "${BUCKET_PREFIX}/tfm_forecasting/runs/${run_name}/checkpoints/best_checkpoint.pth"
}

wait_for_job() {
  local job_id="$1"
  local stage="$2"
  local job_resource="projects/${PROJECT}/locations/${REGION}/customJobs/${job_id}"
  local stream_pid=""

  if [[ "$STREAM_LOGS" -eq 1 ]]; then
    gcloud ai custom-jobs stream-logs "$job_resource" &
    stream_pid="$!"
  fi

  while true; do
    local state
    state="$(gcloud ai custom-jobs describe "$job_id" --project "$PROJECT" --region "$REGION" --verbosity=error --format='value(state)')"
    printf '[research] profile=%s job=%s state=%s\n' "$stage" "$job_id" "$state"
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
        return 1
        ;;
      *)
        sleep "$POLL_SECONDS"
        ;;
    esac
  done
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

decision_for_profile() {
  local profile="$1"
  local eval_json="$2"
  "${PYTHON_CMD[@]}" - <<PY
import json
import math
from pathlib import Path
payload = json.loads(Path(${eval_json@Q}).read_text(encoding="utf-8"))
suites = payload["suites"]
target_loss = float(suites["target_eval"]["loss"])
stable_loss = float(suites["stable_eval"]["loss"])
if not (math.isfinite(target_loss) and math.isfinite(stable_loss)):
    print("reject_nonfinite")
    raise SystemExit(0)
name = ${profile@Q}
baseline = float("nan")
best_transition = float("nan")
if Path(${SCORECARD_TSV@Q}).exists():
    rows = Path(${SCORECARD_TSV@Q}).read_text(encoding="utf-8").strip().splitlines()[1:]
    for row in rows:
        if not row:
            continue
        parts = row.split("\t")
        if parts[0] == "medium32k_live_baseline":
            baseline = float(parts[4])
        if parts[0] in {"medium32k_live_guardrails", "medium32k_live_batch_homogeneous"}:
            candidate = float(parts[4])
            if not math.isfinite(best_transition) or candidate < best_transition:
                best_transition = candidate
stable_baseline = stable_loss
if Path(${SCORECARD_TSV@Q}).exists():
    rows = Path(${SCORECARD_TSV@Q}).read_text(encoding="utf-8").strip().splitlines()[1:]
    for row in rows:
        if not row:
            continue
        parts = row.split("\t")
        if parts[0] == "medium32k_live_baseline":
            stable_baseline = float(parts[5])
            break
stable_degradation = ((stable_loss - stable_baseline) / stable_baseline) if stable_baseline > 0 else float("nan")
if name == "medium32k_live_baseline":
    print(f"baseline\t{stable_degradation}")
elif name == "medium32k_live_guardrails":
    rel = ((baseline - target_loss) / baseline) if baseline > 0 else float("nan")
    ok = rel >= 0.02 and (not math.isfinite(stable_degradation) or stable_degradation <= 0.03)
    print(f"{'keep' if ok else 'reject'}\t{stable_degradation}")
elif name == "medium32k_live_batch_homogeneous":
    rel = ((baseline - target_loss) / baseline) if baseline > 0 else float("nan")
    ok = rel >= 0.03 and (not math.isfinite(stable_degradation) or stable_degradation <= 0.03)
    print(f"{'keep' if ok else 'reject'}\t{stable_degradation}")
elif name == "medium32k_live_mode_ladder":
    reference = best_transition if math.isfinite(best_transition) else baseline
    rel = ((reference - target_loss) / reference) if reference > 0 else float("nan")
    ok = rel >= 0.03 and (not math.isfinite(stable_degradation) or stable_degradation <= 0.03)
    print(f"{'keep' if ok else 'reject'}\t{stable_degradation}")
else:
    ok = target_loss <= 0.00410 and (not math.isfinite(stable_degradation) or stable_degradation <= 0.03)
    print(f"{'win' if ok else 'reject'}\t{stable_degradation}")
PY
}

append_findings_note() {
  local profile="$1"
  local checkpoint="$2"
  local eval_json="$3"
  "${PYTHON_CMD[@]}" - <<PY
import json
from pathlib import Path
findings = Path("workdir/forecast_research/experiment_findings.md")
payload = json.loads(Path(${eval_json@Q}).read_text(encoding="utf-8"))
suites = payload["suites"]
profile = ${profile@Q}
checkpoint = ${checkpoint@Q}
entry = (
    f"\\n\\n### Live research run: {profile}\\n"
    f"- Warm-start checkpoint: `{checkpoint}`\\n"
    f"- target_eval loss: `{suites['target_eval']['loss']:.5f}`\\n"
    f"- stable_eval loss: `{suites['stable_eval']['loss']:.5f}`\\n"
    "- Plausible explanations: this branch was evaluated as a research-only "
    "synthetic comparison run; interpret any gain or regression relative to "
    "the medium-missing warm-start and the active baseline, not as proof of "
    "real-data improvement."
)
with findings.open("a", encoding="utf-8") as handle:
    handle.write(entry + "\n")
PY
}

run_promotion_followup() {
  local profile="$1"
  local winner_checkpoint="$2"
  local promotion_profile="${profile}_full_promotion"
  local promotion_run_name
  local promotion_checkpoint_uri
  local promotion_job_id
  local promotion_submit_output
  local promotion_job_resource
  local promotion_local_ckpt
  local promotion_eval_json
  local promotion_target_eval_loss
  local promotion_stable_eval_loss
  local promotion_stable_degradation

  promotion_run_name="$(promotion_run_name_for_profile "$profile")"
  promotion_checkpoint_uri="$(checkpoint_uri_for_run "$promotion_run_name")"
  promotion_submit_output="$(
    bash scripts/submit_vertex_regression_dynscm_live.sh \
      --project "$PROJECT" \
      --region "$REGION" \
      --bucket "$BUCKET_PREFIX" \
      --machine-type "$MACHINE_TYPE" \
      --accelerator-type "$ACCELERATOR_TYPE" \
      --accelerator-count "$ACCELERATOR_COUNT" \
      --research-profile "$profile" \
      --run-name "$promotion_run_name" \
      -- \
      --promote_to_full \
      "--loadcheckpoint=${winner_checkpoint}" \
      --warm_start
  2>&1)"
  printf '%s\n' "$promotion_submit_output" | tee "${WORK_ROOT}/logs/${promotion_profile}.submit.log"
  promotion_job_resource="$(printf '%s\n' "$promotion_submit_output" | grep -o 'projects/[^[:space:]]*/locations/[^[:space:]]*/customJobs/[0-9]\+' | tail -n 1 || true)"
  if [[ -z "$promotion_job_resource" ]]; then
    echo "Error: failed to parse Vertex job id for ${promotion_profile}." >&2
    printf '%s\n' "$promotion_submit_output" >&2
    exit 1
  fi
  promotion_job_id="${promotion_job_resource##*/}"
  wait_for_job "$promotion_job_id" "$promotion_profile"

  promotion_local_ckpt="${WORK_ROOT}/artifacts/${promotion_profile}.best_checkpoint.pth"
  gcloud storage cp "$promotion_checkpoint_uri" "$promotion_local_ckpt"
  promotion_eval_json="${WORK_ROOT}/eval/${promotion_profile}.synthetic_eval.json"
  "${PYTHON_CMD[@]}" scripts/eval_dynscm_synthetic_suite.py \
    --checkpoint_path "$promotion_local_ckpt" \
    --research_profile "$profile" \
    --output_json "$promotion_eval_json"

  promotion_target_eval_loss="$("${PYTHON_CMD[@]}" - <<PY
import json
from pathlib import Path
payload = json.loads(Path(${promotion_eval_json@Q}).read_text(encoding="utf-8"))
print(payload["suites"]["target_eval"]["loss"])
PY
)"
  promotion_stable_eval_loss="$("${PYTHON_CMD[@]}" - <<PY
import json
from pathlib import Path
payload = json.loads(Path(${promotion_eval_json@Q}).read_text(encoding="utf-8"))
print(payload["suites"]["stable_eval"]["loss"])
PY
)"
  promotion_stable_degradation="$("${PYTHON_CMD[@]}" - <<PY
import json
import math
from pathlib import Path
payload = json.loads(Path(${promotion_eval_json@Q}).read_text(encoding="utf-8"))
stable_loss = float(payload["suites"]["stable_eval"]["loss"])
baseline = float("nan")
rows = Path(${SCORECARD_TSV@Q}).read_text(encoding="utf-8").strip().splitlines()[1:]
for row in rows:
    if not row:
        continue
    parts = row.split("\\t")
    if parts[0] == "medium32k_live_baseline":
        baseline = float(parts[5])
        break
if not (math.isfinite(stable_loss) and math.isfinite(baseline) and baseline > 0):
    print("nan")
else:
    print((stable_loss - baseline) / baseline)
PY
)"
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$promotion_profile" "$promotion_run_name" "$promotion_job_id" "$promotion_checkpoint_uri" "$promotion_target_eval_loss" "$promotion_stable_eval_loss" "$promotion_stable_degradation" "promotion" >> "$SCORECARD_TSV"
  promotion_row_json="$("${PYTHON_CMD[@]}" - <<PY
import json
print(json.dumps({
    "profile": ${promotion_profile@Q},
    "run_name": ${promotion_run_name@Q},
    "job_id": ${promotion_job_id@Q},
    "checkpoint": ${promotion_checkpoint_uri@Q},
    "target_eval_loss": float(${promotion_target_eval_loss@Q}),
    "stable_eval_loss": float(${promotion_stable_eval_loss@Q}),
    "stable_degradation": float(${promotion_stable_degradation@Q}) if ${promotion_stable_degradation@Q} not in ("nan", "") else float("nan"),
    "decision": "promotion",
}))
PY
)"
  append_scorecard_json "$promotion_row_json"
  append_findings_note "$promotion_profile" "$winner_checkpoint" "$promotion_eval_json"
}

if [[ "$DRY_RUN" -eq 1 ]]; then
  for profile in "${EXPERIMENTS[@]}"; do
    run_name="$(run_name_for_profile "$profile")"
    echo "[dry-run] profile=${profile}"
    echo "  run_name=${run_name}"
    echo "  warm_start_checkpoint=${COMMON_WARM_START}"
    echo "  synthetic_eval=enabled"
    echo "  submit_script=scripts/submit_vertex_regression_dynscm_live.sh"
  done
  echo "[dry-run] scorecard_tsv=${SCORECARD_TSV}"
  echo "[dry-run] scorecard_json=${SCORECARD_JSON}"
  exit 0
fi

for profile in "${EXPERIMENTS[@]}"; do
  run_name="$(run_name_for_profile "$profile")"
  checkpoint_uri="$(checkpoint_uri_for_run "$run_name")"
  submit_output="$(
    bash scripts/submit_vertex_regression_dynscm_live.sh \
      --project "$PROJECT" \
      --region "$REGION" \
      --bucket "$BUCKET_PREFIX" \
      --machine-type "$MACHINE_TYPE" \
      --accelerator-type "$ACCELERATOR_TYPE" \
      --accelerator-count "$ACCELERATOR_COUNT" \
      --research-profile "$profile" \
      --run-name "$run_name" \
      -- \
      "--loadcheckpoint=${COMMON_WARM_START}" \
      --warm_start
  2>&1)"
  printf '%s\n' "$submit_output" | tee "${WORK_ROOT}/logs/${profile}.submit.log"
  job_resource="$(printf '%s\n' "$submit_output" | grep -o 'projects/[^[:space:]]*/locations/[^[:space:]]*/customJobs/[0-9]\+' | tail -n 1 || true)"
  if [[ -z "$job_resource" ]]; then
    echo "Error: failed to parse Vertex job id for ${profile}." >&2
    printf '%s\n' "$submit_output" >&2
    exit 1
  fi
  job_id="${job_resource##*/}"
  wait_for_job "$job_id" "$profile"

  local_ckpt="${WORK_ROOT}/artifacts/${profile}.best_checkpoint.pth"
  gcloud storage cp "$checkpoint_uri" "$local_ckpt"
  eval_json="${WORK_ROOT}/eval/${profile}.synthetic_eval.json"
  "${PYTHON_CMD[@]}" scripts/eval_dynscm_synthetic_suite.py \
    --checkpoint_path "$local_ckpt" \
    --research_profile "$profile" \
    --output_json "$eval_json"

  target_eval_loss="$("${PYTHON_CMD[@]}" - <<PY
import json
from pathlib import Path
payload = json.loads(Path(${eval_json@Q}).read_text(encoding="utf-8"))
print(payload["suites"]["target_eval"]["loss"])
PY
)"
  stable_eval_loss="$("${PYTHON_CMD[@]}" - <<PY
import json
from pathlib import Path
payload = json.loads(Path(${eval_json@Q}).read_text(encoding="utf-8"))
print(payload["suites"]["stable_eval"]["loss"])
PY
)"
  IFS=$'\t' read -r decision stable_degradation < <(decision_for_profile "$profile" "$eval_json")
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$profile" "$run_name" "$job_id" "$checkpoint_uri" "$target_eval_loss" "$stable_eval_loss" "$stable_degradation" "$decision" >> "$SCORECARD_TSV"
  row_json="$("${PYTHON_CMD[@]}" - <<PY
import json
print(json.dumps({
    "profile": ${profile@Q},
    "run_name": ${run_name@Q},
    "job_id": ${job_id@Q},
    "checkpoint": ${checkpoint_uri@Q},
    "target_eval_loss": float(${target_eval_loss@Q}),
    "stable_eval_loss": float(${stable_eval_loss@Q}),
    "stable_degradation": float(${stable_degradation@Q}) if ${stable_degradation@Q} not in ("nan", "") else float("nan"),
    "decision": ${decision@Q},
}))
PY
)"
  append_scorecard_json "$row_json"
  append_findings_note "$profile" "$COMMON_WARM_START" "$eval_json"

  if [[ "$decision" == "win" ]]; then
    run_promotion_followup "$profile" "$checkpoint_uri"
    if [[ "$STOP_ON_WIN" -eq 1 ]]; then
      echo "[research] stop-on-win triggered by ${profile}"
      break
    fi
  fi
done

echo "[research] scorecard_tsv=${SCORECARD_TSV}"
echo "[research] scorecard_json=${SCORECARD_JSON}"
