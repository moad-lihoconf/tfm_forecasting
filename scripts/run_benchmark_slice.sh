#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Usage: scripts/run_benchmark_slice.sh --ckpt <path> [OPTIONS]

Runs a small fixed forecasting benchmark slice to gate curriculum stages.

Options:
  --ckpt PATH           Path to DynSCM model checkpoint (required)
  --dist PATH           Optional path to DynSCM bucket edges artifact
  --output_dir DIR      Output directory (default: workdir/forecast_results/slice)
  --device DEV          cpu|cuda (default: cpu)
  --max-series N        Max series per dataset (default: 64)
  --datasets CSV        Comma-separated dataset names (default: exchange_rate,ettm1)
  --python BIN          Python executable to use (default: auto-detect)
  -h, --help            Show help

Notes:
  - This runs only the regression track.
  - Enabled models: nanotabpfn_standard, nanotabpfn_dynscm
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

resolve_python_cmd() {
  local override="$1"
  if [[ -n "$override" ]]; then
    if ! command -v "$override" >/dev/null 2>&1; then
      echo "Error: requested python executable not found: $override" >&2
      exit 2
    fi
    PYTHON_CMD=("$override")
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

CKPT=""
DIST=""
OUTPUT_DIR="workdir/forecast_results/slice"
DEVICE="cpu"
MAX_SERIES=64
DATASETS="exchange_rate,ettm1"
PYTHON_BIN_OVERRIDE=""
PYTHON_CMD=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --ckpt)
      CKPT="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --ckpt=*)
      CKPT="$(require_value --ckpt "${1#*=}")"
      shift
      ;;
    --dist)
      DIST="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --dist=*)
      DIST="$(require_value --dist "${1#*=}")"
      shift
      ;;
    --output_dir)
      OUTPUT_DIR="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --output_dir=*)
      OUTPUT_DIR="$(require_value --output_dir "${1#*=}")"
      shift
      ;;
    --device)
      DEVICE="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --device=*)
      DEVICE="$(require_value --device "${1#*=}")"
      shift
      ;;
    --max-series)
      MAX_SERIES="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --max-series=*)
      MAX_SERIES="$(require_value --max-series "${1#*=}")"
      shift
      ;;
    --datasets)
      DATASETS="$(require_value "$1" "${2-}")"
      shift 2
      ;;
    --datasets=*)
      DATASETS="$(require_value --datasets "${1#*=}")"
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

if [[ -z "$CKPT" ]]; then
  echo "Error: --ckpt is required." >&2
  exit 2
fi

resolve_python_cmd "$PYTHON_BIN_OVERRIDE"

tmp_cfg="$(mktemp -t tfm_bench_slice_XXXX.json)"
trap 'rm -f "$tmp_cfg"' EXIT

IFS=',' read -r -a dataset_arr <<<"$DATASETS"
{
  echo "{"
  echo "  \"mode\": \"regression\","
  echo "  \"datasets\": {"
  echo "    \"suite_name\": \"medium\","
  echo "    \"dataset_names\": ["
  for i in "${!dataset_arr[@]}"; do
    d="${dataset_arr[$i]}"
    d="${d//\"/}"
    if (( i > 0 )); then
      echo "      ,\"${d}\""
    else
      echo "      \"${d}\""
    fi
  done
  echo "    ],"
  echo "    \"max_series_per_dataset\": ${MAX_SERIES},"
  echo "    \"allow_download\": true"
  echo "  }"
  echo "}"
} > "$tmp_cfg"

cmd=(
  "${PYTHON_CMD[@]}"
  -m tfmplayground.benchmarks.forecasting
  --mode regression
  --config "$tmp_cfg"
  --enabled_regression_models nanotabpfn_standard nanotabpfn_dynscm
  --model_dynscm_ckpt "$CKPT"
  --output_dir "$OUTPUT_DIR"
  --device "$DEVICE"
)
if [[ -n "$DIST" ]]; then
  cmd+=(--model_dynscm_dist "$DIST")
fi

printf '[benchmark-slice] %s\n' "${cmd[*]}"
exec "${cmd[@]}"

