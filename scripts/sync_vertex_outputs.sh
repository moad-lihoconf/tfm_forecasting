#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 gs://bucket/tfm_forecasting/runs/<run_name> [local_dest]" >&2
  exit 1
fi

SRC_RAW="$1"
DEST_RAW="${2:-}"

if [[ "$SRC_RAW" != gs://* ]]; then
  echo "Error: source must be a gs:// path" >&2
  exit 1
fi

SRC="${SRC_RAW%/}"
RUN_NAME="${SRC##*/}"

if [[ -z "$DEST_RAW" ]]; then
  DEST="workdir/vertex_runs/${RUN_NAME}"
else
  DEST="$DEST_RAW"
fi

if command -v gsutil >/dev/null 2>&1; then
  SYNC_CMD=(gsutil -m rsync -r "${SRC}/" "${DEST}/")
  EXISTS_CMD=(gsutil ls "${SRC}/")
elif command -v gcloud >/dev/null 2>&1; then
  # gcloud storage rsync mirrors gsutil behavior on modern SDKs.
  SYNC_CMD=(gcloud storage rsync --recursive "${SRC}" "${DEST}")
  EXISTS_CMD=(gcloud storage ls "${SRC}")
else
  echo "Error: neither gsutil nor gcloud is available on PATH." >&2
  exit 1
fi

if ! "${EXISTS_CMD[@]}" >/dev/null 2>&1; then
  echo "Error: GCS path does not exist: ${SRC}/" >&2
  exit 1
fi

mkdir -p "${DEST}"
echo "Syncing run outputs from ${SRC}/ to ${DEST}/"
"${SYNC_CMD[@]}"
printf 'Done. Local outputs at: %s\n' "${DEST}"
