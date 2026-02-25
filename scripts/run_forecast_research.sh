#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python}"
WORKDIR="${WORKDIR:-workdir/forecast_research}"
mkdir -p "${WORKDIR}"

DYN_DUMP="${DYN_DUMP:-${WORKDIR}/dynscm_prior.h5}"
DYN_CKPT="${DYN_CKPT:-${WORKDIR}/nanotabpfn_dynscm_weights.pth}"
DYN_BUCKETS="${DYN_BUCKETS:-${WORKDIR}/nanotabpfn_dynscm_buckets.pth}"
STANDARD_CKPT="${STANDARD_CKPT:-nanotabpfn_weights.pth}"
STANDARD_BUCKETS="${STANDARD_BUCKETS:-nanotabpfn_buckets.pth}"
BENCH_OUT="${BENCH_OUT:-${WORKDIR}/benchmark}"
DYN_WORKERS="${DYN_WORKERS:-1}"
DYN_WORKER_BLAS_THREADS="${DYN_WORKER_BLAS_THREADS:-1}"

echo "[1/4] Generate DynSCM prior dump"
"${PYTHON_BIN}" -m tfmplayground.priors --lib dynscm \
  --num_batches 200 --batch_size 8 \
  --max_seq_len 64 --max_features 128 \
  --dynscm_workers "${DYN_WORKERS}" \
  --dynscm_worker_blas_threads "${DYN_WORKER_BLAS_THREADS}" \
  --no_dynscm_compute_spectral_diagnostics \
  --max_classes 0 \
  --save_path "${DYN_DUMP}"

echo "[2/4] Train nanoTabPFN on DynSCM dump"
"${PYTHON_BIN}" "${REPO_ROOT}/pretrain_regression.py" \
  --priordump "${DYN_DUMP}" \
  --saveweights "${DYN_CKPT}" \
  --savebuckets "${DYN_BUCKETS}" \
  --steps 20 --epochs 1 --batchsize 2 --n_buckets 32

echo "[3/4] Run proxy benchmark (includes NICL if NICL_API_TOKEN is set)"
"${PYTHON_BIN}" -m tfmplayground.benchmarks.forecasting \
  --mode proxy \
  --output_dir "${BENCH_OUT}"

echo "[4/4] Run combined regression + proxy benchmark and generate final report"
"${PYTHON_BIN}" -m tfmplayground.benchmarks.forecasting \
  --mode both \
  --model_standard_ckpt "${STANDARD_CKPT}" \
  --model_standard_dist "${STANDARD_BUCKETS}" \
  --model_dynscm_ckpt "${DYN_CKPT}" \
  --model_dynscm_dist "${DYN_BUCKETS}" \
  --output_dir "${BENCH_OUT}"

echo "Done. Artifacts in ${BENCH_OUT}"
