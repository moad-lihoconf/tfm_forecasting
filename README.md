# TFM-Playground (Regression)

This repository documents and supports regression workflows only:

- training nanoTabPFN-style models for regression,
- training from static prior dumps,
- training with live DynSCM synthetic generation,
- evaluating regression forecasting benchmarks (including NICL integration).

## Installation

Python `3.12` is expected.

Preferred way: `Poetry` (recommended).

```bash
git clone https://github.com/moad-lihoconf/tfm_forecasting.git
cd tfm_forecasting
poetry install
```

If Poetry is not installed yet, use the official installer (not distro package):

```bash
curl -sSL https://install.python-poetry.org | python3 -
```

Optional fallback (without Poetry):

```bash
git clone https://github.com/moad-lihoconf/tfm_forecasting.git
cd tfm_forecasting
pip install -e .
```

## How To Run (Local And Vertex)

### Local Run (End-to-End)

1. Generate a local DynSCM prior dump:

```bash
python -m tfmplayground.priors --lib dynscm \
  --num_batches 1000 --batch_size 8 \
  --max_seq_len 64 --max_features 128 \
  --max_classes 0 \
  --dynscm_workers 4 \
  --dynscm_worker_blas_threads 1 \
  --save_path workdir/dynscm_prior_64x128.h5
```

2. Train locally from the dump:

```bash
python pretrain_regression.py \
  --priordump workdir/dynscm_prior_64x128.h5 \
  --saveweights workdir/nanotabpfn_reg_weights.pth \
  --savebuckets workdir/nanotabpfn_reg_buckets.pth
```

3. Run local benchmark:

```bash
python scripts/final_real_benchmark_standard_vs_nicl.py \
  --mode full \
  --output_dir workdir/forecast_results_standard_vs_nicl_py \
  --max_series_per_dataset 8 \
  --append_stats
```

### Vertex Run (End-to-End)

1. Configure GCP and defaults:

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project <PROJECT_ID>
gcloud config set ai/region <REGION>
gcloud services enable aiplatform.googleapis.com storage.googleapis.com artifactregistry.googleapis.com
export VERTEX_BUCKET=gs://<BUCKET_NAME>
```

2. Build and push GPU image:

```bash
bash scripts/update_docker_gpu.sh --push
```

3. Submit Vertex training:

```bash
bash scripts/submit_vertex_regression.sh \
  --priordump gs://<BUCKET_NAME>/tfm_forecasting/priors/dynscm_prior.h5 \
  --run-name vertex-reg-001
```

4. Sync run outputs locally:

```bash
bash scripts/sync_vertex_outputs.sh \
  gs://<BUCKET_NAME>/tfm_forecasting/runs/vertex-reg-001
```

## Docker GPU Image (For Vertex Jobs)

The Vertex submission scripts expect a GPU training image from
[Dockerfile.gpu](/home/mouad/Desktop/dev_projects/tfm_forecasting/Dockerfile.gpu).

One-time auth to Artifact Registry:

```bash
gcloud auth configure-docker <REGION>-docker.pkg.dev
```

Build and push the default image URI (auto-resolved from env or `gcloud config`):

```bash
bash scripts/update_docker_gpu.sh --push
```

Optional explicit overrides:

```bash
export VERTEX_PROJECT="<PROJECT_ID>"
export VERTEX_REGION="<REGION>"
export AR_REPOSITORY="tfm-forecasting"
export IMAGE_NAME="trainer-gpu"
export IMAGE_TAG="latest"
bash scripts/update_docker_gpu.sh --push
```

## Quickstart: Regressor Inference

```python
from sklearn.datasets import load_diabetes
from sklearn.metrics import root_mean_squared_error
from sklearn.model_selection import train_test_split

from tfmplayground import NanoTabPFNRegressor

X, y = load_diabetes(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.5, random_state=42
)

reg = NanoTabPFNRegressor()
reg.fit(X_train, y_train)
pred = reg.predict(X_test)

print("RMSE:", root_mean_squared_error(y_test, pred))
```

## Train From Regression Prior Dump

Main script: [pretrain_regression.py](/home/mouad/Desktop/dev_projects/tfm_forecasting/pretrain_regression.py)

Example:

```bash
python pretrain_regression.py \
  --priordump path/to/50x3_1280k_regression.h5 \
  --epochs 80 \
  --steps 25 \
  --batchsize 50 \
  --accumulate 1 \
  --saveweights workdir/nanotabpfn_reg_weights.pth \
  --savebuckets workdir/nanotabpfn_reg_buckets.pth
```

Notes:

- `--priordump` supports local paths and `gs://...`.
- `--saveweights` and `--savebuckets` also support local paths and `gs://...`.
- Early stopping, target normalization, feature normalization, and integrity checks are configurable in the CLI flags.

## DynSCM Prior Generation (Regression)

Generate a DynSCM dump:

```bash
python -m tfmplayground.priors --lib dynscm \
  --num_batches 1000 --batch_size 8 \
  --max_seq_len 64 --max_features 128 \
  --max_classes 0 \
  --dynscm_workers 4 \
  --dynscm_worker_blas_threads 1 \
  --save_path workdir/dynscm_prior_64x128.h5
```

## Live DynSCM Regression Training

Main script: [pretrain_regression_dynscm_live.py](/home/mouad/Desktop/dev_projects/tfm_forecasting/pretrain_regression_dynscm_live.py)

List available research profiles:

```bash
poetry run python - <<'PY'
from tfmplayground.priors.dynscm.research_profiles import list_research_profiles
print("\n".join(list_research_profiles()))
PY
```

Run a live profile:

```bash
python pretrain_regression_dynscm_live.py \
  --research_profile medium32k_live_mode_ladder \
  --runname dynscm-live-regression \
  --saveweights workdir/forecast_research/nanotabpfn_dynscm_weights.pth \
  --savebuckets workdir/forecast_research/nanotabpfn_dynscm_buckets.pth
```

## Vertex AI Setup (Regression Training)

Minimal prerequisites:

```bash
gcloud auth login
gcloud auth application-default login
gcloud config set project <PROJECT_ID>
gcloud config set ai/region <REGION>
gcloud services enable aiplatform.googleapis.com storage.googleapis.com artifactregistry.googleapis.com
```

Set a default bucket used by submit scripts:

```bash
export VERTEX_BUCKET=gs://<BUCKET_NAME>
```

Submit regression training from a prior dump:

```bash
# Dry-run first
bash scripts/submit_vertex_regression.sh \
  --priordump gs://<BUCKET_NAME>/tfm_forecasting/priors/dynscm_prior.h5 \
  --run-name vertex-reg-001 \
  --dry-run

# Real submit
bash scripts/submit_vertex_regression.sh \
  --priordump gs://<BUCKET_NAME>/tfm_forecasting/priors/dynscm_prior.h5 \
  --run-name vertex-reg-001
```

Submit live DynSCM regression training:

```bash
# Dry-run first
bash scripts/submit_vertex_regression_dynscm_live.sh \
  --research-profile medium32k_live_mode_ladder \
  --run-name vertex-reg-live-001 \
  --dry-run

# Real submit
bash scripts/submit_vertex_regression_dynscm_live.sh \
  --research-profile medium32k_live_mode_ladder \
  --run-name vertex-reg-live-001
```

Sync run artifacts back to local workspace:

```bash
bash scripts/sync_vertex_outputs.sh \
  gs://<BUCKET_NAME>/tfm_forecasting/runs/vertex-reg-001
```

For a full runbook, see
[loc_vertex_setup.md](/home/mouad/Desktop/dev_projects/tfm_forecasting/loc_vertex_setup.md).

## Regression Forecast Benchmark

### Preflight

Validate benchmark readiness:

```bash
python scripts/check_forecast_final_ready.py \
  --config configs/forecast_bench_final_3model.json
```

### NICL API Smoke Test

```bash
python scripts/nicl_api_smoke.py \
  --url https://api.prediction.neuralk-ai.com/api/v1/inference
```

### Final Standard vs NICL Run

Set your API key:

```bash
export NEURALK_API_KEY="..."
```

Run benchmark and append statistical baseline rows:

```bash
python scripts/final_real_benchmark_standard_vs_nicl.py \
  --mode full \
  --output_dir workdir/forecast_results_standard_vs_nicl_py \
  --max_series_per_dataset 8 \
  --append_stats
```

Primary artifacts:

- `workdir/forecast_results_standard_vs_nicl_py/regression_rows_standard_vs_nicl.csv`
- `workdir/forecast_results_standard_vs_nicl_py/per_dataset_perf_standard_vs_nicl.csv`
- `workdir/forecast_results_standard_vs_nicl_py/status_counts_standard_vs_nicl.csv`
- `workdir/forecast_results_standard_vs_nicl_py/regression_rows_standard_vs_nicl_plus_stats.csv`

## Regression File Map

- [pretrain_regression.py](/home/mouad/Desktop/dev_projects/tfm_forecasting/pretrain_regression.py): main local/GCS training entrypoint from a prior dump.
- [pretrain_regression_dynscm_live.py](/home/mouad/Desktop/dev_projects/tfm_forecasting/pretrain_regression_dynscm_live.py): live DynSCM regression training entrypoint (no pre-generated dump required).
- [tfmplayground/benchmarks/forecasting/adapters.py](/home/mouad/Desktop/dev_projects/tfm_forecasting/tfmplayground/benchmarks/forecasting/adapters.py): forecasting adapters and shared featurization logic, including NICL regression integration.
- [scripts/final_real_benchmark_standard_vs_nicl.py](/home/mouad/Desktop/dev_projects/tfm_forecasting/scripts/final_real_benchmark_standard_vs_nicl.py): final benchmark runner used to generate `standard vs NICL` regression result tables.
- [scripts/check_forecast_final_ready.py](/home/mouad/Desktop/dev_projects/tfm_forecasting/scripts/check_forecast_final_ready.py): preflight contract check for config, datasets, model files, and NICL endpoint/token before final run.
- [scripts/nicl_api_smoke.py](/home/mouad/Desktop/dev_projects/tfm_forecasting/scripts/nicl_api_smoke.py): quick NICL connectivity/auth sanity test.
- [configs/forecast_bench_final_3model.json](/home/mouad/Desktop/dev_projects/tfm_forecasting/configs/forecast_bench_final_3model.json): canonical final benchmark config (dataset/model/protocol contract).
- [notebooks/final_real_benchmark_standard_vs_nicl.ipynb](/home/mouad/Desktop/dev_projects/tfm_forecasting/notebooks/final_real_benchmark_standard_vs_nicl.ipynb): analysis notebook for interpreting/exporting final regression benchmark outputs.
- [scripts/update_docker_gpu.sh](/home/mouad/Desktop/dev_projects/tfm_forecasting/scripts/update_docker_gpu.sh): builds and optionally pushes the Vertex GPU training image.
- [scripts/submit_vertex_regression.sh](/home/mouad/Desktop/dev_projects/tfm_forecasting/scripts/submit_vertex_regression.sh): submits Vertex custom job for regression training from a prior dump.
- [scripts/submit_vertex_regression_dynscm_live.sh](/home/mouad/Desktop/dev_projects/tfm_forecasting/scripts/submit_vertex_regression_dynscm_live.sh): submits Vertex custom job for live DynSCM regression training.
- [scripts/sync_vertex_outputs.sh](/home/mouad/Desktop/dev_projects/tfm_forecasting/scripts/sync_vertex_outputs.sh): syncs Vertex run artifacts from GCS back to `workdir/vertex_runs/...`.
- [tests/test_forecast_final_preflight.py](/home/mouad/Desktop/dev_projects/tfm_forecasting/tests/test_forecast_final_preflight.py): regression tests for preflight checks and failure conditions.
- [tests/test_forecast_bench_adapters.py](/home/mouad/Desktop/dev_projects/tfm_forecasting/tests/test_forecast_bench_adapters.py): regression tests for adapter behavior, leakage-safety assumptions, and NICL adapter modes.
- [tests/test_dynscm_prior_analysis_scripts.py](/home/mouad/Desktop/dev_projects/tfm_forecasting/tests/test_dynscm_prior_analysis_scripts.py): regression tests for DynSCM prior analysis scripts used to validate benchmark alignment/invariants.
