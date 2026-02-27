# TFM-Playground

The purpose of this repository is to provide a fully open source playground for tabular foundation models.
It contains a much smaller and simpler implementation of the TabPFNv2 architecture (nanoTabPFN) as well as a training loop, multiple interfaces to load prior data and an evaluation pipeline. We are planning to rapidly extend the repository with more features, prior interfaces and architectures.
It is supposed to be a good starting point for students and researchers that are interested in learning about how Tabular foundation models work under the hood.

Clone the repository, afterwards install dependencies via:
```
pip install -e .
```

We offer the same interface as TabPFN:
```python
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

from tfmplayground import NanoTabPFNClassifier

# Load data
X, y = load_breast_cancer(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.5, random_state=42)

# Initialize a classifier
clf = NanoTabPFNClassifier()
clf.fit(X_train, y_train)

# Predict probabilities
prediction_probabilities = clf.predict_proba(X_test)
print("ROC AUC:", roc_auc_score(y_test, prediction_probabilities[:, 1]))

# Predict labels
predictions = clf.predict(X_test)
print("Accuracy", accuracy_score(y_test, predictions))
```

### Our Code

`tfmplayground/model.py` contains the implementation of the architecture in less than 250 lines of code. `tfmplayground/train.py` implements a simple training loop in under 100 lines and `tfmplayground/priors.py` implements a dataloader that allows you to load a dump pre-generated from a prior.
We will release multiple dumps of different scales soon. We also offer an interface where you can provide your own get\_batch function.

### Pretrain your own small nanoTabPFN
First we download 100k pre-generated datasets with 50 datapoints, 3 features and up to 3 classes each from [here](https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/50x3_3_100k_classification.h5).

Then you can run:
```
python pretrain_classification.py --epochs 80 --steps 25 --batchsize 50 --priordump 50x3_3_100k_classification.h5
```
This should take less than 5 min on a modern NVIDIA GPU (around 10 minutes on Macbook M4 Pro GPU and around 40 min on M4 Pro CPU).

We also offer a pre-generated dataset containing 1.28M tables with 50 datapoints and 3 features each for regression [here](https://ml.informatik.uni-freiburg.de/research-artifacts/pfefferle/TFM-Playground/50x3_1280k_regression.h5).

You can pretrain on it using `python pretrain_regression.py`.

#### Step by Step Explanation (Classifier)

First we import our Architecture, Prior interface and training loop, etc.
```python
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors import PriorDumpDataLoader
from tfmplayground.train import train
from tfmplayground.utils import get_default_device
from tfmplayground.interface import NanoTabPFNClassifier
from tfmplayground.callbacks import ConsoleLoggerCallback

from torch.nn import CrossEntropyLoss
```
then we instantiate our model and loss criterion:
```python
model = NanoTabPFNModel(
    num_attention_heads=6,
    embedding_size=192,
    mlp_hidden_size=768,
    num_layers=6,
    num_outputs=10,
)
criterion = CrossEntropyLoss()
```
then we instantiate our prior:
```python
device = get_default_device()
prior = PriorDumpDataLoader(filename='50x3_3_100k_classification.h5', num_steps=25, batch_size=50, device=device)
```
and finally train our model:
```python
trained_model, loss = train(
    model=model,
    prior=prior,
    criterion=criterion,
    epochs=80,
    device=device,
    callbacks=[ConsoleLoggerCallback()]
)
```

### Creating your own datasets
Check out [tfmplayground.priors](https://github.com/automl/TFM-Playground/tree/main/tfmplayground/priors) to create your own data using publicly available priors.

You can use tfmplayground.priors as a command-line-tool to pre-generate data from a prior, e.g. via
```
python -m tfmplayground.priors --lib tabicl \
       --prior_type mix_scm \
       --num_batches 1000 --batch_size 4 \
       --min_features 3 --max_features 3 \
       --max_seq_len 50 --max_classes 3 \
       --save_path tabicl_4k_50x3.h5
```
For DynSCM forecasting prior dumps, run:
```
python -m tfmplayground.priors --lib dynscm \
       --num_batches 1000 --batch_size 4 \
       --max_seq_len 64 --max_features 128 \
       --max_classes 0 \
       --dynscm_workers 1 \
       --dynscm_worker_blas_threads 1 \
       --no_dynscm_compute_spectral_diagnostics \
       --dynscm_override num_variables_min=4 \
       --dynscm_override num_variables_max=8 \
       --dynscm_override num_regimes=3 \
       --dynscm_override mechanism_type=\"linear_var\" \
       --save_path dynscm_4k_64x128.h5
```
You can optionally provide a full DynSCM JSON config and then patch fields with overrides:
```
python -m tfmplayground.priors --lib dynscm \
       --num_batches 250 --batch_size 8 \
       --max_seq_len 96 --max_features 128 \
       --dynscm_workers 4 \
       --dynscm_worker_blas_threads 1 \
       --dynscm_config_json path/to/dynscm_config.json \
       --dynscm_override features.num_kernels=2 \
       --dynscm_override missingness.missing_mode=\"mix\"
```
which can afterwards be loaded via
```python
from tfmplayground.priors.dataloader import PriorDumpDataLoader
prior = PriorDumpDataLoader('tabicl_4k_50x3.h5', num_steps=20, batch_size=4, device='cpu')
```
You can also just let it create the data on-the-fly via:
```python
from tfmplayground.priors.dataloader import TabICLPriorDataLoader
prior = TabICLPriorDataLoader(
    num_steps=20,
    batch_size=4,
    num_datapoints_max=50,
    min_features=3,
    max_features=3,
    max_num_classes=3,
    device='cpu'
)
```
You can check out `next(iter(prior))` if you want to see an example batch.

Check out `prior_visualization.ipynb` for some more examples.

### DynSCM: Theory to Code Mapping

The DynSCM forecasting prior in `tfmplayground/priors/dynscm/` follows a direct module split:

- `config.py`: all sampling knobs grouped by shape/regime/graph/mechanism/stability/noise/missingness/features/safety.
- `graph.py`: regime-dependent causal graph sampling (contemporaneous + lagged supports).
- `stability.py`: stable coefficient sampling, optional spectral rescaling, optional spectral diagnostics.
- `mechanisms.py`: linear VAR core + optional residual nonlinear mechanism block.
- `simulate.py`: forward rollout for multivariate regime-switching time series.
- `missingness.py`: raw observation mask generation (`off`, `mcar`, `mar`, `mnar_lite`, `mix`) with outages.
- `features.py`: origin/horizon sampling and forecasting featurization into PFN tables.
- `get_batch.py`: full batch assembly, feature-priority truncation, and model contract output.

### DynSCM Config Knobs (High Signal)

- Shape: `num_variables_min/max`, `series_length_min/max`, `train_rows_min/max`, `test_rows_min/max`, `forecast_horizons`.
- Regime: `num_regimes`, `sticky_rho`, `shared_order`, `share_base_graph`.
- Graph: `max_contemp_parents`, `max_lagged_parents`, edge rates/probabilities.
- Mechanism/Stability: `mechanism_type`, `residual_num_features`, `residual_lipschitz_max`, `col_budget_min/max`.
- Missingness: `missing_mode`, `missing_rate_min/max`, block outage controls, `add_mask_channels`.
- Features: `max_feature_lag`, `explicit_lags`, `num_kernels`, deterministic time/season/horizon toggles.
- Safety: `max_abs_x`, `max_resample_attempts`.

### DynSCM Quickstart Commands

- On-the-fly loader:
```python
from tfmplayground.priors import DynSCMConfig, DynSCMPriorDataLoader

prior = DynSCMPriorDataLoader(
    cfg=DynSCMConfig(),
    num_steps=100,
    batch_size=16,
    num_datapoints_max=64,
    num_features=128,
    device="cpu",
    seed=0,
)
batch = next(iter(prior))
```
- Dump to HDF5:
```bash
python -m tfmplayground.priors --lib dynscm \
  --num_batches 500 --batch_size 8 \
  --max_seq_len 64 --max_features 128 \
  --dynscm_workers 4 \
  --dynscm_worker_blas_threads 1 \
  --no_dynscm_compute_spectral_diagnostics \
  --max_classes 0 \
  --save_path dynscm_dump.h5
```
- Throughput benchmark:
```bash
python scripts/benchmark_dynscm_generation.py \
  --num_batches 20 --batch_size 4 \
  --max_seq_len 64 --max_features 128 \
  --workers 1 --worker_blas_threads 1 \
  --seed 0 --profile_top 20
```
- Train from dump:
```python
from tfmplayground.priors.dataloader import PriorDumpDataLoader
prior = PriorDumpDataLoader("dynscm_dump.h5", num_steps=20, batch_size=8, device="cpu")
```

### Forecast Research Validation (DynSCM vs Baselines)

The forecasting benchmark package lives in `tfmplayground/benchmarks/forecasting/` and provides:

- dataset loading with deterministic cache under `workdir/forecast_data/`,
- leakage-safe rolling-origin splits,
- shared featurization for all models (DynSCM feature builder wrapper),
- baseline adapters (nanoTabPFN standard + DynSCM-trained + TabICL),
- proxy classification track (NanoTabPFN classifier + TabICL classifier + NICL API),
- statistical summary with win-rate and bootstrap confidence intervals.

Quickstart benchmark command:
```bash
python -m tfmplayground.benchmarks.forecasting \
  --mode both \
  --model_standard_ckpt nanotabpfn_weights.pth \
  --model_standard_dist nanotabpfn_buckets.pth \
  --model_dynscm_ckpt workdir/forecast_research/nanotabpfn_dynscm_weights.pth \
  --model_dynscm_dist workdir/forecast_research/nanotabpfn_dynscm_buckets.pth \
  --output_dir workdir/forecast_results
```

One-command reproducibility script:
```bash
bash scripts/run_forecast_research.sh
```
This script executes:
1. DynSCM prior dump generation.
2. DynSCM fine-tuning of nanoTabPFN.
3. Proxy benchmark run (NICL used if `NICL_API_TOKEN` is set).
4. Combined regression + proxy benchmark run with final report generation.

Generated artifacts:

- `workdir/forecast_results/regression_rows.csv`
- `workdir/forecast_results/regression_summary.json`
- `workdir/forecast_results/proxy_rows.csv`
- `workdir/forecast_results/proxy_summary.json`
- `workdir/forecast_results/report.md`

### Vertex GPU Training (DynSCM Regression)

This repository supports a GCS-canonical workflow so synthetic data can be generated
locally or in cloud, then trained on Vertex GPU with the same paths.

1. Build and optionally push the GPU image:
```bash
export VERTEX_PROJECT="your-gcp-project"
export VERTEX_REGION="us-central1"
bash scripts/update_docker_gpu.sh --push
```
This builds and pushes `trainer-gpu:latest` by default. If you pass `--tag`,
the script also mirrors that same image to `:latest`.

2. Generate a DynSCM prior directly to GCS (or generate locally and upload):
```bash
python -m tfmplayground.priors --lib dynscm \
  --num_batches 200 --batch_size 8 \
  --max_seq_len 64 --max_features 128 \
  --max_classes 0 \
  --save_path gs://your-bucket/tfm_forecasting/priors/dynscm_prior.h5
```

3. Submit a Vertex GPU regression job (dry-run first):
```bash
bash scripts/submit_vertex_regression.sh \
  --priordump gs://your-bucket/tfm_forecasting/priors/dynscm_prior.h5 \
  --run-name dynscm-vertex-regression \
  --dry-run
```
Then remove `--dry-run` to submit.

4. Sync run outputs back to local:
```bash
bash scripts/sync_vertex_outputs.sh \
  gs://your-bucket/tfm_forecasting/runs/dynscm-vertex-regression
```

Canonical storage layout:
- Priors: `gs://<bucket>/tfm_forecasting/priors/<name>.h5`
- Run outputs: `gs://<bucket>/tfm_forecasting/runs/<run_name>/...`
- Local mirror: `workdir/vertex_runs/<run_name>/...`

### Supported Priors

- [TabICL](https://github.com/soda-inria/tabicl) (Classification)
- [TICL](https://github.com/microsoft/ticl) (Regression, Classification)
- DynSCM forecasting prior (Regression)
