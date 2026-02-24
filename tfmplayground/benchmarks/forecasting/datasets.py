"""Dataset loading utilities for forecasting benchmarks."""

from __future__ import annotations

import gzip
from dataclasses import dataclass
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from .config import ForecastBenchmarkConfig

__all__ = [
    "DEFAULT_DATASET_SPECS",
    "DatasetBundle",
    "DatasetSpec",
    "load_suite",
]


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    """Metadata and source details for one benchmark dataset."""

    name: str
    frequency: str
    seasonality: int
    cache_file: str
    source_url: str | None = None
    notes: str | None = None


@dataclass(frozen=True, slots=True)
class DatasetBundle:
    """Loaded dataset content ready for benchmark evaluation."""

    name: str
    series: np.ndarray  # shape: (num_series, T)
    frequency: str
    seasonality: int
    skipped: bool = False
    skip_reason: str | None = None


DEFAULT_DATASET_SPECS: dict[str, DatasetSpec] = {
    "m3_monthly": DatasetSpec(
        name="m3_monthly",
        frequency="monthly",
        seasonality=12,
        cache_file="m3_monthly.npz",
        source_url=None,
        notes=(
            "Provide cached npz manually at workdir/forecast_data/m3_monthly.npz "
            "with key 'series' and shape (num_series, T)."
        ),
    ),
    "m3_quarterly": DatasetSpec(
        name="m3_quarterly",
        frequency="quarterly",
        seasonality=4,
        cache_file="m3_quarterly.npz",
        source_url=None,
        notes="Provide cached npz manually (key='series').",
    ),
    "m4_monthly": DatasetSpec(
        name="m4_monthly",
        frequency="monthly",
        seasonality=12,
        cache_file="m4_monthly.npz",
        source_url=None,
        notes="Provide cached npz manually (key='series').",
    ),
    "m4_weekly": DatasetSpec(
        name="m4_weekly",
        frequency="weekly",
        seasonality=52,
        cache_file="m4_weekly.npz",
        source_url=None,
        notes="Provide cached npz manually (key='series').",
    ),
    "tourism_monthly": DatasetSpec(
        name="tourism_monthly",
        frequency="monthly",
        seasonality=12,
        cache_file="tourism_monthly.npz",
        source_url=None,
        notes="Provide cached npz manually (key='series').",
    ),
    "tourism_quarterly": DatasetSpec(
        name="tourism_quarterly",
        frequency="quarterly",
        seasonality=4,
        cache_file="tourism_quarterly.npz",
        source_url=None,
        notes="Provide cached npz manually (key='series').",
    ),
    "ettm1": DatasetSpec(
        name="ettm1",
        frequency="15min",
        seasonality=96,
        cache_file="ettm1.npz",
        source_url=(
            "https://raw.githubusercontent.com/zhouhaoyi/ETDataset/main/"
            "ETT-small/ETTm1.csv"
        ),
    ),
    "exchange_rate": DatasetSpec(
        name="exchange_rate",
        frequency="daily",
        seasonality=7,
        cache_file="exchange_rate.npz",
        source_url=(
            "https://raw.githubusercontent.com/laiguokun/"
            "multivariate-time-series-data/master/exchange_rate/exchange_rate.txt.gz"
        ),
    ),
}


def load_suite(cfg: ForecastBenchmarkConfig) -> dict[str, DatasetBundle]:
    """Load all configured datasets with deterministic caching and skip reasons."""
    cache_dir = Path(cfg.datasets.cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    bundles: dict[str, DatasetBundle] = {}
    for dataset_name in cfg.datasets.dataset_names:
        spec = DEFAULT_DATASET_SPECS.get(dataset_name)
        if spec is None:
            bundles[dataset_name] = DatasetBundle(
                name=dataset_name,
                series=np.zeros((0, 0), dtype=np.float64),
                frequency="unknown",
                seasonality=1,
                skipped=True,
                skip_reason=f"Unknown dataset name: {dataset_name}",
            )
            continue
        bundles[dataset_name] = _load_one_dataset(spec, cfg)
    return bundles


def _load_one_dataset(spec: DatasetSpec, cfg: ForecastBenchmarkConfig) -> DatasetBundle:
    cache_path = Path(cfg.datasets.cache_dir) / spec.cache_file
    max_series = int(cfg.datasets.max_series_per_dataset)

    try:
        if cache_path.exists():
            series = _read_cached_series(cache_path)
        elif cfg.datasets.allow_download and spec.source_url:
            series = _download_and_cache_series(spec, cache_path)
        else:
            note = spec.notes or "Dataset is unavailable and download is disabled."
            return DatasetBundle(
                name=spec.name,
                series=np.zeros((0, 0), dtype=np.float64),
                frequency=spec.frequency,
                seasonality=int(spec.seasonality),
                skipped=True,
                skip_reason=note,
            )
    except Exception as exc:  # pragma: no cover - exercised via tests with monkeypatch.
        return DatasetBundle(
            name=spec.name,
            series=np.zeros((0, 0), dtype=np.float64),
            frequency=spec.frequency,
            seasonality=int(spec.seasonality),
            skipped=True,
            skip_reason=f"Failed to load dataset {spec.name}: {exc}",
        )

    if series.shape[0] > max_series:
        series = _deterministic_subsample(series, max_series=max_series, seed=cfg.seed)

    return DatasetBundle(
        name=spec.name,
        series=series,
        frequency=spec.frequency,
        seasonality=int(spec.seasonality),
        skipped=False,
        skip_reason=None,
    )


def _read_cached_series(path: Path) -> np.ndarray:
    arr = np.load(path, allow_pickle=False)["series"]
    series = np.asarray(arr, dtype=np.float64)
    if series.ndim == 1:
        series = series[None, :]
    if series.ndim != 2:
        raise ValueError(f"Expected cached series shape (S,T), got {series.shape}")
    if series.shape[1] < 2:
        raise ValueError("Cached dataset must have at least 2 time steps.")
    return series


def _download_and_cache_series(spec: DatasetSpec, cache_path: Path) -> np.ndarray:
    if spec.source_url is None:
        raise ValueError(f"Cannot download {spec.name}: source_url is not set.")
    if spec.name == "ettm1":
        series = _download_ettm1(spec.source_url)
    elif spec.name == "exchange_rate":
        series = _download_exchange_rate(spec.source_url)
    else:
        raise ValueError(
            f"No automated downloader implemented for {spec.name}. "
            "Provide cached npz manually."
        )

    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, series=series)
    return series


def _download_ettm1(url: str) -> np.ndarray:
    response = requests.get(url, timeout=30)
    response.raise_for_status()

    # ETTm1 has one datetime column plus numeric sensor columns.
    df = pd.read_csv(StringIO(response.text))
    value_columns = [
        col for col in df.columns if col.lower() not in {"date", "datetime"}
    ]
    if not value_columns:
        raise ValueError("ETTm1 download did not contain numeric value columns.")
    values = df[value_columns].to_numpy(dtype=np.float64)

    # Convert multivariate table (T,p) into univariate panel (p,T).
    series = values.T
    return _ensure_finite_panel(series)


def _download_exchange_rate(url: str) -> np.ndarray:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    if url.endswith(".gz"):
        text = gzip.decompress(response.content).decode("utf-8")
    else:
        text = response.text
    values = np.loadtxt(StringIO(text), delimiter=",")
    if values.ndim == 1:
        values = values[:, None]

    # Convert (T,p) into univariate panel (p,T).
    series = np.asarray(values, dtype=np.float64).T
    return _ensure_finite_panel(series)


def _ensure_finite_panel(series: np.ndarray) -> np.ndarray:
    if series.ndim != 2:
        raise ValueError(f"Expected panel shape (S,T), got {series.shape}")
    if series.shape[1] < 2:
        raise ValueError("Dataset must have at least 2 time steps.")
    # Keep NaN values: missingness/featurization pipeline handles them explicitly.
    finite_or_nan = np.isfinite(series) | np.isnan(series)
    if not finite_or_nan.all():
        raise ValueError("Dataset contains non-finite values other than NaN.")
    return series


def _deterministic_subsample(
    series: np.ndarray,
    *,
    max_series: int,
    seed: int,
) -> np.ndarray:
    if max_series >= series.shape[0]:
        return series
    rng = np.random.default_rng(seed)
    picked = np.sort(rng.choice(series.shape[0], size=max_series, replace=False))
    return series[picked]
