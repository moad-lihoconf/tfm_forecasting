from __future__ import annotations

from pathlib import Path

import numpy as np

from tfmplayground.benchmarks.forecasting.config import ForecastBenchmarkConfig
from tfmplayground.benchmarks.forecasting.datasets import (
    DatasetSpec,
    _load_one_dataset,
    load_suite,
)


def test_load_suite_reads_cached_npz_and_applies_deterministic_subsample(
    tmp_path: Path,
):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    series = np.arange(20 * 50, dtype=np.float64).reshape(20, 50)
    np.savez_compressed(cache_dir / "m3_monthly.npz", series=series)

    cfg = ForecastBenchmarkConfig.from_dict(
        {
            "datasets": {
                "cache_dir": str(cache_dir),
                "dataset_names": ["m3_monthly"],
                "max_series_per_dataset": 8,
                "allow_download": False,
            }
        }
    )

    suite = load_suite(cfg)
    bundle = suite["m3_monthly"]

    assert not bundle.skipped
    assert bundle.series.shape == (8, 50)

    suite_two = load_suite(cfg)
    assert np.array_equal(bundle.series, suite_two["m3_monthly"].series)


def test_load_suite_marks_unknown_dataset_as_skipped(tmp_path: Path):
    cfg = ForecastBenchmarkConfig.from_dict(
        {
            "datasets": {
                "cache_dir": str(tmp_path / "cache"),
                "dataset_names": ["unknown_dataset"],
                "allow_download": False,
            }
        }
    )

    suite = load_suite(cfg)
    bundle = suite["unknown_dataset"]
    assert bundle.skipped
    assert "Unknown dataset" in (bundle.skip_reason or "")


def test_load_one_dataset_reports_manual_cache_requirement(tmp_path: Path):
    cfg = ForecastBenchmarkConfig.from_dict(
        {
            "datasets": {
                "cache_dir": str(tmp_path / "cache"),
                "allow_download": False,
            }
        }
    )

    spec = DatasetSpec(
        name="manual_only",
        frequency="monthly",
        seasonality=12,
        cache_file="manual_only.npz",
        source_url=None,
        notes="Provide cached data manually.",
    )

    bundle = _load_one_dataset(spec, cfg)
    assert bundle.skipped
    assert "Provide cached data manually" in (bundle.skip_reason or "")


def test_load_suite_reads_nan_padded_cached_panel(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir(parents=True)
    series = np.array(
        [
            [1.0, 2.0, 3.0, np.nan],
            [4.0, 5.0, np.nan, np.nan],
        ],
        dtype=np.float64,
    )
    np.savez_compressed(cache_dir / "m4_weekly.npz", series=series)
    np.savez_compressed(cache_dir / "tourism_monthly.npz", series=series)

    cfg = ForecastBenchmarkConfig.from_dict(
        {
            "datasets": {
                "cache_dir": str(cache_dir),
                "dataset_names": ["m4_weekly", "tourism_monthly"],
                "allow_download": False,
            }
        }
    )

    suite = load_suite(cfg)

    assert not suite["m4_weekly"].skipped
    assert not suite["tourism_monthly"].skipped
    assert suite["m4_weekly"].series.shape == (2, 4)
    assert np.isnan(suite["tourism_monthly"].series[0, -1])
