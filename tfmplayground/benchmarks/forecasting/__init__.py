"""Forecasting benchmark package for DynSCM research validation."""

from .config import ForecastBenchmarkConfig
from .datasets import DatasetBundle, load_suite
from .runner import (
    BenchmarkArtifacts,
    evaluate_proxy,
    evaluate_regression,
    run_benchmark,
)

__all__ = [
    "BenchmarkArtifacts",
    "DatasetBundle",
    "ForecastBenchmarkConfig",
    "evaluate_proxy",
    "evaluate_regression",
    "load_suite",
    "run_benchmark",
]
