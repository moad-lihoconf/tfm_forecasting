"""Configuration models for forecasting benchmark evaluation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_validator

DEFAULT_MEDIUM_SUITE: tuple[str, ...] = (
    "m3_monthly",
    "m3_quarterly",
    "m4_monthly",
    "m4_weekly",
    "tourism_monthly",
    "tourism_quarterly",
    "ettm1",
    "exchange_rate",
)


class _FrozenConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DatasetConfig(_FrozenConfigModel):
    """Dataset registry and caching configuration."""

    suite_name: Literal["medium"] = "medium"
    dataset_names: tuple[str, ...] = DEFAULT_MEDIUM_SUITE
    max_series_per_dataset: int = Field(default=128, ge=1)
    cache_dir: Path = Field(default=Path("workdir/forecast_data"))
    allow_download: bool = True

    @field_validator("dataset_names")
    @classmethod
    def _validate_dataset_names(cls, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError("dataset_names must be non-empty.")
        if len(set(value)) != len(value):
            raise ValueError("dataset_names must not contain duplicates.")
        return value


class ProtocolConfig(_FrozenConfigModel):
    """Split and featurization protocol settings."""

    horizons: tuple[int, ...] = (1, 3, 6, 12)
    context_rows: int = Field(default=32, ge=2)
    test_rows: int = Field(default=16, ge=1)
    max_feature_lag: int = Field(default=32, ge=1)
    explicit_lags: tuple[int, ...] = (0, 1, 2, 5, 10)
    num_kernels: int = Field(default=3, ge=0)
    add_mask_channels: bool = True

    @field_validator("horizons")
    @classmethod
    def _validate_horizons(cls, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("horizons must be non-empty.")
        if min(value) <= 0:
            raise ValueError("horizons must contain positive integers.")
        return value

    @property
    def required_lag(self) -> int:
        return max(int(self.max_feature_lag), int(max(self.explicit_lags)))


class ModelConfig(_FrozenConfigModel):
    """Model and checkpoint settings for baselines."""

    model_standard_ckpt: str | None = None
    model_standard_dist: str | None = None
    model_dynscm_ckpt: str | None = None
    model_dynscm_dist: str | None = None
    tabicl_checkpoint_version: str = "tabicl-regressor-v2-20260212.ckpt"
    tabicl_model_path: str | None = None
    nicl_api_url: str = "https://prediction.neuralk-ai.com/predict"
    nicl_timeout_seconds: float = Field(default=20.0, gt=0.0)
    nicl_max_retries: int = Field(default=3, ge=1)


class StatisticsConfig(_FrozenConfigModel):
    """Statistical comparison and claim thresholds."""

    bootstrap_samples: int = Field(default=2000, ge=100)
    confidence_level: float = Field(default=0.95, gt=0.5, lt=1.0)
    claim_metrics: tuple[str, ...] = ("mase", "smape", "rmse")
    min_metrics_to_pass: int = Field(default=2, ge=1)


class ProxyConfig(_FrozenConfigModel):
    """Proxy classification benchmark settings."""

    num_classes: int = Field(default=4, ge=2)
    min_samples_per_class: int = Field(default=4, ge=1)


class ForecastBenchmarkConfig(_FrozenConfigModel):
    """Top-level benchmark configuration."""

    mode: Literal["regression", "proxy", "both"] = "both"
    seed: int = Field(default=42, ge=0)
    output_dir: Path = Field(default=Path("workdir/forecast_results"))
    datasets: DatasetConfig = Field(default_factory=DatasetConfig)
    protocol: ProtocolConfig = Field(default_factory=ProtocolConfig)
    models: ModelConfig = Field(default_factory=ModelConfig)
    stats: StatisticsConfig = Field(default_factory=StatisticsConfig)
    proxy: ProxyConfig = Field(default_factory=ProxyConfig)

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> ForecastBenchmarkConfig:
        try:
            return cls.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc

    @classmethod
    def from_json(cls, path: str | Path) -> ForecastBenchmarkConfig:
        try:
            return cls.model_validate_json(Path(path).read_bytes())
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc

    def to_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="python")

    def with_overrides(self, **overrides: Any) -> ForecastBenchmarkConfig:
        payload = self.to_dict()
        payload.update(overrides)
        return type(self).from_dict(payload)
