"""Configuration objects for DynSCM prior generation."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any, Literal

import numpy as np
from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)


class _FrozenConfigModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class DynSCMShapeConfig(_FrozenConfigModel):
    """Shape knobs for sampled tasks."""

    num_variables_min: int = Field(default=4, ge=2)
    num_variables_max: int = Field(default=24)
    series_length_min: int = Field(default=96, ge=8)
    series_length_max: int = Field(default=384)
    max_lag: int = Field(default=16, ge=1)
    train_rows_min: int = Field(default=16, ge=1)
    train_rows_max: int = Field(default=48)
    test_rows_min: int = Field(default=8, ge=1)
    test_rows_max: int = Field(default=24)
    forecast_horizons: tuple[int, ...] = (1, 2, 3, 5, 10)

    @field_validator("forecast_horizons")
    @classmethod
    def _check_forecast_horizons(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if not v:
            raise ValueError("forecast_horizons must be non-empty.")
        if min(v) <= 0:
            raise ValueError("forecast_horizons must contain positive integers.")
        return v

    @model_validator(mode="after")
    def _cross_validate(self) -> DynSCMShapeConfig:
        errors = []

        if self.num_variables_max < self.num_variables_min:
            errors.append("num_variables_max must be >= num_variables_min.")
        if self.series_length_max < self.series_length_min:
            errors.append("series_length_max must be >= series_length_min.")
        if self.train_rows_max < self.train_rows_min:
            errors.append("train_rows_max must be >= train_rows_min.")
        if self.test_rows_max < self.test_rows_min:
            errors.append("test_rows_max must be >= test_rows_min.")

        max_horizon = max(self.forecast_horizons)
        if (
            self.train_rows_max + self.test_rows_max
            > self.series_length_min - max_horizon - 1
        ):
            errors.append(
                "train_rows_max + test_rows_max exceeds feasible minimum "
                "timeline budget for series_length_min."
            )

        if errors:
            raise ValueError("\n".join(errors))
        return self


class DynSCMRegimeConfig(_FrozenConfigModel):
    """Regime and drift controls."""

    num_regimes: int = Field(default=3, ge=1)
    sticky_rho: float = Field(default=0.96, ge=0.0, le=1.0)
    shared_order: bool = True
    share_base_graph: bool = True
    drift_std: float = Field(default=0.01, ge=0.0)


class DynSCMGraphConfig(_FrozenConfigModel):
    """Graph sparsity and temporal dynamics knobs."""

    use_contemporaneous: bool = True
    dmax_0: int = Field(default=3, ge=0)
    dmax_lag: int = Field(default=2, ge=0)
    lambda_0: float = Field(default=1.2, ge=0.0)
    lambda_lag: float = Field(default=1.5, ge=0.0)
    base_lag_prob: float = Field(default=0.25, ge=0.0, le=1.0)
    lag_decay_gamma: float = Field(default=0.35, ge=0.0)
    q_add_0: float = Field(default=0.02, ge=0.0, le=1.0)
    q_del_0: float = Field(default=0.02, ge=0.0, le=1.0)
    q_add_lag: float = Field(default=0.03, ge=0.0, le=1.0)
    q_del_lag: float = Field(default=0.03, ge=0.0, le=1.0)


class DynSCMMechanismConfig(_FrozenConfigModel):
    """Mechanism sampling knobs."""

    mechanism_type: Literal["linear_var", "linear_plus_residual"] = (
        "linear_plus_residual"
    )
    residual_num_features: int = Field(default=12, ge=0)
    residual_lipschitz_max: float = Field(default=0.10, ge=0.0)


class DynSCMStabilityConfig(_FrozenConfigModel):
    """Stability guard configuration."""

    stability_mode: Literal["guardA"] = "guardA"
    col_budget_min: float = Field(default=0.35, gt=0.0, lt=1.0)
    col_budget_max: float = Field(default=0.85, gt=0.0, lt=1.0)
    contemp_budget_max: float = Field(default=0.25, ge=0.0, lt=1.0)
    enable_guardc_rescale: bool = False
    guardc_delta: float = Field(default=0.95, gt=0.0, lt=1.0)

    @model_validator(mode="after")
    def _cross_validate(self) -> DynSCMStabilityConfig:
        if self.col_budget_min > self.col_budget_max:
            raise ValueError("col_budget_min must be <= col_budget_max.")
        return self


class DynSCMNoiseConfig(_FrozenConfigModel):
    """Noise generation knobs."""

    noise_family: Literal["normal", "student_t"] = "student_t"
    student_df: float = Field(default=5.0)
    noise_scale_min: float = Field(default=0.02, gt=0.0)
    noise_scale_max: float = Field(default=0.20)

    @model_validator(mode="after")
    def _cross_validate(self) -> DynSCMNoiseConfig:
        errors = []
        if self.noise_scale_max < self.noise_scale_min:
            errors.append("noise_scale_max must be >= noise_scale_min.")
        if self.noise_family == "student_t" and self.student_df <= 2.0:
            errors.append("student_df must be > 2 for finite variance.")
        if errors:
            raise ValueError("\n".join(errors))
        return self


class DynSCMMissingnessConfig(_FrozenConfigModel):
    """Missingness and imputation knobs."""

    missing_mode: Literal["off", "mcar", "mar", "mnar_lite", "mix"] = "mix"
    missing_rate_min: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_rate_max: float = Field(default=0.25, ge=0.0, le=1.0)
    block_missing_prob: float = Field(default=0.25, ge=0.0, le=1.0)
    block_len_min: int = Field(default=3, ge=1)
    block_len_max: int = Field(default=18)
    impute_strategy: Literal["train_mean"] = "train_mean"
    add_mask_channels: bool = True

    @model_validator(mode="after")
    def _cross_validate(self) -> DynSCMMissingnessConfig:
        errors = []
        if self.missing_rate_max < self.missing_rate_min:
            errors.append("missing_rate_max must be >= missing_rate_min.")
        if self.block_len_max < self.block_len_min:
            errors.append("block_len_max must be >= block_len_min.")
        if errors:
            raise ValueError("\n".join(errors))
        return self


class DynSCMFeatureConfig(_FrozenConfigModel):
    """Feature extraction knobs."""

    l_max_feature: int = Field(default=32, ge=1)
    explicit_lags: tuple[int, ...] = (0, 1, 2, 5, 10)
    num_kernels: int = Field(default=3, ge=0)
    kernel_family: Literal["exp_decay", "power_law", "mix"] = "mix"
    add_time_feature: bool = True
    add_seasonality: bool = True
    season_period_choices: tuple[int, ...] = (7, 12, 24, 30)
    add_horizon_feature: bool = True
    add_log_horizon: bool = True

    @field_validator("explicit_lags")
    @classmethod
    def _check_explicit_lags(cls, v: tuple[int, ...]) -> tuple[int, ...]:
        if not v:
            raise ValueError("explicit_lags must be non-empty.")
        if min(v) < 0:
            raise ValueError("explicit_lags must contain non-negative integers.")
        return v

    @model_validator(mode="after")
    def _cross_validate(self) -> DynSCMFeatureConfig:
        errors = []

        if self.l_max_feature < max(self.explicit_lags):
            errors.append("l_max_feature must be >= max(explicit_lags).")
        if self.add_seasonality:
            if not self.season_period_choices:
                errors.append(
                    "season_period_choices must be non-empty when add_seasonality=True."
                )
            elif min(self.season_period_choices) <= 1:
                errors.append("season_period_choices must contain integers > 1.")
        if errors:
            raise ValueError("\n".join(errors))
        return self


class DynSCMSafetyConfig(_FrozenConfigModel):
    """Safety knobs for bounded sampling."""

    max_abs_x: float = Field(default=1e4, gt=0.0)
    max_resample_attempts: int = Field(default=8, ge=1)


_SECTION_MODELS: dict[str, type[_FrozenConfigModel]] = {
    "shape": DynSCMShapeConfig,
    "regime": DynSCMRegimeConfig,
    "graph": DynSCMGraphConfig,
    "mechanism": DynSCMMechanismConfig,
    "stability": DynSCMStabilityConfig,
    "noise": DynSCMNoiseConfig,
    "missingness": DynSCMMissingnessConfig,
    "features": DynSCMFeatureConfig,
    "safety": DynSCMSafetyConfig,
}

_FLAT_FIELD_TO_SECTION: dict[str, str] = {}
for _section_name, _section_model in _SECTION_MODELS.items():
    for _field_name in _section_model.model_fields:
        _FLAT_FIELD_TO_SECTION[_field_name] = _section_name
del _section_name, _section_model, _field_name


class DynSCMConfig(_FrozenConfigModel):
    """Aggregate DynSCM config with nested sections and flat compatibility."""

    random_seed: int | None = Field(default=0, ge=0)
    shape: DynSCMShapeConfig = Field(default_factory=DynSCMShapeConfig)
    regime: DynSCMRegimeConfig = Field(default_factory=DynSCMRegimeConfig)
    graph: DynSCMGraphConfig = Field(default_factory=DynSCMGraphConfig)
    mechanism: DynSCMMechanismConfig = Field(default_factory=DynSCMMechanismConfig)
    stability: DynSCMStabilityConfig = Field(default_factory=DynSCMStabilityConfig)
    noise: DynSCMNoiseConfig = Field(default_factory=DynSCMNoiseConfig)
    missingness: DynSCMMissingnessConfig = Field(
        default_factory=DynSCMMissingnessConfig
    )
    features: DynSCMFeatureConfig = Field(default_factory=DynSCMFeatureConfig)
    safety: DynSCMSafetyConfig = Field(default_factory=DynSCMSafetyConfig)

    @model_validator(mode="before")
    @classmethod
    def _lift_flat_fields(cls, data: Any) -> Any:
        if not isinstance(data, Mapping):
            return data

        normalized = dict(data)
        for section_name, section_model in _SECTION_MODELS.items():
            section_value = normalized.get(section_name)
            section_payload: dict[str, Any] | None = None
            if section_value is not None:
                if not isinstance(section_value, Mapping):
                    continue
                section_payload = dict(section_value)

            for field_name in section_model.model_fields:
                if field_name not in normalized:
                    continue
                if section_payload is not None and field_name in section_payload:
                    raise ValueError(
                        "Received both flat and nested values for "
                        f"'{field_name}' in section '{section_name}'."
                    )
                if section_payload is None:
                    section_payload = {}
                section_payload[field_name] = normalized.pop(field_name)
            if section_payload is not None:
                normalized[section_name] = section_payload
        return normalized

    @model_validator(mode="after")
    def _cross_validate(self) -> DynSCMConfig:
        errors = []

        if self.features.l_max_feature > self.shape.series_length_min - 2:
            errors.append("l_max_feature must be <= series_length_min - 2.")
        if (
            self.regime.share_base_graph
            and not self.regime.shared_order
            and self.graph.use_contemporaneous
        ):
            errors.append(
                "share_base_graph=True with shared_order=False is invalid when "
                "contemporaneous edges are enabled."
            )

        if errors:
            raise ValueError("\n".join(errors))
        return self

    def __getattr__(self, name: str) -> Any:
        section_name = _FLAT_FIELD_TO_SECTION.get(name)
        if section_name is not None:
            section = super().__getattribute__(section_name)
            return getattr(section, name)
        raise AttributeError(f"{type(self).__name__!s} has no attribute {name!r}")

    def validate(self) -> DynSCMConfig:
        return self

    def to_dict(self) -> dict[str, Any]:
        payload = {"random_seed": self.random_seed}
        for section_name in _SECTION_MODELS:
            section = getattr(self, section_name)
            payload.update(section.model_dump(mode="python"))
        return payload

    def with_overrides(self, **overrides: Any) -> DynSCMConfig:
        return type(self).from_dict(self.to_dict() | overrides)

    def resolved_seed(self, seed: int | None = None) -> int | None:
        return self.random_seed if seed is None else seed

    def make_rng(self, seed: int | None = None) -> np.random.Generator:
        return np.random.default_rng(self.resolved_seed(seed))

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> DynSCMConfig:
        try:
            return cls.model_validate(payload)
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc

    @classmethod
    def from_json(cls, path: str | Path) -> DynSCMConfig:
        try:
            return cls.model_validate_json(Path(path).read_bytes())
        except ValidationError as exc:
            raise ValueError(str(exc)) from exc
