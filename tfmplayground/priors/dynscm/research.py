"""Research helpers for live DynSCM curriculum experiments."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import torch

from .config import DynSCMConfig

_FAMILY_FIELDS = (
    "mechanism_type",
    "noise_family",
    "missing_mode",
    "kernel_family",
)
_SCHEDULE_EPS = 1e-12


def _normalize_weights(weights: Sequence[float]) -> np.ndarray:
    arr = np.asarray(weights, dtype=np.float64)
    if arr.ndim != 1 or arr.size == 0:
        raise ValueError("weights must be a non-empty 1D sequence.")
    if not np.isfinite(arr).all():
        raise ValueError("weights must contain only finite values.")
    if np.any(arr < 0.0):
        raise ValueError("weights must be non-negative.")
    total = float(arr.sum())
    if total <= 0.0:
        raise ValueError("weights must sum to a positive value.")
    return arr / total


def _sample_choice_from_cfg(
    cfg: DynSCMConfig,
    rng: np.random.Generator,
    field_name: str,
) -> str:
    default_value = str(getattr(cfg, field_name))
    choices = getattr(cfg, f"{field_name}_choices", None)
    probs = getattr(cfg, f"{field_name}_probs", None)
    if choices is None or probs is None:
        return default_value
    prob_arr = _normalize_weights([float(value) for value in probs])
    index = int(rng.choice(np.arange(len(choices), dtype=np.int64), p=prob_arr))
    return str(choices[index])


def sample_batch_shared_family_overrides(
    cfg: DynSCMConfig,
    rng: np.random.Generator,
    *,
    family_fields: Sequence[str],
) -> dict[str, object]:
    """Draw one family assignment and pin it for the full batch."""
    overrides: dict[str, object] = {}
    for field_name in family_fields:
        if field_name not in _FAMILY_FIELDS:
            raise ValueError(f"Unsupported batch-shared family field: {field_name!r}.")
        overrides[field_name] = _sample_choice_from_cfg(cfg, rng, field_name)
        overrides[f"{field_name}_choices"] = None
        overrides[f"{field_name}_probs"] = None
    return overrides


@dataclass(frozen=True, slots=True)
class DynSCMSampleFilterConfig:
    reject_clipped: bool = False
    max_abs_value_cap: float | None = None
    min_train_target_std: float | None = None
    min_probe_r2: float | None = None
    max_probe_r2: float | None = None
    min_informative_feature_count: int | None = None
    informative_feature_std_floor: float | None = None
    max_missing_fraction: float | None = None
    max_block_missing_fraction: float | None = None

    def to_payload(self) -> dict[str, object]:
        return {
            "reject_clipped": bool(self.reject_clipped),
            "max_abs_value_cap": self.max_abs_value_cap,
            "min_train_target_std": self.min_train_target_std,
            "min_probe_r2": self.min_probe_r2,
            "max_probe_r2": self.max_probe_r2,
            "min_informative_feature_count": self.min_informative_feature_count,
            "informative_feature_std_floor": self.informative_feature_std_floor,
            "max_missing_fraction": self.max_missing_fraction,
            "max_block_missing_fraction": self.max_block_missing_fraction,
        }

    @classmethod
    def from_payload(
        cls,
        payload: Mapping[str, object] | None,
    ) -> DynSCMSampleFilterConfig | None:
        if payload is None:
            return None
        max_abs_value_cap = payload.get("max_abs_value_cap")
        min_train_target_std = payload.get("min_train_target_std")
        min_probe_r2 = payload.get("min_probe_r2")
        max_probe_r2 = payload.get("max_probe_r2")
        min_informative_feature_count = payload.get("min_informative_feature_count")
        informative_feature_std_floor = payload.get("informative_feature_std_floor")
        max_missing_fraction = payload.get("max_missing_fraction")
        max_block_missing_fraction = payload.get("max_block_missing_fraction")
        return cls(
            reject_clipped=bool(payload.get("reject_clipped", False)),
            max_abs_value_cap=(
                None
                if max_abs_value_cap is None
                else float(cast(int | float, max_abs_value_cap))
            ),
            min_train_target_std=(
                None
                if min_train_target_std is None
                else float(cast(int | float, min_train_target_std))
            ),
            min_probe_r2=(
                None if min_probe_r2 is None else float(cast(int | float, min_probe_r2))
            ),
            max_probe_r2=(
                None if max_probe_r2 is None else float(cast(int | float, max_probe_r2))
            ),
            min_informative_feature_count=(
                None
                if min_informative_feature_count is None
                else int(cast(int | float, min_informative_feature_count))
            ),
            informative_feature_std_floor=(
                None
                if informative_feature_std_floor is None
                else float(cast(int | float, informative_feature_std_floor))
            ),
            max_missing_fraction=(
                None
                if max_missing_fraction is None
                else float(cast(int | float, max_missing_fraction))
            ),
            max_block_missing_fraction=(
                None
                if max_block_missing_fraction is None
                else float(cast(int | float, max_block_missing_fraction))
            ),
        )

    def accepts(self, metadata: Mapping[str, object]) -> bool:
        return self.rejection_reason(metadata) is None

    def rejection_reason(self, metadata: Mapping[str, object]) -> str | None:
        if self.reject_clipped and bool(metadata.get("sampled_simulation_clipped", 0)):
            return "clipped"
        max_abs_value = metadata.get("sampled_simulation_max_abs_value")
        if (
            self.max_abs_value_cap is not None
            and isinstance(max_abs_value, int | float)
            and float(max_abs_value) > self.max_abs_value_cap
        ):
            return "max_abs_value"
        target_std = metadata.get("sampled_train_target_std")
        if (
            self.min_train_target_std is not None
            and isinstance(target_std, int | float)
            and float(target_std) < self.min_train_target_std
        ):
            return "low_std"
        probe_r2 = metadata.get("sampled_probe_r2")
        if self.min_probe_r2 is not None and (
            not isinstance(probe_r2, int | float)
            or not np.isfinite(float(probe_r2))
            or float(probe_r2) < self.min_probe_r2
        ):
            return "probe_r2_low"
        if (
            self.max_probe_r2 is not None
            and isinstance(probe_r2, int | float)
            and np.isfinite(float(probe_r2))
            and float(probe_r2) > self.max_probe_r2
        ):
            return "probe_r2_high"
        informative_feature_count = metadata.get("sampled_informative_feature_count")
        if (
            self.min_informative_feature_count is not None
            and isinstance(informative_feature_count, int | float)
            and int(informative_feature_count) < self.min_informative_feature_count
        ):
            return "informative_features_low"
        missing_fraction = metadata.get("sampled_missing_fraction")
        if (
            self.max_missing_fraction is not None
            and isinstance(missing_fraction, int | float)
            and float(missing_fraction) > self.max_missing_fraction
        ):
            return "missing_fraction_high"
        block_missing_fraction = metadata.get("sampled_block_missing_fraction")
        if (
            self.max_block_missing_fraction is not None
            and isinstance(block_missing_fraction, int | float)
            and float(block_missing_fraction) > self.max_block_missing_fraction
        ):
            return "block_missing_fraction_high"
        return None


@dataclass(frozen=True, slots=True)
class DynSCMBatchMode:
    name: str
    cfg_overrides: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _SchedulePhase:
    end_progress: float
    start_weights: tuple[float, ...]
    end_weights: tuple[float, ...]


class LinearPhaseSchedule:
    """Piecewise-linear weight schedule over normalized training progress."""

    def __init__(
        self,
        phases: Sequence[tuple[float, Sequence[float], Sequence[float] | None]],
    ) -> None:
        if not phases:
            raise ValueError("phases must be non-empty.")
        normalized_phases: list[_SchedulePhase] = []
        prev_end = 0.0
        width: int | None = None
        for end_progress, start_weights, end_weights in phases:
            end_value = float(end_progress)
            if end_value <= prev_end or end_value > 1.0 + _SCHEDULE_EPS:
                raise ValueError(
                    "phase end_progress values must increase within (0, 1]."
                )
            start_tuple = tuple(float(value) for value in start_weights)
            end_tuple = (
                start_tuple
                if end_weights is None
                else tuple(float(value) for value in end_weights)
            )
            if width is None:
                width = len(start_tuple)
            if len(start_tuple) != width or len(end_tuple) != width:
                raise ValueError("all phase weight vectors must have identical widths.")
            normalized_phases.append(
                _SchedulePhase(
                    end_progress=min(1.0, end_value),
                    start_weights=tuple(_normalize_weights(start_tuple).tolist()),
                    end_weights=tuple(_normalize_weights(end_tuple).tolist()),
                )
            )
            prev_end = end_value
        if abs(normalized_phases[-1].end_progress - 1.0) > _SCHEDULE_EPS:
            raise ValueError("the final phase must end at progress 1.0.")
        self._phases = tuple(normalized_phases)

    def weights_at(self, progress: float) -> np.ndarray:
        progress_value = min(max(float(progress), 0.0), 1.0)
        prev_end = 0.0
        for phase in self._phases:
            if progress_value <= phase.end_progress + _SCHEDULE_EPS:
                width = max(phase.end_progress - prev_end, _SCHEDULE_EPS)
                alpha = min(max((progress_value - prev_end) / width, 0.0), 1.0)
                start = np.asarray(phase.start_weights, dtype=np.float64)
                end = np.asarray(phase.end_weights, dtype=np.float64)
                return _normalize_weights(
                    (((1.0 - alpha) * start) + (alpha * end)).tolist()
                )
            prev_end = phase.end_progress
        return np.asarray(self._phases[-1].end_weights, dtype=np.float64)


class SelfPacedWeightSchedule(LinearPhaseSchedule):
    """Alias for readability in curriculum definitions."""


class CurriculumProgressGate:
    """Track whether a scheduler update improved a probe metric."""

    def __init__(self, *, min_improvement: float = 0.0) -> None:
        self.min_improvement = float(min_improvement)
        self.best_score: float | None = None

    def accept(self, score: float) -> bool:
        if self.best_score is None or score >= self.best_score + self.min_improvement:
            self.best_score = float(score)
            return True
        return False


class DifficultyBinnedSampler:
    """Sample named difficulty bins using a progress schedule."""

    def __init__(
        self,
        *,
        bin_names: Sequence[str],
        schedule: LinearPhaseSchedule,
        total_batches: int,
    ) -> None:
        if not bin_names:
            raise ValueError("bin_names must be non-empty.")
        if len(bin_names) != len(schedule.weights_at(0.0)):
            raise ValueError("bin_names and schedule widths must match.")
        if total_batches < 1:
            raise ValueError("total_batches must be >= 1.")
        self.bin_names = tuple(bin_names)
        self.schedule = schedule
        self.total_batches = int(total_batches)
        self.last_bin_name: str | None = None

    def _progress_for_batch(self, batch_index: int) -> float:
        if self.total_batches <= 1:
            return 1.0
        return min(max(float(batch_index) / float(self.total_batches - 1), 0.0), 1.0)

    def __call__(self, rng: np.random.Generator, batch_index: int) -> str:
        weights = self.schedule.weights_at(self._progress_for_batch(batch_index))
        bin_index = int(
            rng.choice(np.arange(len(self.bin_names), dtype=np.int64), p=weights)
        )
        self.last_bin_name = self.bin_names[bin_index]
        return self.last_bin_name


class NamedBatchModeSampler:
    """Select one named config mode per batch using a progress schedule."""

    def __init__(
        self,
        *,
        base_cfg: DynSCMConfig,
        modes: Sequence[DynSCMBatchMode],
        schedule: LinearPhaseSchedule,
        total_batches: int,
        family_fields: Sequence[str] = (),
    ) -> None:
        if total_batches < 1:
            raise ValueError("total_batches must be >= 1.")
        if not modes:
            raise ValueError("modes must be non-empty.")
        if len(modes) != len(schedule.weights_at(0.0)):
            raise ValueError("modes and schedule widths must match.")
        self.base_cfg = base_cfg
        self.modes = tuple(modes)
        self.schedule = schedule
        self.total_batches = int(total_batches)
        self.family_fields = tuple(family_fields)
        self.last_mode_name: str | None = None

    def _progress_for_batch(self, batch_index: int) -> float:
        if self.total_batches <= 1:
            return 1.0
        return min(max(float(batch_index) / float(self.total_batches - 1), 0.0), 1.0)

    def __call__(
        self,
        rng: np.random.Generator,
        batch_index: int,
    ) -> dict[str, object]:
        weights = self.schedule.weights_at(self._progress_for_batch(batch_index))
        mode_index = int(
            rng.choice(np.arange(len(self.modes), dtype=np.int64), p=weights)
        )
        mode = self.modes[mode_index]
        self.last_mode_name = mode.name
        overrides = dict(mode.cfg_overrides)
        if self.family_fields:
            mode_cfg = self.base_cfg.with_overrides(**overrides)
            overrides.update(
                sample_batch_shared_family_overrides(
                    mode_cfg,
                    rng,
                    family_fields=self.family_fields,
                )
            )
        return overrides


class MixtureGetBatch:
    """Choose one child get-batch function per batch using a schedule."""

    def __init__(
        self,
        *,
        children: Sequence[Callable[[int, int, int], dict[str, torch.Tensor | int]]],
        child_names: Sequence[str],
        schedule: LinearPhaseSchedule,
        seed: int | None = None,
        total_batches: int,
    ) -> None:
        if len(children) != len(child_names):
            raise ValueError("children and child_names must have identical lengths.")
        if len(children) != len(schedule.weights_at(0.0)):
            raise ValueError("children and schedule widths must match.")
        if total_batches < 1:
            raise ValueError("total_batches must be >= 1.")
        self.children = tuple(children)
        self.child_names = tuple(child_names)
        self.schedule = schedule
        self.total_batches = int(total_batches)
        self.rng = np.random.default_rng(seed)
        self.batch_index = 0
        self.selection_counts = {name: 0 for name in self.child_names}
        self.last_child_name: str | None = None

    def _progress(self) -> float:
        if self.total_batches <= 1:
            return 1.0
        return min(
            max(float(self.batch_index) / float(self.total_batches - 1), 0.0),
            1.0,
        )

    def __call__(
        self,
        batch_size: int,
        num_datapoints_max: int,
        num_features: int,
    ) -> dict[str, torch.Tensor | int]:
        weights = self.schedule.weights_at(self._progress())
        child_index = int(
            self.rng.choice(
                np.arange(len(self.children), dtype=np.int64),
                p=_normalize_weights(weights.tolist()),
            )
        )
        child = self.children[child_index]
        child_name = self.child_names[child_index]
        payload = child(batch_size, num_datapoints_max, num_features)
        self.batch_index += 1
        self.selection_counts[child_name] += 1
        self.last_child_name = child_name
        return payload

    def close(self) -> None:
        for child in self.children:
            close_fn = getattr(child, "close", None)
            if callable(close_fn):
                close_fn()
