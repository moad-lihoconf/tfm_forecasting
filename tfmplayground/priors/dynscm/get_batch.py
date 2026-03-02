"""Batch assembly for DynSCM forecasting prior samples."""

from __future__ import annotations

import contextlib
import multiprocessing as mp
from collections.abc import Callable, Mapping
from concurrent.futures import ProcessPoolExecutor

import numpy as np
import torch

from .config import DynSCMConfig
from .parallel import (
    DynSCMWorkerTask,
    build_single_dynscm_sample,
    compute_feasible_row_pairs,
    draw_seed_bundles,
    fit_feature_budget,
    generate_dynscm_worker_sample,
    init_dynscm_worker,
    pad_rows_2d,
    pad_rows_3d,
    prioritize_feature_blocks,
    required_min_series_length,
    sample_dataset_dimensions,
    slice_or_empty,
)
from .research import DynSCMSampleFilterConfig


class DynSCMBatchGenerator:
    """Stateful callable that yields deterministic DynSCM prior batches."""

    def __init__(
        self,
        cfg: DynSCMConfig,
        *,
        device: torch.device,
        seed: int | None = None,
        workers: int = 1,
        worker_blas_threads: int = 1,
        cfg_override_sampler: (
            Callable[[np.random.Generator, int], dict[str, object] | None] | None
        ) = None,
        sample_filter: DynSCMSampleFilterConfig | None = None,
        max_sample_attempts_per_item: int = 1,
        share_system_within_batch: bool = False,
    ) -> None:
        self.cfg = cfg
        self.target_device = torch.device(device)
        self.state_rng = cfg.make_rng(seed)
        if workers < 1:
            raise ValueError("workers must be >= 1.")
        if worker_blas_threads < 1:
            raise ValueError("worker_blas_threads must be >= 1.")
        if max_sample_attempts_per_item < 1:
            raise ValueError("max_sample_attempts_per_item must be >= 1.")
        self.workers = int(workers)
        self.worker_blas_threads = int(worker_blas_threads)
        self.cfg_override_sampler = cfg_override_sampler
        self.sample_filter = sample_filter
        self.max_sample_attempts_per_item = int(max_sample_attempts_per_item)
        self.share_system_within_batch = bool(share_system_within_batch)
        self._executor: ProcessPoolExecutor | None = None
        self._row_pair_cache: dict[int, np.ndarray] = {}
        self._batch_index = 0

    def __call__(
        self,
        batch_size: int,
        num_datapoints_max: int,
        num_features: int,
    ) -> dict[str, torch.Tensor | int]:
        if batch_size < 1:
            raise ValueError("batch_size must be >= 1.")
        if num_datapoints_max < 2:
            raise ValueError("num_datapoints_max must be >= 2.")
        if num_features < 1:
            raise ValueError("num_features must be >= 1.")

        n_train, n_test = _sample_shared_row_counts(
            cfg=self.cfg,
            num_datapoints_max=num_datapoints_max,
            rng=self.state_rng,
            cache=self._row_pair_cache,
        )
        cfg_overrides = None
        if self.cfg_override_sampler is not None:
            cfg_overrides = self.cfg_override_sampler(self.state_rng, self._batch_index)
        self._batch_index += 1
        shared_system_seed: int | None = None
        if self.share_system_within_batch:
            shared_system_seed = int(
                draw_seed_bundles(
                    self.state_rng,
                    batch_size=1,
                    bundle_width=1,
                )[0, 0]
            )

        sample_seeds = draw_seed_bundles(
            self.state_rng,
            batch_size=batch_size,
            bundle_width=1,
        )[:, 0]

        x_batch = np.empty(
            (batch_size, num_datapoints_max, num_features),
            dtype=np.float32,
        )
        y_batch = np.empty((batch_size, num_datapoints_max), dtype=np.float32)
        sample_metadata: dict[str, list[int | float]] = {}

        if self.workers == 1:
            for idx, sample_seed in enumerate(sample_seeds):
                x_i, y_i, metadata_i = build_single_dynscm_sample(
                    self.cfg,
                    sample_seed=int(sample_seed),
                    n_train=n_train,
                    n_test=n_test,
                    row_budget=num_datapoints_max,
                    num_features=num_features,
                    cfg_overrides=cfg_overrides,
                    sample_filter=self.sample_filter,
                    max_generation_attempts=self.max_sample_attempts_per_item,
                    shared_system_seed=shared_system_seed,
                )
                x_batch[idx] = x_i
                y_batch[idx] = y_i
                _accumulate_numeric_metadata(sample_metadata, metadata_i)
        else:
            tasks = [
                DynSCMWorkerTask(
                    sample_seed=int(sample_seed),
                    n_train=n_train,
                    n_test=n_test,
                    row_budget=num_datapoints_max,
                    num_features=num_features,
                    cfg_overrides=(
                        None if cfg_overrides is None else dict(cfg_overrides)
                    ),
                    filter_payload=(
                        None
                        if self.sample_filter is None
                        else self.sample_filter.to_payload()
                    ),
                    max_generation_attempts=self.max_sample_attempts_per_item,
                    shared_system_seed=shared_system_seed,
                )
                for sample_seed in sample_seeds
            ]
            try:
                for idx, (x_i, y_i, metadata_i) in enumerate(
                    self._get_executor().map(
                        generate_dynscm_worker_sample,
                        tasks,
                        chunksize=1,
                    )
                ):
                    x_batch[idx] = x_i
                    y_batch[idx] = y_i
                    _accumulate_numeric_metadata(sample_metadata, metadata_i)
            except Exception as exc:
                raise RuntimeError("DynSCM parallel batch generation failed.") from exc

        if not np.isfinite(x_batch).all():
            raise RuntimeError(
                "DynSCM batch feature tensor contains non-finite values."
            )
        if not np.isfinite(y_batch).all():
            raise RuntimeError("DynSCM batch label tensor contains non-finite values.")

        x_tensor = torch.from_numpy(x_batch).to(self.target_device)
        y_tensor = torch.from_numpy(y_batch).to(self.target_device)
        row_positions = torch.arange(num_datapoints_max, device=self.target_device)
        target_mask = (row_positions >= int(n_train)) & (
            row_positions < int(n_train + n_test)
        )
        target_mask = target_mask.unsqueeze(0).expand(batch_size, -1).clone()

        batch_payload: dict[str, torch.Tensor | int] = {
            "x": x_tensor,
            "y": y_tensor,
            "target_y": y_tensor.clone(),
            "single_eval_pos": int(n_train),
            "num_datapoints": int(n_train + n_test),
            "target_mask": target_mask,
        }
        for key, values in sample_metadata.items():
            if len(values) != batch_size:
                raise RuntimeError(
                    f"Metadata field {key!r} length mismatch: "
                    f"{len(values)} != batch_size={batch_size}"
                )
            values_arr = np.asarray(values)
            if np.issubdtype(values_arr.dtype, np.integer):
                batch_payload[key] = torch.from_numpy(values_arr.astype(np.int64)).to(
                    self.target_device
                )
            elif np.issubdtype(values_arr.dtype, np.floating):
                batch_payload[key] = torch.from_numpy(values_arr.astype(np.float32)).to(
                    self.target_device
                )
        return batch_payload

    def _get_executor(self) -> ProcessPoolExecutor:
        if self._executor is not None:
            return self._executor

        self._executor = ProcessPoolExecutor(
            max_workers=self.workers,
            mp_context=mp.get_context("spawn"),
            initializer=init_dynscm_worker,
            initargs=(self.cfg.to_dict(), self.worker_blas_threads),
        )
        return self._executor

    def close(self) -> None:
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


def make_get_batch_dynscm(
    cfg: DynSCMConfig,
    device: torch.device,
    seed: int | None = None,
    *,
    workers: int = 1,
    worker_blas_threads: int = 1,
    cfg_override_sampler: (
        Callable[[np.random.Generator, int], dict[str, object] | None] | None
    ) = None,
    sample_filter: DynSCMSampleFilterConfig | None = None,
    max_sample_attempts_per_item: int = 1,
    share_system_within_batch: bool = False,
) -> Callable[[int, int, int], dict[str, torch.Tensor | int]]:
    """Return stateful DynSCM batch generator for `PriorDataLoader`."""
    return DynSCMBatchGenerator(
        cfg,
        device=device,
        seed=seed,
        workers=workers,
        worker_blas_threads=worker_blas_threads,
        cfg_override_sampler=cfg_override_sampler,
        sample_filter=sample_filter,
        max_sample_attempts_per_item=max_sample_attempts_per_item,
        share_system_within_batch=share_system_within_batch,
    )


def _sample_shared_row_counts(
    *,
    cfg: DynSCMConfig,
    num_datapoints_max: int,
    rng: np.random.Generator,
    cache: dict[int, np.ndarray] | None = None,
) -> tuple[int, int]:
    if cache is None:
        feasible_pairs = compute_feasible_row_pairs(
            cfg=cfg,
            num_datapoints_max=num_datapoints_max,
        )
    else:
        if num_datapoints_max not in cache:
            cache[num_datapoints_max] = compute_feasible_row_pairs(
                cfg=cfg,
                num_datapoints_max=num_datapoints_max,
            )
        feasible_pairs = cache[num_datapoints_max]

    pair_idx = int(rng.integers(0, feasible_pairs.shape[0]))
    return int(feasible_pairs[pair_idx, 0]), int(feasible_pairs[pair_idx, 1])


def _required_min_series_length(cfg: DynSCMConfig) -> int:
    return required_min_series_length(cfg)


def _sample_dataset_dimensions(
    cfg: DynSCMConfig,
    rng: np.random.Generator,
) -> tuple[int, int, int]:
    return sample_dataset_dimensions(cfg, rng)


def _slice_or_empty(
    x: np.ndarray,
    feature_slices: Mapping[str, tuple[int, int]],
    key: str,
) -> np.ndarray:
    return slice_or_empty(x, feature_slices, key)


def _prioritize_feature_blocks(
    x: np.ndarray,
    *,
    feature_slices: Mapping[str, tuple[int, int]],
) -> np.ndarray:
    return prioritize_feature_blocks(x, feature_slices=feature_slices)


def _fit_feature_budget(x: np.ndarray, *, num_features: int) -> np.ndarray:
    return fit_feature_budget(x, num_features=num_features)


def _pad_rows_3d(x: np.ndarray, *, row_budget: int) -> np.ndarray:
    return pad_rows_3d(x, row_budget=row_budget)


def _pad_rows_2d(y: np.ndarray, *, row_budget: int) -> np.ndarray:
    return pad_rows_2d(y, row_budget=row_budget)


def _draw_seed(rng: np.random.Generator) -> int:
    return int(draw_seed_bundles(rng, batch_size=1, bundle_width=1)[0, 0])


def _accumulate_numeric_metadata(
    accum: dict[str, list[int | float]],
    metadata: Mapping[str, int | float | str],
) -> None:
    for key, value in metadata.items():
        if isinstance(value, bool):
            accum.setdefault(key, []).append(int(value))
            continue
        if isinstance(value, int | np.integer):
            accum.setdefault(key, []).append(int(value))
            continue
        if isinstance(value, float | np.floating):
            accum.setdefault(key, []).append(float(value))
