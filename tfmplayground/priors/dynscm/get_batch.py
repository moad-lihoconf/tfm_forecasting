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
    ) -> None:
        self.cfg = cfg
        self.target_device = torch.device(device)
        self.state_rng = cfg.make_rng(seed)
        if workers < 1:
            raise ValueError("workers must be >= 1.")
        if worker_blas_threads < 1:
            raise ValueError("worker_blas_threads must be >= 1.")
        self.workers = int(workers)
        self.worker_blas_threads = int(worker_blas_threads)
        self._executor: ProcessPoolExecutor | None = None
        self._row_pair_cache: dict[int, np.ndarray] = {}

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

        if self.workers == 1:
            for idx, sample_seed in enumerate(sample_seeds):
                x_i, y_i = build_single_dynscm_sample(
                    self.cfg,
                    sample_seed=int(sample_seed),
                    n_train=n_train,
                    n_test=n_test,
                    row_budget=num_datapoints_max,
                    num_features=num_features,
                )
                x_batch[idx] = x_i
                y_batch[idx] = y_i
        else:
            tasks = [
                DynSCMWorkerTask(
                    sample_seed=int(sample_seed),
                    n_train=n_train,
                    n_test=n_test,
                    row_budget=num_datapoints_max,
                    num_features=num_features,
                )
                for sample_seed in sample_seeds
            ]
            try:
                for idx, (x_i, y_i) in enumerate(
                    self._get_executor().map(
                        generate_dynscm_worker_sample,
                        tasks,
                        chunksize=1,
                    )
                ):
                    x_batch[idx] = x_i
                    y_batch[idx] = y_i
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

        return {
            "x": x_tensor,
            "y": y_tensor,
            "target_y": y_tensor.clone(),
            "single_eval_pos": int(n_train),
        }

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
) -> Callable[[int, int, int], dict[str, torch.Tensor | int]]:
    """Return stateful DynSCM batch generator for `PriorDataLoader`."""
    return DynSCMBatchGenerator(
        cfg,
        device=device,
        seed=seed,
        workers=workers,
        worker_blas_threads=worker_blas_threads,
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
