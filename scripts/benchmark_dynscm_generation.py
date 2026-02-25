#!/usr/bin/env python3
"""Reproducible benchmark for DynSCM synthetic batch generation."""

from __future__ import annotations

import argparse
import cProfile
import json
import pstats
import time
from pathlib import Path

import numpy as np
import torch

from tfmplayground.priors.dataloader import DynSCMPriorDataLoader
from tfmplayground.priors.dynscm import DynSCMConfig
from tfmplayground.priors.main import _parse_dynscm_overrides


def _load_dynscm_cfg(config_json: str | None, raw_overrides: list[str]) -> DynSCMConfig:
    cfg = DynSCMConfig.from_json(config_json) if config_json else DynSCMConfig()
    overrides = _parse_dynscm_overrides(raw_overrides)
    return cfg.with_overrides(**overrides) if overrides else cfg


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--num_batches", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--max_seq_len", type=int, default=64)
    parser.add_argument("--max_features", type=int, default=128)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--worker_blas_threads", type=int, default=1)
    parser.add_argument("--profile_top", type=int, default=0)
    parser.add_argument("--dynscm_config_json", type=str, default=None)
    parser.add_argument("--dynscm_override", action="append", default=[])
    args = parser.parse_args()

    if args.num_batches < 1:
        raise ValueError("--num_batches must be >= 1.")
    if args.batch_size < 1:
        raise ValueError("--batch_size must be >= 1.")
    if args.max_seq_len < 2:
        raise ValueError("--max_seq_len must be >= 2.")
    if args.max_features < 1:
        raise ValueError("--max_features must be >= 1.")
    if args.workers < 1:
        raise ValueError("--workers must be >= 1.")
    if args.worker_blas_threads < 1:
        raise ValueError("--worker_blas_threads must be >= 1.")

    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    cfg = _load_dynscm_cfg(args.dynscm_config_json, args.dynscm_override)

    prior = DynSCMPriorDataLoader(
        cfg=cfg,
        num_steps=args.num_batches,
        batch_size=args.batch_size,
        num_datapoints_max=args.max_seq_len,
        num_features=args.max_features,
        device=torch.device("cpu"),
        seed=args.seed,
        workers=args.workers,
        worker_blas_threads=args.worker_blas_threads,
    )

    profiler: cProfile.Profile | None = None
    if args.profile_top > 0:
        profiler = cProfile.Profile()
        profiler.enable()

    t0 = time.perf_counter()
    for _ in prior:
        pass
    elapsed = time.perf_counter() - t0
    prior.close()

    sec_per_batch = elapsed / args.num_batches
    ms_per_sample = (sec_per_batch / args.batch_size) * 1000.0
    report = {
        "num_batches": args.num_batches,
        "batch_size": args.batch_size,
        "max_seq_len": args.max_seq_len,
        "max_features": args.max_features,
        "seed": args.seed,
        "workers": args.workers,
        "worker_blas_threads": args.worker_blas_threads,
        "config_json": str(Path(args.dynscm_config_json))
        if args.dynscm_config_json
        else None,
        "dynscm_overrides": args.dynscm_override,
        "elapsed_sec": elapsed,
        "sec_per_batch": sec_per_batch,
        "ms_per_sample": ms_per_sample,
    }
    print(json.dumps(report, indent=2, sort_keys=True))

    if profiler is not None:
        profiler.disable()
        stats = pstats.Stats(profiler).sort_stats("cumtime")
        print("\nTop profile entries:")
        stats.print_stats(args.profile_top)


if __name__ == "__main__":
    main()
