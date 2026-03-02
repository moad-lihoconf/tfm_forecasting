#!/usr/bin/env python3
"""Download/cache m4_weekly and tourism_monthly for local forecast benchmarks."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from scripts.prepare_forecast_datasets import prepare_dataset

TARGET_DATASETS = ("m4_weekly", "tourism_monthly")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and cache m4_weekly + tourism_monthly locally."
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path("workdir/forecast_data"),
        help="Destination cache directory for .npz files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing cached files.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds.",
    )
    return parser


def _verify_panel(path: Path) -> tuple[int, int]:
    arr = np.load(path, allow_pickle=False)["series"]
    panel = np.asarray(arr, dtype=np.float64)
    if panel.ndim != 2:
        raise ValueError(f"{path} has invalid shape {panel.shape}; expected 2D panel.")
    if panel.shape[1] < 2:
        raise ValueError(f"{path} has too-short series length {panel.shape[1]}.")
    return int(panel.shape[0]), int(panel.shape[1])


def main() -> None:
    args = _build_parser().parse_args()
    args.cache_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in TARGET_DATASETS:
        target_path = args.cache_dir / f"{dataset_name}.npz"
        existed = target_path.exists()
        output_path = prepare_dataset(
            dataset_name,
            cache_dir=args.cache_dir,
            timeout=args.timeout,
            force=args.force,
        )
        status = "reused" if existed and not args.force else "written"
        n_series, series_len = _verify_panel(output_path)
        print(
            f"{dataset_name}: {status} -> {output_path} "
            f"(series={n_series}, max_len={series_len})"
        )

    print("Done.")


if __name__ == "__main__":
    main()
