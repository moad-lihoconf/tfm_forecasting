#!/usr/bin/env python3
"""Prepare cached forecasting benchmark datasets for final runs."""

from __future__ import annotations

import argparse
import csv
import io
import shutil
import tempfile
import zipfile
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import requests

M4_WEEKLY_TRAIN_URL = (
    "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/"
    "Dataset/Train/Weekly-train.csv"
)
M4_WEEKLY_TEST_URL = (
    "https://raw.githubusercontent.com/Mcompetitions/M4-methods/master/"
    "Dataset/Test/Weekly-test.csv"
)
TOURISM_MONTHLY_ZIP_URL = (
    "https://zenodo.org/records/4656096/files/tourism_monthly_dataset.zip?download=1"
)


def _parse_numeric_row(values: Iterable[str]) -> np.ndarray:
    parsed: list[float] = []
    for raw in values:
        text = str(raw).strip().strip('"')
        if not text:
            continue
        parsed.append(float(text))
    return np.asarray(parsed, dtype=np.float64)


def _right_pad_nan(series_list: list[np.ndarray]) -> np.ndarray:
    if not series_list:
        raise ValueError("No series were parsed.")
    max_len = max(int(series.size) for series in series_list)
    if max_len < 2:
        raise ValueError("Expected at least 2 time steps in parsed dataset.")
    panel = np.full((len(series_list), max_len), np.nan, dtype=np.float64)
    for idx, series in enumerate(series_list):
        panel[idx, : series.size] = series
    return panel


def _download_bytes(url: str, *, timeout: float) -> bytes:
    response = requests.get(url, timeout=timeout)
    response.raise_for_status()
    return response.content


def build_m4_weekly_panel(*, timeout: float) -> np.ndarray:
    train_bytes = _download_bytes(M4_WEEKLY_TRAIN_URL, timeout=timeout)
    test_bytes = _download_bytes(M4_WEEKLY_TEST_URL, timeout=timeout)

    train_rows = list(csv.reader(io.StringIO(train_bytes.decode("utf-8"))))
    test_rows = list(csv.reader(io.StringIO(test_bytes.decode("utf-8"))))
    if len(train_rows) != len(test_rows):
        raise ValueError("M4 weekly train/test row counts do not match.")
    if not train_rows:
        raise ValueError("M4 weekly CSV download was empty.")

    series_list: list[np.ndarray] = []
    for train_row, test_row in zip(train_rows[1:], test_rows[1:], strict=True):
        if not train_row or not test_row:
            continue
        train_id = train_row[0].strip().strip('"')
        test_id = test_row[0].strip().strip('"')
        if train_id != test_id:
            raise ValueError(
                f"M4 weekly row id mismatch: train={train_id!r}, test={test_id!r}."
            )
        train_series = _parse_numeric_row(train_row[1:])
        test_series = _parse_numeric_row(test_row[1:])
        combined = np.concatenate([train_series, test_series], axis=0)
        if combined.size < 2:
            raise ValueError(f"Series {train_id!r} is too short after concatenation.")
        series_list.append(combined)

    return _right_pad_nan(series_list)


def _parse_tsf_series(tsf_text: str) -> list[np.ndarray]:
    in_data = False
    series_list: list[np.ndarray] = []
    for raw_line in tsf_text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        lowered = line.lower()
        if not in_data:
            if lowered == "@data":
                in_data = True
            continue
        parts = line.split(":")
        if len(parts) < 3:
            raise ValueError(f"Malformed TSF row: {line!r}")
        values = _parse_numeric_row(parts[-1].split(","))
        if values.size < 2:
            raise ValueError("TSF dataset contains a series shorter than 2 time steps.")
        series_list.append(values)
    if not in_data:
        raise ValueError("TSF file did not contain an @data section.")
    if not series_list:
        raise ValueError("TSF file contained no data rows.")
    return series_list


def build_tourism_monthly_panel(*, timeout: float) -> np.ndarray:
    zip_bytes = _download_bytes(TOURISM_MONTHLY_ZIP_URL, timeout=timeout)
    with zipfile.ZipFile(io.BytesIO(zip_bytes)) as zf:
        tsf_names = [name for name in zf.namelist() if name.lower().endswith(".tsf")]
        if not tsf_names:
            raise ValueError("Tourism monthly archive did not contain a .tsf file.")
        tsf_text = zf.read(tsf_names[0]).decode("utf-8", errors="replace")
    return _right_pad_nan(_parse_tsf_series(tsf_text))


def _save_panel(path: Path, panel: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, series=np.asarray(panel, dtype=np.float64))


def prepare_dataset(
    dataset_name: str,
    *,
    cache_dir: Path,
    timeout: float,
    force: bool,
) -> Path:
    output_map = {
        "m4_weekly": cache_dir / "m4_weekly.npz",
        "tourism_monthly": cache_dir / "tourism_monthly.npz",
    }
    if dataset_name not in output_map:
        raise ValueError(f"Unsupported dataset {dataset_name!r}.")

    output_path = output_map[dataset_name]
    if output_path.exists() and not force:
        return output_path

    if dataset_name == "m4_weekly":
        panel = build_m4_weekly_panel(timeout=timeout)
    else:
        panel = build_tourism_monthly_panel(timeout=timeout)

    _save_panel(output_path, panel)
    return output_path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Download and prepare cached forecasting benchmark datasets."
    )
    parser.add_argument(
        "--dataset",
        choices=["m4_weekly", "tourism_monthly", "all"],
        default="all",
        help="Dataset cache(s) to prepare.",
    )
    parser.add_argument(
        "--cache_dir",
        type=Path,
        default=Path("workdir/forecast_data"),
        help="Output directory for cached .npz files.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite existing cached files.",
    )
    parser.add_argument(
        "--keep_downloads",
        action="store_true",
        help="Keep temporary downloaded files for debugging.",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    datasets = (
        ["m4_weekly", "tourism_monthly"] if args.dataset == "all" else [args.dataset]
    )

    temp_dir_obj = None
    if args.keep_downloads:
        temp_dir = tempfile.mkdtemp(prefix="forecast_dataset_prepare_")
        print(f"download_debug_dir={temp_dir}")
    else:
        temp_dir_obj = tempfile.TemporaryDirectory(prefix="forecast_dataset_prepare_")
        temp_dir = temp_dir_obj.name

    try:
        for dataset_name in datasets:
            output_path = args.cache_dir / f"{dataset_name}.npz"
            existed = output_path.exists()
            path = prepare_dataset(
                dataset_name,
                cache_dir=args.cache_dir,
                timeout=args.timeout,
                force=args.force,
            )
            status = "reused" if existed and not args.force else "written"
            print(f"{dataset_name}: {status} -> {path}")
    except Exception as exc:
        raise SystemExit(f"Failed preparing datasets: {exc}") from exc
    finally:
        if temp_dir_obj is not None:
            temp_dir_obj.cleanup()
        elif not args.keep_downloads:
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
