from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean, median
from typing import Any

from tfmplayground.gcs_utils import path_for_read


def _float_stats(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"min": None, "median": None, "mean": None, "max": None}
    return {
        "min": min(values),
        "median": median(values),
        "mean": mean(values),
        "max": max(values),
    }


def _nested_stat(
    records: list[dict[str, Any]],
    key: str,
    stat_name: str,
) -> list[float]:
    values: list[float] = []
    for record in records:
        nested = record.get(key)
        if not isinstance(nested, dict):
            continue
        value = nested.get(stat_name)
        if isinstance(value, int | float):
            values.append(float(value))
    return values


def build_summary(payload: dict[str, Any]) -> dict[str, Any]:
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError("Trace payload must contain a list under 'records'.")

    total_records = len(records)
    skipped_records = [
        record for record in records if bool(record.get("skipped", False))
    ]
    active_records = [
        record for record in records if not bool(record.get("skipped", False))
    ]

    def scalar_values(
        key: str,
        *,
        source: list[dict[str, Any]] | None = None,
    ) -> list[float]:
        rows = active_records if source is None else source
        values: list[float] = []
        for record in rows:
            value = record.get(key)
            if isinstance(value, int | float):
                values.append(float(value))
        return values

    grad_before = scalar_values("grad_norm_before_clip")
    grad_after = scalar_values("grad_norm_after_clip")
    clipped_records = 0
    for record in active_records:
        before = record.get("grad_norm_before_clip")
        after = record.get("grad_norm_after_clip")
        if not isinstance(before, int | float) or not isinstance(after, int | float):
            continue
        if float(before) > 1.0 and float(after) < float(before):
            clipped_records += 1

    summary = {
        "metadata": payload.get("metadata", {}),
        "num_records": total_records,
        "num_active_records": len(active_records),
        "num_skipped_records": len(skipped_records),
        "skipped_fraction": (
            len(skipped_records) / total_records if total_records > 0 else 0.0
        ),
        "filtered_low_std_functions_total": int(
            sum(
                int(record.get("filtered_low_std_functions", 0))
                for record in active_records
                if isinstance(record.get("filtered_low_std_functions", 0), int | float)
            )
        ),
        "loss": _float_stats(scalar_values("loss")),
        "valid_supervised_targets": _float_stats(
            scalar_values("valid_supervised_targets")
        ),
        "grad_norm_before_clip": _float_stats(grad_before),
        "grad_norm_after_clip": _float_stats(grad_after),
        "decoder_linear2_weight_update_norm": _float_stats(
            scalar_values("decoder_linear2_weight_update_norm")
        ),
        "decoder_linear2_weight_norm": _float_stats(
            scalar_values("decoder_linear2_weight_norm")
        ),
        "decoder_linear2_bias_norm": _float_stats(
            scalar_values("decoder_linear2_bias_norm")
        ),
        "output_abs_mean": _float_stats(
            _nested_stat(active_records, "output_abs", "mean")
        ),
        "output_abs_max": _float_stats(
            _nested_stat(active_records, "output_abs", "max")
        ),
        "normalized_target_abs_mean": _float_stats(
            _nested_stat(active_records, "normalized_target_abs", "mean")
        ),
        "normalized_target_abs_max": _float_stats(
            _nested_stat(active_records, "normalized_target_abs", "max")
        ),
        "train_target_std_mean": _float_stats(
            _nested_stat(active_records, "train_target_std", "mean")
        ),
        "train_target_std_min": _float_stats(
            _nested_stat(active_records, "train_target_std", "min")
        ),
        "num_batches_with_grad_clipping": clipped_records,
        "grad_clipping_fraction": (
            clipped_records / len(active_records) if active_records else 0.0
        ),
    }
    return summary


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--trace_json",
        type=str,
        required=True,
        help="local path or gs:// URI to the training trace JSON",
    )
    parser.add_argument(
        "--json_out",
        type=str,
        default=None,
        help="optional local path to write the summarized JSON",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    args = _build_parser().parse_args(argv)
    trace_path = Path(path_for_read(args.trace_json))
    with trace_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    summary = build_summary(payload)
    print(json.dumps(summary, indent=2, sort_keys=True))
    if args.json_out is not None:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(summary, indent=2, sort_keys=True),
            encoding="utf-8",
        )


if __name__ == "__main__":
    main()
