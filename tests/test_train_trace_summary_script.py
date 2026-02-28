from __future__ import annotations

import importlib.util
from pathlib import Path

_SCRIPT_PATH = (
    Path(__file__).resolve().parents[1] / "scripts" / "summarize_train_trace.py"
)
_SPEC = importlib.util.spec_from_file_location("summarize_train_trace", _SCRIPT_PATH)
assert _SPEC is not None
assert _SPEC.loader is not None
_MODULE = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MODULE)
build_summary = _MODULE.build_summary


def test_build_summary_aggregates_trace_records() -> None:
    payload = {
        "metadata": {"optimizer_name": "adamw", "regression_loss": "mse"},
        "records": [
            {
                "skipped": False,
                "loss": 1.0,
                "valid_supervised_targets": 8,
                "filtered_low_std_functions": 1,
                "grad_norm_before_clip": 2.0,
                "grad_norm_after_clip": 1.0,
                "decoder_linear2_weight_update_norm": 0.1,
                "decoder_linear2_weight_norm": 2.5,
                "decoder_linear2_bias_norm": 0.4,
                "output_abs": {"mean": 0.5, "max": 1.5},
                "normalized_target_abs": {"mean": 0.3, "max": 0.9},
                "train_target_std": {"min": 0.01, "mean": 0.05},
            },
            {
                "skipped": False,
                "loss": 3.0,
                "valid_supervised_targets": 16,
                "filtered_low_std_functions": 0,
                "grad_norm_before_clip": 0.5,
                "grad_norm_after_clip": 0.5,
                "decoder_linear2_weight_update_norm": 0.2,
                "decoder_linear2_weight_norm": 2.0,
                "decoder_linear2_bias_norm": 0.3,
                "output_abs": {"mean": 0.7, "max": 1.9},
                "normalized_target_abs": {"mean": 0.4, "max": 1.2},
                "train_target_std": {"min": 0.02, "mean": 0.06},
            },
            {
                "skipped": True,
                "skip_reason": "invalid_loss_or_weight",
            },
        ],
    }

    summary = build_summary(payload)

    assert summary["metadata"]["optimizer_name"] == "adamw"
    assert summary["num_records"] == 3
    assert summary["num_active_records"] == 2
    assert summary["num_skipped_records"] == 1
    assert summary["filtered_low_std_functions_total"] == 1
    assert summary["loss"]["mean"] == 2.0
    assert summary["valid_supervised_targets"]["max"] == 16.0
    assert summary["num_batches_with_grad_clipping"] == 1
    assert summary["grad_clipping_fraction"] == 0.5
    assert summary["output_abs_max"]["max"] == 1.9
