from __future__ import annotations

import json

from tfmplayground.benchmarks.forecasting.report import (
    _sanitize_for_json,
    build_markdown_report,
    write_json,
)


def test_write_json_handles_nan_values(tmp_path):
    path = tmp_path / "out.json"
    payload = {
        "mean_improvement": float("nan"),
        "ci_low": float("-inf"),
        "valid": 1.5,
        "nested": {"also_nan": float("nan"), "ok": 42},
        "list_with_nan": [1.0, float("nan"), 3.0],
    }
    write_json(path, payload)
    text = path.read_text(encoding="utf-8")
    data = json.loads(text)
    assert data["mean_improvement"] is None
    assert data["ci_low"] is None
    assert data["valid"] == 1.5
    assert data["nested"]["also_nan"] is None
    assert data["nested"]["ok"] == 42
    assert data["list_with_nan"] == [1.0, None, 3.0]


def test_sanitize_for_json_preserves_normal_values():
    assert _sanitize_for_json(3.14) == 3.14
    assert _sanitize_for_json("hello") == "hello"
    assert _sanitize_for_json(None) is None
    assert _sanitize_for_json(42) == 42


def test_build_markdown_report_both_tracks():
    regression_summary = {
        "claim": {
            "primary_claim_pass": True,
            "required_metric_passes": 2,
            "achieved_metric_passes": 3,
        },
        "comparisons": [
            {
                "baseline": "nanotabpfn_standard",
                "metric": "mase",
                "mean_improvement": 0.05,
                "ci_low": 0.01,
                "ci_high": 0.09,
                "win_rate": 0.6,
                "pass": True,
            }
        ],
    }
    proxy_summary = {
        "models": [
            {
                "model": "nanotabpfn_classifier",
                "balanced_accuracy": 0.75,
                "macro_auroc": 0.8,
                "n_rows": 10,
            }
        ]
    }
    report = build_markdown_report(
        regression_summary=regression_summary,
        proxy_summary=proxy_summary,
    )
    assert "# DynSCM Forecast Benchmark Report" in report
    assert "Regression Track" in report
    assert "Proxy Classification Track" in report
    assert "nanotabpfn_standard" in report
    assert "nanotabpfn_classifier" in report


def test_build_markdown_report_no_outputs():
    report = build_markdown_report(regression_summary=None, proxy_summary=None)
    assert "No benchmark outputs were generated." in report


def test_build_markdown_report_nan_in_comparisons():
    regression_summary = {
        "claim": {
            "primary_claim_pass": False,
            "required_metric_passes": 2,
            "achieved_metric_passes": 0,
        },
        "comparisons": [
            {
                "baseline": "nanotabpfn_standard",
                "metric": "mase",
                "mean_improvement": float("nan"),
                "ci_low": float("nan"),
                "ci_high": float("nan"),
                "win_rate": float("nan"),
                "pass": False,
            }
        ],
    }
    report = build_markdown_report(
        regression_summary=regression_summary, proxy_summary=None
    )
    assert "nan" in report
    assert "nanotabpfn_standard" in report
