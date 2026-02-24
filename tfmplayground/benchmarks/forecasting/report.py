"""Report rendering for forecasting benchmark results."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

__all__ = ["build_markdown_report", "write_json"]


def _sanitize_for_json(obj: Any) -> Any:
    """Replace NaN/Inf floats with None for JSON compliance."""
    if isinstance(obj, float) and not math.isfinite(obj):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_sanitize_for_json(payload), indent=2, sort_keys=True),
        encoding="utf-8",
    )


def build_markdown_report(
    *,
    regression_summary: dict[str, Any] | None,
    proxy_summary: dict[str, Any] | None,
) -> str:
    lines: list[str] = ["# DynSCM Forecast Benchmark Report", ""]

    if regression_summary is not None:
        lines.extend(_render_regression_summary(regression_summary))
        lines.append("")

    if proxy_summary is not None:
        lines.extend(_render_proxy_summary(proxy_summary))
        lines.append("")

    if regression_summary is None and proxy_summary is None:
        lines.append("No benchmark outputs were generated.")

    return "\n".join(lines).rstrip() + "\n"


def _render_regression_summary(summary: dict[str, Any]) -> list[str]:
    lines = ["## Regression Track", ""]

    claim = summary.get("claim", {})
    lines.append(
        f"- Primary claim pass: `{bool(claim.get('primary_claim_pass', False))}`"
    )
    lines.append(
        "- Required passing metrics vs standard baseline: "
        f"`{claim.get('required_metric_passes', 0)}`"
    )
    lines.append(
        "- Achieved passing metrics vs standard baseline: "
        f"`{claim.get('achieved_metric_passes', 0)}`"
    )
    nicl_status = summary.get("nicl_regression", {})
    if nicl_status:
        lines.append(
            "- NICL regression rows: "
            f"`ok={nicl_status.get('ok_rows', 0)}`, "
            f"`skipped={nicl_status.get('skipped_rows', 0)}`"
        )
    nicl_caps = summary.get("nicl_capabilities", {})
    if nicl_caps:
        lines.append(
            "- NICL mode/endpoint: "
            f"`{nicl_caps.get('regression_mode', 'off')}` / "
            f"`{nicl_caps.get('regression_endpoint')}`"
        )
    lines.append("")

    comparisons = summary.get("comparisons", [])
    if not comparisons:
        lines.append("No valid regression comparison rows were available.")
        return lines

    lines.append(
        "| Baseline | Metric | Mean Improvement | CI Low | CI High | Win Rate | Pass |"
    )
    lines.append("|---|---:|---:|---:|---:|---:|---:|")
    for row in comparisons:
        lines.append(
            "| {baseline} | {metric} | {mean:.4f} | {ci_low:.4f} | {ci_high:.4f} | "
            "{win_rate:.4f} | {passed} |".format(
                baseline=row.get("baseline", "-"),
                metric=row.get("metric", "-"),
                mean=float(row.get("mean_improvement", float("nan"))),
                ci_low=float(row.get("ci_low", float("nan"))),
                ci_high=float(row.get("ci_high", float("nan"))),
                win_rate=float(row.get("win_rate", float("nan"))),
                passed=bool(row.get("pass", False)),
            )
        )
    return lines


def _render_proxy_summary(summary: dict[str, Any]) -> list[str]:
    lines = ["## Proxy Classification Track", ""]

    rows = summary.get("models", [])
    if not rows:
        lines.append("No valid proxy comparison rows were available.")
        return lines

    lines.append("| Model | Mean Balanced Accuracy | Mean Macro AUROC | Num Rows |")
    lines.append("|---|---:|---:|---:|")
    for row in rows:
        lines.append(
            "| {model} | {bal:.4f} | {auc:.4f} | {count} |".format(
                model=row.get("model", "-"),
                bal=float(row.get("balanced_accuracy", float("nan"))),
                auc=float(row.get("macro_auroc", float("nan"))),
                count=int(row.get("n_rows", 0)),
            )
        )
    return lines
