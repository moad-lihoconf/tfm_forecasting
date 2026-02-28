#!/usr/bin/env python3
"""Profile duplicate rate and diversity characteristics of a DynSCM prior dump."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from tfmplayground.priors.audit import audit_prior_dump


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--priordump", type=str, required=True)
    parser.add_argument("--sample-limit", type=int, default=4096)
    parser.add_argument(
        "--near-duplicate-round-decimals",
        type=int,
        default=3,
        help="Rounding precision used for approximate duplicate hashing.",
    )
    parser.add_argument("--json-out", type=str, default=None)
    return parser


def build_diversity_report(
    priordump: str,
    *,
    sample_limit: int | None = 4096,
    near_duplicate_round_decimals: int = 3,
) -> dict[str, Any]:
    audit = audit_prior_dump(
        priordump,
        sample_limit=sample_limit,
        duplicate_round_decimals=near_duplicate_round_decimals,
    )

    findings: list[str] = []
    if float(audit["duplicate_fraction"]) > 0.01:
        findings.append("non-trivial exact duplicate rate")
    if float(audit["near_duplicate_fraction"]) > 0.05:
        findings.append("non-trivial near-duplicate rate")
    if float(audit["feature_truncation_fraction"]) > 0.0:
        findings.append("feature truncation present")
    if float(audit["feature_budget_saturation_fraction"]) > 0.0:
        findings.append("feature budget saturation present")

    return {
        "priordump": priordump,
        "sample_limit": sample_limit,
        "duplicate_hash_round_decimals": int(near_duplicate_round_decimals),
        "num_tables": int(audit["num_functions"]),
        "inspected_tables": int(audit["inspected_functions"]),
        "duplicate_fraction": float(audit["duplicate_fraction"]),
        "near_duplicate_fraction": float(audit["near_duplicate_fraction"]),
        "effective_target_entropy": float(audit["effective_target_entropy"]),
        "feature_truncation_fraction": float(audit["feature_truncation_fraction"]),
        "feature_budget_saturation_fraction": float(
            audit["feature_budget_saturation_fraction"]
        ),
        "active_feature_count": {
            "min": float(audit["active_features_min"]),
            "mean": float(audit["active_features_mean"]),
            "median": float(audit["active_features_median"]),
            "max": float(audit["active_features_max"]),
        },
        "num_vars_support": list(audit.get("num_vars_support", [])),
        "n_train_support": list(audit.get("n_train_support", [])),
        "n_test_support": list(audit.get("n_test_support", [])),
        "horizon_support": list(audit.get("horizon_support", [])),
        "feature_block_counts": dict(audit.get("feature_block_counts", {})),
        "family_distributions": dict(audit.get("family_distributions", {})),
        "findings": findings,
        "status": "warning" if findings else "ok",
        "audit": audit,
    }


def main() -> None:
    args = _build_parser().parse_args()
    payload = build_diversity_report(
        args.priordump,
        sample_limit=args.sample_limit,
        near_duplicate_round_decimals=args.near_duplicate_round_decimals,
    )
    print(json.dumps(payload, indent=2, sort_keys=True))
    if args.json_out is not None:
        out_path = Path(args.json_out)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )


if __name__ == "__main__":
    main()
