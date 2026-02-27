#!/usr/bin/env python3
"""Audit a DynSCM prior dump for padding/metadata integrity issues."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from tfmplayground.priors.audit import audit_prior_dump, integrity_errors


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--priordump",
        type=str,
        required=True,
        help="Path to the HDF5 prior dump.",
    )
    parser.add_argument(
        "--sample_limit",
        type=int,
        default=None,
        help="Optional number of tables to inspect from the start of the dump.",
    )
    parser.add_argument(
        "--max_padded_target_fraction",
        type=float,
        default=0.05,
        help="Hard-fail threshold for inferred padded target fraction.",
    )
    parser.add_argument(
        "--max_num_datapoints_mismatch_fraction",
        type=float,
        default=0.01,
        help="Hard-fail threshold for inferred num_datapoints mismatch fraction.",
    )
    parser.add_argument(
        "--min_family_fraction",
        type=float,
        default=0.05,
        help=(
            "When family metadata is available and enough rows are inspected, each "
            "family category must appear at least at this fraction."
        ),
    )
    parser.add_argument(
        "--max_feature_truncation_fraction",
        type=float,
        default=0.40,
        help=(
            "Hard-fail threshold for rows with pre-budget feature count above "
            "the configured max_features budget."
        ),
    )
    parser.add_argument(
        "--min_diversity_sample_size",
        type=int,
        default=1000,
        help=(
            "Minimum number of inspected rows required before diversity gates "
            "are enforced."
        ),
    )
    parser.add_argument(
        "--json_out",
        type=str,
        default=None,
        help="Optional file to write the full audit JSON report.",
    )
    parser.add_argument(
        "--fail_on_issues",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Exit non-zero when hard integrity issues are found.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    report = audit_prior_dump(
        args.priordump,
        sample_limit=args.sample_limit,
    )
    issues = integrity_errors(
        report,
        max_padded_target_fraction=args.max_padded_target_fraction,
        max_num_datapoints_mismatch_fraction=args.max_num_datapoints_mismatch_fraction,
        min_family_fraction=args.min_family_fraction,
        max_feature_truncation_fraction=args.max_feature_truncation_fraction,
        min_diversity_sample_size=args.min_diversity_sample_size,
    )

    print(json.dumps(report, indent=2, sort_keys=True))
    if args.json_out is not None:
        output_path = Path(args.json_out)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(
            json.dumps(report, indent=2, sort_keys=True),
            encoding="utf-8",
        )
    if issues:
        print("\nIntegrity issues:")
        for issue in issues:
            print(f"- {issue}")
        if args.fail_on_issues:
            raise SystemExit(2)
    else:
        print("\nIntegrity issues: none")


if __name__ == "__main__":
    main()
