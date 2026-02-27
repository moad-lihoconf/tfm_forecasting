from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import h5py
import numpy as np
import pytest

from tfmplayground.priors.audit import audit_prior_dump, integrity_errors


def _write_rich_audit_fixture(
    path: Path,
    *,
    mechanism_ids: np.ndarray,
    noise_ids: np.ndarray,
    missing_ids: np.ndarray,
    kernel_ids: np.ndarray,
    pre_budget_feature_count: np.ndarray,
) -> None:
    n_rows = int(mechanism_ids.shape[0])
    seq_len = 8
    feature_dim = 4
    x = np.zeros((n_rows, seq_len, feature_dim), dtype=np.float32)
    x[:, :6, :] = 1.0
    y = np.random.default_rng(0).normal(size=(n_rows, seq_len)).astype(np.float32)
    num_datapoints = np.full((n_rows,), 6, dtype=np.int32)
    single_eval_pos = np.full((n_rows,), 2, dtype=np.int32)
    num_features = np.full((n_rows,), feature_dim, dtype=np.int32)

    metadata_payload = {
        "dynscm_family_id_mappings": {
            "mechanism_type": {"0": "linear_var", "1": "linear_plus_residual"},
            "noise_family": {"0": "normal", "1": "student_t"},
            "missing_mode": {
                "0": "off",
                "1": "mcar",
                "2": "mar",
                "3": "mnar_lite",
                "4": "mix",
            },
            "kernel_family": {"0": "exp_decay", "1": "power_law", "2": "mix"},
        }
    }

    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=x)
        f.create_dataset("y", data=y)
        f.create_dataset("num_features", data=num_features)
        f.create_dataset("num_datapoints", data=num_datapoints)
        f.create_dataset("single_eval_pos", data=single_eval_pos)
        f.create_dataset("problem_type", data="regression", dtype=h5py.string_dtype())
        f.create_dataset(
            "sampled_mechanism_type_id", data=mechanism_ids.astype(np.int32)
        )
        f.create_dataset("sampled_noise_family_id", data=noise_ids.astype(np.int32))
        f.create_dataset("sampled_missing_mode_id", data=missing_ids.astype(np.int32))
        f.create_dataset("sampled_kernel_family_id", data=kernel_ids.astype(np.int32))
        f.create_dataset(
            "sampled_pre_budget_feature_count",
            data=pre_budget_feature_count.astype(np.int32),
        )
        f.create_dataset(
            "dump_metadata_json",
            data=json.dumps(metadata_payload),
            dtype=h5py.string_dtype(),
        )


def test_audit_reports_family_distributions_and_truncation(tmp_path: Path) -> None:
    n_rows = 1200
    mechanism_ids = np.tile(np.array([0, 1], dtype=np.int32), n_rows // 2)
    noise_ids = np.tile(np.array([0, 1], dtype=np.int32), n_rows // 2)
    missing_ids = np.tile(np.array([0, 1, 2, 3, 4], dtype=np.int32), n_rows // 5)
    kernel_ids = np.tile(np.array([0, 1, 2], dtype=np.int32), n_rows // 3)
    pre_budget = np.concatenate(
        [
            np.full((840,), 4, dtype=np.int32),
            np.full((360,), 7, dtype=np.int32),
        ]
    )
    fixture_path = tmp_path / "audit_rich.h5"
    _write_rich_audit_fixture(
        fixture_path,
        mechanism_ids=mechanism_ids,
        noise_ids=noise_ids,
        missing_ids=missing_ids,
        kernel_ids=kernel_ids,
        pre_budget_feature_count=pre_budget,
    )

    report = audit_prior_dump(str(fixture_path))
    family_distributions = cast(
        dict[str, dict[str, float]],
        report["family_distributions"],
    )
    family_entropies = cast(dict[str, float], report["family_entropies"])
    assert report["has_variant_family_metadata"] is True
    assert report["has_pre_budget_feature_count_dataset"] is True
    assert cast(float, report["feature_truncation_fraction"]) == pytest.approx(0.3)
    assert family_distributions["mechanism_type"]["linear_var"] > 0.45
    assert family_distributions["mechanism_type"]["linear_plus_residual"] > 0.45
    assert family_entropies["mechanism_type"] > 0.0


def test_integrity_errors_flags_diversity_coverage_and_truncation(
    tmp_path: Path,
) -> None:
    n_rows = 1200
    mechanism_ids = np.concatenate(
        [
            np.zeros((1176,), dtype=np.int32),
            np.ones((24,), dtype=np.int32),
        ]
    )
    noise_ids = np.tile(np.array([0, 1], dtype=np.int32), n_rows // 2)
    missing_ids = np.tile(np.array([0, 1, 2, 3, 4], dtype=np.int32), n_rows // 5)
    kernel_ids = np.tile(np.array([0, 1, 2], dtype=np.int32), n_rows // 3)
    pre_budget = np.concatenate(
        [
            np.full((480,), 4, dtype=np.int32),
            np.full((720,), 8, dtype=np.int32),
        ]
    )
    fixture_path = tmp_path / "audit_gate_fail.h5"
    _write_rich_audit_fixture(
        fixture_path,
        mechanism_ids=mechanism_ids,
        noise_ids=noise_ids,
        missing_ids=missing_ids,
        kernel_ids=kernel_ids,
        pre_budget_feature_count=pre_budget,
    )
    report = audit_prior_dump(str(fixture_path))
    issues = integrity_errors(
        report,
        min_family_fraction=0.05,
        max_feature_truncation_fraction=0.40,
        min_diversity_sample_size=1000,
    )
    assert any("family coverage below threshold" in issue for issue in issues)
    assert any(
        "feature truncation fraction exceeds threshold" in issue for issue in issues
    )
