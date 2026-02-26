from __future__ import annotations

from pathlib import Path

import h5py
import torch

import pretrain_regression


def _create_minimal_prior_dump(path: Path) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=torch.zeros((4, 6, 3), dtype=torch.float32).numpy())
        f.create_dataset("y", data=torch.zeros((4, 6), dtype=torch.float32).numpy())
        f.create_dataset("num_features", data=[3, 3, 3, 3])
        f.create_dataset("num_datapoints", data=[6, 6, 6, 6])
        f.create_dataset("single_eval_pos", data=[3, 3, 3, 3])
        f.create_dataset("problem_type", data="regression", dtype=h5py.string_dtype())


def test_pretrain_regression_main_local_paths(monkeypatch, tmp_path: Path) -> None:
    prior_path = tmp_path / "prior.h5"
    weights_path = tmp_path / "weights.pth"
    buckets_path = tmp_path / "buckets.pth"
    _create_minimal_prior_dump(prior_path)

    monkeypatch.setattr(
        pretrain_regression,
        "get_default_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        pretrain_regression,
        "make_global_bucket_edges",
        lambda filename, n_buckets, device: torch.linspace(-1.0, 1.0, n_buckets + 1),
    )

    def _fake_train(**kwargs):
        assert kwargs["run_name"] == "local-smoke"
        model = kwargs["model"]
        return model, {"best_epoch": 1, "best_metric": 0.1, "stop_reason": "completed"}

    monkeypatch.setattr(pretrain_regression, "train", _fake_train)

    pretrain_regression.main(
        [
            "--priordump",
            str(prior_path),
            "--saveweights",
            str(weights_path),
            "--savebuckets",
            str(buckets_path),
            "--epochs",
            "1",
            "--steps",
            "1",
            "--batchsize",
            "1",
            "--n_buckets",
            "8",
            "--val_split",
            "0.5",
            "--runname",
            "local-smoke",
        ]
    )

    assert weights_path.exists()
    assert buckets_path.exists()
    assert Path(f"{weights_path}.best.pth").exists()
