from __future__ import annotations

from pathlib import Path

import h5py
import torch

import pretrain_regression


def _create_minimal_prior_dump(path: Path) -> None:
    with h5py.File(path, "w") as f:
        x = torch.ones((4, 6, 3), dtype=torch.float32)
        y = torch.linspace(-1.0, 1.0, steps=24, dtype=torch.float32).reshape(4, 6)
        f.create_dataset("X", data=x.numpy())
        f.create_dataset("y", data=y.numpy())
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
        lambda filename, n_buckets, device, indices=None: torch.linspace(
            -1.0,
            1.0,
            n_buckets + 1,
        ),
    )
    captured_train_kwargs: dict[str, object] = {}

    def _fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
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
            "--optimizer",
            "adamw",
            "--loss_weighting",
            "per_function",
            "--target_normalization",
            "none",
            "--target_std_floor",
            "0.02",
            "--split_seed",
            "7",
            "--min_train_target_std",
            "0.05",
            "--grad_clip_norm",
            "5.0",
            "--feature_normalization",
            "none",
            "--debug_output_clamp",
            "5.0",
            "--debug_trace_first_n_batches",
            "3",
            "--debug_trace_every_n_batches",
            "4",
            "--runname",
            "local-smoke",
        ]
    )

    assert weights_path.exists()
    assert buckets_path.exists()
    assert Path(f"{weights_path}.best.pth").exists()
    assert captured_train_kwargs["use_amp"] is False
    assert captured_train_kwargs["amp_dtype"] == torch.float16
    assert captured_train_kwargs["loss_weighting"] == "per_function"
    assert captured_train_kwargs["optimizer_name"] == "adamw"
    assert captured_train_kwargs["target_normalization"] == "none"
    assert captured_train_kwargs["target_std_floor"] == 0.02
    assert captured_train_kwargs["min_train_target_std"] == 0.05
    assert captured_train_kwargs["grad_clip_norm"] == 5.0
    assert captured_train_kwargs["debug_trace_first_n_batches"] == 3
    assert captured_train_kwargs["debug_trace_every_n_batches"] == 4
    assert captured_train_kwargs["model"].feature_normalization == "none"
    assert captured_train_kwargs["model"].debug_output_clamp == 5.0
    expected_perm = torch.randperm(4, generator=torch.Generator().manual_seed(7))
    expected_val = expected_perm[:2].tolist()
    expected_train = expected_perm[2:].tolist()
    assert captured_train_kwargs["prior"].indices.tolist() == expected_train
    assert captured_train_kwargs["val_prior"].indices.tolist() == expected_val


def test_pretrain_regression_main_scalar_loss_skips_bucket_edges(
    monkeypatch, tmp_path: Path
) -> None:
    prior_path = tmp_path / "prior.h5"
    weights_path = tmp_path / "weights.pth"
    buckets_path = tmp_path / "buckets.pth"
    _create_minimal_prior_dump(prior_path)

    monkeypatch.setattr(
        pretrain_regression,
        "get_default_device",
        lambda: torch.device("cpu"),
    )

    def _fail_bucket_edges(*args, **kwargs):
        raise AssertionError("bucket edges should not be computed for mse regression")

    monkeypatch.setattr(
        pretrain_regression,
        "make_global_bucket_edges",
        _fail_bucket_edges,
    )
    captured_train_kwargs: dict[str, object] = {}

    def _fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
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
            "--val_split",
            "0.5",
            "--regression_loss",
            "mse",
            "--runname",
            "scalar-smoke",
        ]
    )

    assert captured_train_kwargs["regression_loss_name"] == "mse"
    weights_payload = torch.load(weights_path)
    buckets_payload = torch.load(buckets_path)
    assert weights_payload["architecture"]["num_outputs"] == 1
    assert weights_payload["regression_loss"] == "mse"
    assert buckets_payload == {"regression_loss": "mse", "bucket_edges": None}


def test_pretrain_regression_applies_deterministic_subset_sizes(
    monkeypatch, tmp_path: Path
) -> None:
    prior_path = tmp_path / "prior.h5"
    weights_path = tmp_path / "weights.pth"
    buckets_path = tmp_path / "buckets.pth"
    with h5py.File(prior_path, "w") as f:
        x = torch.ones((8, 6, 3), dtype=torch.float32)
        y = torch.linspace(-1.0, 1.0, steps=48, dtype=torch.float32).reshape(8, 6)
        f.create_dataset("X", data=x.numpy())
        f.create_dataset("y", data=y.numpy())
        f.create_dataset("num_features", data=[3] * 8)
        f.create_dataset("num_datapoints", data=[6] * 8)
        f.create_dataset("single_eval_pos", data=[3] * 8)
        f.create_dataset("problem_type", data="regression", dtype=h5py.string_dtype())

    monkeypatch.setattr(
        pretrain_regression,
        "get_default_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        pretrain_regression,
        "make_global_bucket_edges",
        lambda filename, n_buckets, device, indices=None: torch.linspace(
            -1.0,
            1.0,
            n_buckets + 1,
        ),
    )
    captured_train_kwargs: dict[str, object] = {}

    def _fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
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
            "2",
            "--n_buckets",
            "8",
            "--val_split",
            "0.5",
            "--split_seed",
            "11",
            "--train_subset_size",
            "3",
            "--val_subset_size",
            "2",
            "--runname",
            "subset-smoke",
        ]
    )

    expected_perm = torch.randperm(8, generator=torch.Generator().manual_seed(11))
    expected_val = expected_perm[:4].tolist()[:2]
    expected_train = expected_perm[4:].tolist()[:3]
    assert captured_train_kwargs["prior"].indices.tolist() == expected_train
    assert captured_train_kwargs["val_prior"].indices.tolist() == expected_val


def test_pretrain_regression_resume_accepts_legacy_top_level_target_normalization(
    monkeypatch, tmp_path: Path
) -> None:
    prior_path = tmp_path / "prior.h5"
    weights_path = tmp_path / "weights.pth"
    buckets_path = tmp_path / "buckets.pth"
    checkpoint_path = tmp_path / "legacy_resume_checkpoint.pth"
    _create_minimal_prior_dump(prior_path)

    torch.save(
        {
            "epoch": 0,
            "architecture": {
                "num_layers": 2,
                "embedding_size": 32,
                "num_attention_heads": 2,
                "mlp_hidden_size": 64,
                "num_outputs": 1,
                "dropout": 0.0,
                "feature_normalization": "per_function_zscore",
            },
            "model": {},
            "optimizer": {},
            "optimizer_name": "adamw",
            "regression_loss": "mse",
            "target_normalization": "none",
            "training": {},
        },
        checkpoint_path,
    )

    monkeypatch.setattr(
        pretrain_regression,
        "get_default_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        pretrain_regression,
        "make_global_bucket_edges",
        lambda filename, n_buckets, device, indices=None: torch.linspace(
            -1.0,
            1.0,
            n_buckets + 1,
        ),
    )
    monkeypatch.setattr(
        pretrain_regression.NanoTabPFNModel,
        "load_state_dict",
        lambda self, state_dict, strict=True: None,
    )

    def _fake_train(**kwargs):
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
            "--val_split",
            "0.5",
            "--regression_loss",
            "mse",
            "--target_normalization",
            "none",
            "--feature_normalization",
            "per_function_zscore",
            "--optimizer",
            "adamw",
            "--loadcheckpoint",
            str(checkpoint_path),
            "--runname",
            "legacy-resume-smoke",
        ]
    )

    assert weights_path.exists()


def test_pretrain_regression_warm_start_loads_model_but_resets_training_state(
    monkeypatch, tmp_path: Path
) -> None:
    prior_path = tmp_path / "prior.h5"
    weights_path = tmp_path / "weights.pth"
    buckets_path = tmp_path / "buckets.pth"
    checkpoint_path = tmp_path / "warm_start_checkpoint.pth"
    _create_minimal_prior_dump(prior_path)

    torch.save(
        {
            "epoch": 7,
            "architecture": {
                "num_layers": 2,
                "embedding_size": 32,
                "num_attention_heads": 2,
                "mlp_hidden_size": 64,
                "num_outputs": 1,
                "dropout": 0.0,
                "feature_normalization": "per_function_zscore",
            },
            "model": {"dummy": torch.tensor(1.0)},
            "optimizer": {"state": {}, "param_groups": []},
            "optimizer_name": "adamw",
            "regression_loss": "mse",
            "target_normalization": "none",
            "best_metric": 0.123,
            "best_epoch": 7,
            "early_stopping_state": {"patience_counter": 5},
        },
        checkpoint_path,
    )

    monkeypatch.setattr(
        pretrain_regression,
        "get_default_device",
        lambda: torch.device("cpu"),
    )
    monkeypatch.setattr(
        pretrain_regression,
        "make_global_bucket_edges",
        lambda filename, n_buckets, device, indices=None: torch.linspace(
            -1.0,
            1.0,
            n_buckets + 1,
        ),
    )
    loaded_state: dict[str, object] = {}
    original_load_state_dict = pretrain_regression.NanoTabPFNModel.load_state_dict

    def _capture_load_state_dict(self, state_dict, strict=True):
        loaded_state["state_dict"] = state_dict
        return original_load_state_dict(self, {}, strict=False)

    monkeypatch.setattr(
        pretrain_regression.NanoTabPFNModel,
        "load_state_dict",
        _capture_load_state_dict,
    )
    captured_train_kwargs: dict[str, object] = {}

    def _fake_train(**kwargs):
        captured_train_kwargs.update(kwargs)
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
            "2",
            "--steps",
            "1",
            "--batchsize",
            "1",
            "--val_split",
            "0.5",
            "--regression_loss",
            "mse",
            "--target_normalization",
            "none",
            "--feature_normalization",
            "per_function_zscore",
            "--optimizer",
            "adamw",
            "--loadcheckpoint",
            str(checkpoint_path),
            "--warm_start",
            "--runname",
            "warm-start-smoke",
        ]
    )

    assert torch.equal(loaded_state["state_dict"]["dummy"], torch.tensor(1.0))
    assert captured_train_kwargs["ckpt"] is None
    assert captured_train_kwargs["prior"].pointer == 0
