from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest
import torch
from pfns.bar_distribution import get_bucket_limits
from torch import nn

import pretrain_regression
import tfmplayground.train as train_mod
from tfmplayground.priors.dataloader import PriorDumpDataLoader
from tfmplayground.priors.utils import dump_prior_to_h5
from tfmplayground.utils import make_global_bucket_edges


class _ZeroOutputModel(nn.Module):
    def forward(self, data, single_eval_pos: int):
        x, _y = data
        return torch.zeros(
            (x.shape[0], x.shape[1] - int(single_eval_pos)),
            dtype=torch.float32,
            device=x.device,
        )


class _AbsCriterion:
    def __call__(self, output: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        return torch.abs(output - targets)


def _write_dump(
    path: Path,
    *,
    x: np.ndarray,
    y: np.ndarray,
    num_datapoints: np.ndarray,
    single_eval_pos: np.ndarray,
) -> None:
    with h5py.File(path, "w") as f:
        f.create_dataset("X", data=x.astype(np.float32))
        f.create_dataset("y", data=y.astype(np.float32))
        f.create_dataset(
            "num_features",
            data=np.full((x.shape[0],), x.shape[2], dtype=np.int32),
        )
        f.create_dataset("num_datapoints", data=num_datapoints.astype(np.int32))
        f.create_dataset("single_eval_pos", data=single_eval_pos.astype(np.int32))
        f.create_dataset("problem_type", data="regression", dtype=h5py.string_dtype())


def test_masked_loss_excludes_padded_targets() -> None:
    model = _ZeroOutputModel()
    criterion = _AbsCriterion()

    full_data = {
        "x": torch.zeros((1, 5, 3), dtype=torch.float32),
        "y": torch.tensor([[-1.0, 1.0, 1.0, 100.0, 3.0]], dtype=torch.float32),
        "target_y": torch.tensor([[-1.0, 1.0, 1.0, 100.0, 3.0]], dtype=torch.float32),
        "single_eval_pos": 2,
        "target_mask": torch.tensor([[False, False, True, False, True]]),
    }

    masked = train_mod._compute_loss(
        wrapped_model=model,
        criterion=criterion,
        full_data=full_data,
        device=torch.device("cpu"),
        regression_task=True,
        classification_task=False,
    )
    unmasked = train_mod._compute_loss(
        wrapped_model=model,
        criterion=criterion,
        full_data={k: v for k, v in full_data.items() if k != "target_mask"},
        device=torch.device("cpu"),
        regression_task=True,
        classification_task=False,
    )

    assert float(masked) < float(unmasked)
    assert masked == pytest.approx(1.4142135, rel=1e-5)


def test_dump_prior_persists_true_num_datapoints(tmp_path: Path) -> None:
    path = tmp_path / "dump.h5"

    class _OneBatchPrior:
        def __iter__(self):
            x = torch.ones((2, 6, 3), dtype=torch.float32)
            y = torch.arange(12, dtype=torch.float32).reshape(2, 6)
            yield {
                "x": x,
                "y": y,
                "single_eval_pos": torch.tensor([2, 3], dtype=torch.long),
                "num_datapoints": torch.tensor([4, 6], dtype=torch.long),
            }

    dump_prior_to_h5(
        _OneBatchPrior(),
        max_classes=0,
        batch_size=2,
        save_path=str(path),
        problem_type="regression",
        max_seq_len=6,
        max_features=3,
    )

    with h5py.File(path, "r") as f:
        assert f["num_datapoints"][:].tolist() == [4, 6]
        assert f["single_eval_pos"][:].tolist() == [2, 3]


def test_prior_dump_loader_emits_target_mask_for_mixed_lengths(tmp_path: Path) -> None:
    path = tmp_path / "mixed_lengths.h5"
    x = np.zeros((2, 6, 3), dtype=np.float32)
    x[0, :4, :] = 1.0
    x[1, :6, :] = 1.0
    y = np.arange(12, dtype=np.float32).reshape(2, 6)
    _write_dump(
        path,
        x=x,
        y=y,
        num_datapoints=np.array([4, 6], dtype=np.int32),
        single_eval_pos=np.array([2, 3], dtype=np.int32),
    )

    loader = PriorDumpDataLoader(
        filename=str(path),
        num_steps=1,
        batch_size=2,
        device=torch.device("cpu"),
    )
    batch = next(iter(loader))

    assert torch.is_tensor(batch["num_datapoints"])
    assert batch["num_datapoints"].tolist() == [4, 6]
    assert batch["target_mask"].tolist() == [
        [False, False, True, True, False, False],
        [False, False, False, True, True, True],
    ]


def test_resume_starting_index_scales_with_batchsize(
    monkeypatch,
    tmp_path: Path,
) -> None:
    prior_path = tmp_path / "prior.h5"
    weights_path = tmp_path / "weights.pth"
    buckets_path = tmp_path / "buckets.pth"
    ckpt_path = tmp_path / "resume.pth"

    x = np.ones((20, 6, 3), dtype=np.float32)
    y = np.linspace(-1.0, 1.0, num=120, dtype=np.float32).reshape(20, 6)
    _write_dump(
        prior_path,
        x=x,
        y=y,
        num_datapoints=np.full((20,), 6, dtype=np.int32),
        single_eval_pos=np.full((20,), 3, dtype=np.int32),
    )
    torch.save(
        {
            "epoch": 2,
            "architecture": {"dropout": 0.1},
            "model": {},
        },
        ckpt_path,
    )

    starting_indices: list[int] = []

    class _FakeLoader:
        def __init__(
            self,
            *,
            filename,
            num_steps,
            batch_size,
            device,
            starting_index=0,
            indices=None,
        ):
            del filename, device, indices
            self.num_steps = num_steps
            self.batch_size = batch_size
            starting_indices.append(int(starting_index))

        def __len__(self):
            return int(self.num_steps)

    class _FakeModel:
        def __init__(self, **kwargs):
            del kwargs
            self.num_layers = 1
            self.embedding_size = 8
            self.num_attention_heads = 1
            self.mlp_hidden_size = 16
            self.num_outputs = 8
            self.dropout = 0.1

        def load_state_dict(self, state):
            del state

        def state_dict(self):
            return {}

        def to(self, _device):
            return self

    monkeypatch.setattr(pretrain_regression, "PriorDumpDataLoader", _FakeLoader)
    monkeypatch.setattr(pretrain_regression, "NanoTabPFNModel", _FakeModel)
    monkeypatch.setattr(pretrain_regression, "get_default_device", lambda: "cpu")
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
        pretrain_regression,
        "audit_prior_dump",
        lambda *args, **kwargs: {
            "has_num_datapoints_dataset": True,
            "inferred_padded_target_fraction": 0.0,
            "inferred_num_datapoints_mismatch_fraction": 0.0,
            "feature_budget_saturation_fraction": 0.0,
            "train_y_std_mean": 0.1,
            "target_y_std_mean": 0.1,
        },
    )
    monkeypatch.setattr(pretrain_regression, "integrity_errors", lambda *_a, **_k: [])
    monkeypatch.setattr(
        pretrain_regression,
        "train",
        lambda **kwargs: (
            kwargs["model"],
            {"best_epoch": 1, "best_metric": 0.1, "stop_reason": "completed"},
        ),
    )

    pretrain_regression.main(
        [
            "--priordump",
            str(prior_path),
            "--saveweights",
            str(weights_path),
            "--savebuckets",
            str(buckets_path),
            "--loadcheckpoint",
            str(ckpt_path),
            "--steps",
            "3",
            "--batchsize",
            "4",
            "--epochs",
            "1",
            "--n_buckets",
            "8",
            "--val_split",
            "0.2",
            "--no-strict_prior_integrity",
        ]
    )

    assert starting_indices[0] == 24
    assert starting_indices[1] == 0


def test_bucket_edges_use_only_valid_targets(tmp_path: Path) -> None:
    path = tmp_path / "buckets.h5"
    x = np.ones((2, 6, 3), dtype=np.float32)
    y = np.array(
        [
            [0.0, 1.0, 2.0, 4.0, 6.0, 8.0],
            [0.0, 1.0, 100.0, 0.0, 0.0, 0.0],
        ],
        dtype=np.float32,
    )
    num_datapoints = np.array([6, 3], dtype=np.int32)
    single_eval_pos = np.array([2, 2], dtype=np.int32)
    _write_dump(
        path,
        x=x,
        y=y,
        num_datapoints=num_datapoints,
        single_eval_pos=single_eval_pos,
    )

    computed = make_global_bucket_edges(
        filename=str(path),
        n_buckets=4,
        device="cpu",
    )

    normalized_targets: list[np.ndarray] = []
    for row_idx in range(2):
        sep = int(single_eval_pos[row_idx])
        nd = int(num_datapoints[row_idx])
        train_target = y[row_idx, :sep]
        test_target = y[row_idx, sep:nd]
        normalized_targets.append(
            (test_target - train_target.mean()) / (train_target.std(ddof=1) + 1e-8)
        )
    ys = torch.tensor(
        np.concatenate(normalized_targets, axis=0),
        dtype=torch.float32,
    )
    expected = get_bucket_limits(4, ys=ys)
    assert torch.allclose(computed.cpu(), expected.cpu())


def test_bucket_edges_accept_unsorted_indices(tmp_path: Path) -> None:
    path = tmp_path / "buckets_unsorted.h5"
    x = np.ones((3, 6, 3), dtype=np.float32)
    y = np.array(
        [
            [0.0, 1.0, 2.0, 4.0, 6.0, 8.0],
            [1.0, 3.0, 5.0, 7.0, 9.0, 11.0],
            [2.0, 4.0, 6.0, 8.0, 10.0, 12.0],
        ],
        dtype=np.float32,
    )
    num_datapoints = np.array([6, 6, 6], dtype=np.int32)
    single_eval_pos = np.array([2, 3, 2], dtype=np.int32)
    _write_dump(
        path,
        x=x,
        y=y,
        num_datapoints=num_datapoints,
        single_eval_pos=single_eval_pos,
    )

    computed = make_global_bucket_edges(
        filename=str(path),
        n_buckets=4,
        device="cpu",
        indices=[2, 0, 1],
    )

    normalized_targets: list[np.ndarray] = []
    for row_idx in [2, 0, 1]:
        sep = int(single_eval_pos[row_idx])
        nd = int(num_datapoints[row_idx])
        train_target = y[row_idx, :sep]
        test_target = y[row_idx, sep:nd]
        normalized_targets.append(
            (test_target - train_target.mean()) / (train_target.std(ddof=1) + 1e-8)
        )
    ys = torch.tensor(
        np.concatenate(normalized_targets, axis=0),
        dtype=torch.float32,
    )
    expected = get_bucket_limits(4, ys=ys)
    assert torch.allclose(computed.cpu(), expected.cpu())


def test_loader_to_compute_loss_integration_ignores_padded_rows(tmp_path: Path) -> None:
    path = tmp_path / "integration.h5"
    x = np.zeros((2, 6, 3), dtype=np.float32)
    x[0, :4, :] = 1.0
    x[1, :5, :] = 1.0
    y = np.array(
        [
            [-1.0, 1.0, 2.0, 4.0, 0.0, 0.0],
            [-2.0, 2.0, 1.0, 5.0, 6.0, 0.0],
        ],
        dtype=np.float32,
    )
    _write_dump(
        path,
        x=x,
        y=y,
        num_datapoints=np.array([4, 5], dtype=np.int32),
        single_eval_pos=np.array([2, 3], dtype=np.int32),
    )

    loader = PriorDumpDataLoader(
        filename=str(path),
        num_steps=1,
        batch_size=2,
        device=torch.device("cpu"),
    )
    batch = next(iter(loader))
    model = _ZeroOutputModel()
    criterion = _AbsCriterion()
    loss = train_mod._compute_loss(
        wrapped_model=model,
        criterion=criterion,
        full_data=batch,
        device=torch.device("cpu"),
        regression_task=True,
        classification_task=False,
    )
    assert torch.isfinite(loss)

    expected_vals = []
    # sample 0
    expected_vals.extend([(2.0 - 0.0) / 1.4142135, (4.0 - 0.0) / 1.4142135])
    # sample 1, train=[-2,2,1]
    std_sample_1 = float(np.std(np.array([-2.0, 2.0, 1.0]), ddof=1)) + 1e-8
    expected_vals.extend(
        [
            (5.0 - (1.0 / 3.0)) / std_sample_1,
            (6.0 - (1.0 / 3.0)) / std_sample_1,
        ]
    )
    expected = float(np.mean(np.abs(np.array(expected_vals, dtype=np.float32))))
    assert float(loss) == pytest.approx(expected, rel=1e-5)


def test_validation_loss_is_stable_across_repeated_calls(tmp_path: Path) -> None:
    path = tmp_path / "stable_val.h5"
    x = np.ones((4, 4, 3), dtype=np.float32)
    y = np.array(
        [
            [0.0, 1.0, 10.0, 10.0],
            [0.0, 1.0, 20.0, 20.0],
            [0.0, 1.0, 30.0, 30.0],
            [0.0, 1.0, 40.0, 40.0],
        ],
        dtype=np.float32,
    )
    _write_dump(
        path,
        x=x,
        y=y,
        num_datapoints=np.full((4,), 4, dtype=np.int32),
        single_eval_pos=np.full((4,), 2, dtype=np.int32),
    )

    val_loader = PriorDumpDataLoader(
        filename=str(path),
        num_steps=1,
        batch_size=2,
        device=torch.device("cpu"),
    )
    model = _ZeroOutputModel()
    criterion = _AbsCriterion()
    first = train_mod._evaluate_prior_loss(
        wrapped_model=model,
        val_prior=val_loader,
        criterion=criterion,
        device=torch.device("cpu"),
        regression_task=True,
        classification_task=False,
    )
    second = train_mod._evaluate_prior_loss(
        wrapped_model=model,
        val_prior=val_loader,
        criterion=criterion,
        device=torch.device("cpu"),
        regression_task=True,
        classification_task=False,
    )
    assert first == pytest.approx(second)


def test_pretrain_strict_integrity_rejects_legacy_like_dump(tmp_path: Path) -> None:
    path = tmp_path / "legacy_like.h5"
    x = np.zeros((8, 6, 3), dtype=np.float32)
    y = np.zeros((8, 6), dtype=np.float32)
    _write_dump(
        path,
        x=x,
        y=y,
        num_datapoints=np.full((8,), 6, dtype=np.int32),
        single_eval_pos=np.full((8,), 3, dtype=np.int32),
    )

    with pytest.raises(ValueError, match="Prior integrity audit failed"):
        pretrain_regression.main(
            [
                "--priordump",
                str(path),
                "--saveweights",
                str(tmp_path / "weights.pth"),
                "--savebuckets",
                str(tmp_path / "buckets.pth"),
                "--steps",
                "1",
                "--epochs",
                "1",
                "--batchsize",
                "1",
                "--n_buckets",
                "8",
            ]
        )
