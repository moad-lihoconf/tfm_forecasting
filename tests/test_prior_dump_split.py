from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch

from tfmplayground.priors.dataloader import PriorDumpDataLoader


def _write_tiny_dump(path: Path, n: int = 10, rows: int = 6, cols: int = 3) -> None:
    with h5py.File(path, "w") as f:
        x = np.zeros((n, rows, cols), dtype=np.float32)
        y = np.zeros((n, rows), dtype=np.float32)
        num_features = np.full((n,), cols, dtype=np.int32)
        single_eval_pos = np.full((n,), rows // 2, dtype=np.int32)
        for i in range(n):
            x[i, :, :] = float(i)
            y[i, :] = float(i)
        f.create_dataset("X", data=x)
        f.create_dataset("y", data=y)
        f.create_dataset("num_features", data=num_features)
        f.create_dataset("single_eval_pos", data=single_eval_pos)
        f.create_dataset("problem_type", data=np.bytes_("regression"))


def test_prior_dump_train_val_split_indices_are_disjoint_and_deterministic(
    tmp_path: Path,
):
    dump_path = tmp_path / "tiny_dump.h5"
    _write_tiny_dump(dump_path)

    total = 10
    val_split = 0.2
    torch.manual_seed(2402)
    perm = torch.randperm(total)
    val_count = int(total * val_split)
    val_indices = perm[:val_count].tolist()
    train_indices = perm[val_count:].tolist()

    assert set(train_indices).isdisjoint(set(val_indices))
    assert sorted(train_indices + val_indices) == list(range(total))

    train_loader = PriorDumpDataLoader(
        filename=str(dump_path),
        num_steps=4,
        batch_size=2,
        device=torch.device("cpu"),
        indices=train_indices,
    )
    val_loader = PriorDumpDataLoader(
        filename=str(dump_path),
        num_steps=1,
        batch_size=2,
        device=torch.device("cpu"),
        indices=val_indices,
    )

    # x is filled by source row id, so unique values identify split membership.
    train_batch = next(iter(train_loader))
    val_batch = next(iter(val_loader))
    assert isinstance(train_batch["single_eval_pos"], int)
    assert isinstance(val_batch["single_eval_pos"], int)
    assert isinstance(train_batch["num_datapoints"], int)
    assert isinstance(val_batch["num_datapoints"], int)
    assert train_batch["target_mask"].dtype == torch.bool
    assert val_batch["target_mask"].dtype == torch.bool
    train_ids = set(torch.unique(train_batch["x"]).to(torch.int64).tolist())
    val_ids = set(torch.unique(val_batch["x"]).to(torch.int64).tolist())
    assert train_ids.isdisjoint(val_ids)


def test_prior_dump_loader_preserves_per_row_single_eval_pos(tmp_path: Path):
    dump_path = tmp_path / "mixed_eval_pos.h5"
    with h5py.File(dump_path, "w") as f:
        x = np.zeros((4, 6, 2), dtype=np.float32)
        y = np.zeros((4, 6), dtype=np.float32)
        num_features = np.full((4,), 2, dtype=np.int32)
        single_eval_pos = np.array([2, 4, 2, 4], dtype=np.int32)
        for i in range(4):
            x[i, :, :] = float(i)
            y[i, :] = float(i)
        f.create_dataset("X", data=x)
        f.create_dataset("y", data=y)
        f.create_dataset("num_features", data=num_features)
        f.create_dataset("single_eval_pos", data=single_eval_pos)
        f.create_dataset("problem_type", data=np.bytes_("regression"))

    loader = PriorDumpDataLoader(
        filename=str(dump_path),
        num_steps=1,
        batch_size=2,
        device=torch.device("cpu"),
        indices=[0, 1],
    )
    batch = next(iter(loader))
    assert torch.is_tensor(batch["single_eval_pos"])
    assert batch["single_eval_pos"].dtype == torch.long
    assert batch["single_eval_pos"].tolist() == [2, 4]
    assert isinstance(batch["num_datapoints"], int)
    assert batch["num_datapoints"] == 6
    assert batch["target_mask"].dtype == torch.bool
    assert batch["target_mask"].tolist() == [
        [False, False, True, True, True, True],
        [False, False, False, False, True, True],
    ]
