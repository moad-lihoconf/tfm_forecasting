import random

import h5py
import numpy as np
import torch
from pfns.bar_distribution import get_bucket_limits


def set_randomness_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def get_default_device():
    device = "cpu"
    if torch.backends.mps.is_available():
        device = "mps"
    if torch.cuda.is_available():
        device = "cuda"
    return device


def make_global_bucket_edges(
    filename,
    n_buckets=100,
    device=None,
    max_y=5_000_000,
    indices: list[int] | np.ndarray | None = None,
):
    if device is None:
        device = get_default_device()
    with h5py.File(filename, "r") as f:
        y = f["y"]
        num_tables, stored_max_seq_len = y.shape
        if indices is None:
            selected_indices = np.arange(num_tables, dtype=np.int64)
        else:
            selected_indices = np.asarray(indices, dtype=np.int64).reshape(-1)
            if selected_indices.size == 0:
                raise ValueError("indices must be non-empty when provided.")
            if selected_indices.min() < 0 or selected_indices.max() >= num_tables:
                raise ValueError("indices contain out-of-range row ids for this dump.")

        # h5py fancy indexing requires monotonically increasing indices. Preserve the
        # caller's order after loading metadata so downstream sampling behavior stays
        # unchanged when `indices` comes from a random permutation.
        sorted_order = np.argsort(selected_indices, kind="stable")
        sorted_indices = selected_indices[sorted_order]
        inverse_order = np.empty_like(sorted_order)
        inverse_order[sorted_order] = np.arange(sorted_order.size)

        single_eval_pos = np.asarray(
            f["single_eval_pos"][sorted_indices],
            dtype=np.int64,
        )[inverse_order]
        if "num_datapoints" in f:
            num_datapoints = np.asarray(
                f["num_datapoints"][sorted_indices],
                dtype=np.int64,
            )[inverse_order]
        else:
            num_datapoints = np.full(
                selected_indices.shape[0],
                stored_max_seq_len,
                dtype=np.int64,
            )

        valid_target_counts = np.clip(num_datapoints - single_eval_pos, 0, None)
        keep_mask = valid_target_counts > 0
        selected_indices = selected_indices[keep_mask]
        single_eval_pos = single_eval_pos[keep_mask]
        num_datapoints = num_datapoints[keep_mask]
        valid_target_counts = valid_target_counts[keep_mask]
        if selected_indices.size == 0:
            raise ValueError("No valid target rows available for bucket estimation.")

        budget = max(1, int(max_y))
        cumulative_targets = np.cumsum(valid_target_counts)
        num_tables_to_use = (
            int(np.searchsorted(cumulative_targets, budget, side="right")) + 1
        )
        num_tables_to_use = min(num_tables_to_use, selected_indices.size)
        selected_indices = selected_indices[:num_tables_to_use]
        single_eval_pos = single_eval_pos[:num_tables_to_use]
        num_datapoints = num_datapoints[:num_tables_to_use]

        normalized_targets: list[np.ndarray] = []
        for row_id, eval_pos, total_rows in zip(
            selected_indices,
            single_eval_pos,
            num_datapoints,
            strict=False,
        ):
            row = np.asarray(y[int(row_id), : int(total_rows)], dtype=np.float32)
            train_targets = row[: int(eval_pos)]
            test_targets = row[int(eval_pos) : int(total_rows)]
            if train_targets.size == 0 or test_targets.size == 0:
                continue
            ddof = 1 if train_targets.size > 1 else 0
            y_mean = float(train_targets.mean())
            y_std = float(train_targets.std(ddof=ddof)) + 1e-8
            normalized_targets.append((test_targets - y_mean) / y_std)

    if not normalized_targets:
        raise ValueError(
            "Could not collect normalized target values for bucket estimation."
        )
    ys_concat = np.concatenate(normalized_targets, axis=0)

    if ys_concat.size < n_buckets:
        raise ValueError(
            f"Too few target samples ({ys_concat.size}) to compute {n_buckets} buckets."
        )

    ys_tensor = torch.tensor(ys_concat, dtype=torch.float32, device=device)
    global_bucket_edges = get_bucket_limits(n_buckets, ys=ys_tensor).to(device)
    return global_bucket_edges
