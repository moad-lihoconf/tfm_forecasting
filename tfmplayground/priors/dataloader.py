"""Data loading utilities for tabular priors."""

import contextlib
from collections.abc import Callable, Iterator
from typing import TYPE_CHECKING

import h5py
import torch
from torch.utils.data import DataLoader

try:
    from tabicl.prior.dataset import PriorDataset as TabICLPriorDataset
except ModuleNotFoundError:  # pragma: no cover - optional dependency.
    TabICLPriorDataset = None

try:
    from ticl.dataloader import PriorDataLoader as TICLPriorDataset
except ModuleNotFoundError:  # pragma: no cover - optional dependency.
    TICLPriorDataset = None

if TYPE_CHECKING:
    from .dynscm import DynSCMConfig


class PriorDataLoader(DataLoader):
    """Generic DataLoader for synthetic data generation using a get_batch function.

    Args:
        get_batch_function (Callable): A function returning batches of data.
        num_steps (int): Number of batches per epoch.
        batch_size (int): Number of functions per batch.
        num_datapoints_max (int): Max sequence length per function.
        num_features (int): Number of input features.
        device (torch.device): Device to move tensors to.
    """

    def __init__(
        self,
        get_batch_function: Callable[..., dict[str, torch.Tensor | int]],
        num_steps: int,
        batch_size: int,
        num_datapoints_max: int,
        num_features: int,
        device: torch.device,
    ):
        self.get_batch_function = get_batch_function
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_datapoints_max = num_datapoints_max
        self.num_features = num_features
        self.device = device

    def __iter__(self) -> Iterator[dict[str, torch.Tensor | int]]:
        return iter(
            self.get_batch_function(
                self.batch_size, self.num_datapoints_max, self.num_features
            )
            for _ in range(self.num_steps)
        )

    def __len__(self) -> int:
        return self.num_steps

    def close(self) -> None:
        close_fn = getattr(self.get_batch_function, "close", None)
        if callable(close_fn):
            close_fn()

    def __del__(self) -> None:
        with contextlib.suppress(Exception):
            self.close()


class PriorDumpDataLoader(DataLoader):
    """DataLoader that loads synthetic prior data from an HDF5 dump.

    Args:
        filename (str): Path to the HDF5 file.
        num_steps (int): Number of batches per epoch.
        batch_size (int): Batch size.
        device (torch.device): Device to load tensors onto.
    """

    def __init__(
        self,
        filename,
        num_steps,
        batch_size,
        device,
        starting_index=0,
        indices: list[int] | None = None,
    ):
        self.filename = filename
        self.num_steps = num_steps
        self.batch_size = batch_size
        with h5py.File(self.filename, "r") as f:
            self.total_num_functions = int(f["X"].shape[0])
            self.num_datapoints_max = self.total_num_functions
            if "max_num_classes" in f:
                self.max_num_classes = f["max_num_classes"][0]
            else:
                self.max_num_classes = None
            self.problem_type = f["problem_type"][()].decode("utf-8")
            self.has_num_datapoints = "num_datapoints" in f
            self.stored_max_seq_len = f["X"].shape[1]
        self.device = device
        self.indices = (
            torch.arange(self.total_num_functions, dtype=torch.long)
            if indices is None
            else torch.as_tensor(indices, dtype=torch.long)
        )
        if self.indices.ndim != 1 or self.indices.numel() == 0:
            raise ValueError("indices must be a non-empty 1D list of row indices.")
        if (
            int(self.indices.min()) < 0
            or int(self.indices.max()) >= self.total_num_functions
        ):
            raise ValueError("indices contain out-of-range row ids for this dump.")
        self.num_functions = int(self.indices.numel())
        self.pointer = starting_index % self.num_functions

    def _next_batch_indices(self) -> torch.Tensor:
        positions = (torch.arange(self.batch_size) + self.pointer) % self.num_functions
        batch_indices = self.indices[positions]
        self.pointer = (self.pointer + self.batch_size) % self.num_functions
        return batch_indices

    def __iter__(self):
        with h5py.File(self.filename, "r") as f:
            for _ in range(self.num_steps):
                batch_idx = self._next_batch_indices().cpu().numpy()
                sort_order = batch_idx.argsort()
                sorted_idx = batch_idx[sort_order]
                inverse_order = sort_order.argsort()

                num_features = f["num_features"][sorted_idx].max()
                if self.has_num_datapoints:
                    num_datapoints_batch = f["num_datapoints"][sorted_idx]
                    max_seq_in_batch = int(num_datapoints_batch.max())
                else:
                    max_seq_in_batch = int(self.stored_max_seq_len)

                x_np = f["X"][sorted_idx, :max_seq_in_batch, :num_features][
                    inverse_order
                ]
                y_np = f["y"][sorted_idx, :max_seq_in_batch][inverse_order]
                single_eval_pos = f["single_eval_pos"][sorted_idx][inverse_order]
                x = torch.from_numpy(x_np)
                y = torch.from_numpy(y_np)
                single_eval_pos_tensor = torch.as_tensor(
                    single_eval_pos, dtype=torch.long, device=self.device
                )
                single_eval_pos_value: int | torch.Tensor
                if torch.all(single_eval_pos_tensor == single_eval_pos_tensor[0]):
                    single_eval_pos_value = int(single_eval_pos_tensor[0].item())
                else:
                    single_eval_pos_value = single_eval_pos_tensor

                yield dict(
                    x=x.to(self.device),
                    y=y.to(self.device),
                    target_y=y.to(
                        self.device
                    ),  # target_y is identical to y (for downstream compatibility)
                    single_eval_pos=single_eval_pos_value,
                )

    def __len__(self):
        return self.num_steps


class TabICLPriorDataLoader(DataLoader):
    """DataLoader sampling synthetic prior data on-the-fly from TabICL's PriorDataset.

    Args:
        num_steps (int): Number of batches to generate per epoch.
        batch_size (int): Number of functions per batch.
        num_datapoints_min (int): Minimum number of datapoints per function.
        num_datapoints_max (int): Maximum number of datapoints per function.
        min_features (int): Minimum number of features in x.
        max_features (int): Maximum number of features in x.
        max_num_classes (int): Maximum number of classes (for classification tasks).
        prior_type (str): Type of prior: 'mlp_scm', 'tree_scm',
            'mix_scm' (default), or 'dummy'.
        device (torch.device): Target device for tensors.
    """

    def __init__(
        self,
        num_steps: int,
        batch_size: int,
        num_datapoints_min: int,
        num_datapoints_max: int,
        min_features: int,
        max_features: int,
        max_num_classes: int,
        device: torch.device,
        prior_type: str = "mix_scm",
    ):
        self.num_steps = num_steps
        self.batch_size = batch_size
        self.num_datapoints_min = num_datapoints_min
        self.num_datapoints_max = num_datapoints_max
        self.min_features = min_features
        self.max_features = max_features
        self.max_num_classes = max_num_classes
        self.prior_type = prior_type
        self.device = device

        if TabICLPriorDataset is None:
            raise RuntimeError(
                "TabICL dependencies are unavailable. Install tabicl extras "
                "to use TabICLPriorDataLoader."
            )

        self.pd = TabICLPriorDataset(
            batch_size=batch_size,
            batch_size_per_gp=batch_size,
            min_features=min_features,
            max_features=max_features,
            max_classes=max_num_classes,
            min_seq_len=num_datapoints_min,
            max_seq_len=num_datapoints_max,
            prior_type=prior_type,
        )

    def tabicl_to_ours(self, d):
        x, y, active_features, seqlen, train_size = d
        # Should match in practice for batch_size_per_gp=batch_size.
        active_features = active_features[0].item()
        x = x[:, :, :active_features]
        single_eval_pos = train_size[0].item()
        return dict(
            x=x.to(self.device),
            y=y.to(self.device),
            target_y=y.to(
                self.device
            ),  # target_y is identical to y (for downstream compatibility)
            single_eval_pos=single_eval_pos,
        )

    def __iter__(self):
        return iter(self.tabicl_to_ours(next(self.pd)) for _ in range(self.num_steps))

    def __len__(self):
        return self.num_steps


class DynSCMPriorDataLoader(PriorDataLoader):
    """DataLoader sampling DynSCM forecasting tables on-the-fly.

    Args:
        cfg: DynSCM configuration object.
        num_steps: Number of batches per epoch.
        batch_size: Number of functions sampled per batch.
        num_datapoints_max: Maximum sequence length per function.
        num_features: Number of input features.
        device: Target device for tensors.
        seed: Optional deterministic seed for closure RNG state.
        workers: Number of process workers (1 keeps in-process generation).
        worker_blas_threads: BLAS thread cap per worker process.
    """

    def __init__(
        self,
        cfg: "DynSCMConfig",
        num_steps: int,
        batch_size: int,
        num_datapoints_max: int,
        num_features: int,
        device: torch.device,
        seed: int | None = None,
        workers: int = 1,
        worker_blas_threads: int = 1,
    ):
        from .dynscm import make_get_batch_dynscm

        super().__init__(
            get_batch_function=make_get_batch_dynscm(
                cfg,
                device=device,
                seed=seed,
                workers=workers,
                worker_blas_threads=worker_blas_threads,
            ),
            num_steps=num_steps,
            batch_size=batch_size,
            num_datapoints_max=num_datapoints_max,
            num_features=num_features,
            device=device,
        )


class TICLPriorDataLoader(DataLoader):
    """DataLoader sampling synthetic prior data from TICL's PriorDataLoader.

    Args:
        prior (Any): A TICL prior object supporting get_batch.
        num_steps (int): Number of batches per epoch.
        batch_size (int): Number of functions sampled per batch.
        num_datapoints_max (int): Number of datapoints sampled per function.
        num_features (int): Dimensionality of x vectors.
        device (torch.device): Target device for tensors.
        min_eval_pos (int, optional): Minimum evaluation position in the sequence.
    """

    def __init__(
        self,
        prior,
        num_steps: int,
        batch_size: int,
        num_datapoints_max: int,
        num_features: int,
        min_eval_pos: int,
        device: torch.device,
    ):
        self.num_steps = num_steps
        self.device = device

        if TICLPriorDataset is None:
            raise RuntimeError(
                "TICL dependencies are unavailable. Install ticl extras "
                "to use TICLPriorDataLoader."
            )

        self.pd = TICLPriorDataset(
            prior=prior,
            num_steps=num_steps,
            batch_size=batch_size,
            min_eval_pos=min_eval_pos,
            n_samples=num_datapoints_max,
            device=device,
            num_features=num_features,
        )

    def ticl_to_ours(self, d):
        (info, x, y), target_y, single_eval_pos = d
        x = x.permute(1, 0, 2)
        y = y.permute(1, 0)
        target_y = target_y.permute(1, 0)

        return dict(
            x=x.to(self.device),
            y=y.to(self.device),
            target_y=target_y.to(
                self.device
            ),  # target_y is identical to y (for downstream compatibility)
            single_eval_pos=single_eval_pos,
        )

    def __iter__(self):
        return (self.ticl_to_ours(batch) for batch in self.pd)

    def __len__(self):
        return self.num_steps
