"""Utility functions for priors."""

import h5py
import numpy as np
import torch
from tqdm import tqdm

from .config import get_tabpfn_prior_config, get_ticl_prior_config


def _coerce_batch_vector(
    raw_value: torch.Tensor | np.ndarray | int | float,
    *,
    batch_n: int,
    dtype: np.dtype,
    key: str,
) -> np.ndarray:
    if isinstance(raw_value, torch.Tensor):
        raw_np = raw_value.detach().to("cpu").numpy()
    else:
        raw_np = np.asarray(raw_value)

    if raw_np.ndim == 0:
        return np.full((batch_n,), raw_np.item(), dtype=dtype)

    vector = raw_np.reshape(-1)
    if vector.shape[0] != batch_n:
        raise ValueError(
            f"{key} length must match batch size when provided per row; "
            f"got {vector.shape[0]} vs {batch_n}."
        )
    return vector.astype(dtype, copy=False)


def build_ticl_prior(
    prior_type: str,
    base_prior: str | None = None,
    max_num_classes: int | None = None,
) -> object:
    """Build a TICL prior from defaults defined in `priors/config.py`.

    Args:
        prior_type: Type of TICL prior
            ('mlp', 'gp', 'classification_adapter', etc.).
        base_prior: Base regression prior for composite priors
            (e.g., 'mlp' or 'gp' for classification_adapter).
        max_num_classes: Maximum number of classes for classification priors
    """
    try:
        from ticl.priors import (
            BooleanConjunctionPrior,
            ClassificationAdapterPrior,
            GPPrior,
            MLPPrior,
            StepFunctionPrior,
        )
    except ModuleNotFoundError as exc:  # pragma: no cover - optional dependency.
        raise RuntimeError(
            "TICL dependencies are unavailable. Install optional TICL extras "
            "to use build_ticl_prior."
        ) from exc

    cfg = get_ticl_prior_config(prior_type)

    if prior_type == "mlp":
        return MLPPrior(cfg)
    elif prior_type == "gp":
        return GPPrior(cfg)
    elif prior_type == "classification_adapter":
        if base_prior is None:
            base_prior = "mlp"  # default to MLP
        # build the base regression prior
        base_prior_obj = build_ticl_prior(base_prior)

        # We equate these values rather than sampling `num_classes` separately:
        # - max_num_classes serves as the upper bound for TICL's internal sampling
        # - even with fixed num_classes, TICL's class_sampler_f() can still vary
        #   the effective class count (50% chance of 2, else uniform in
        #   [2, num_classes])
        cfg["max_num_classes"] = max_num_classes
        cfg["num_classes"] = max_num_classes
        return ClassificationAdapterPrior(base_prior_obj, **cfg)
    elif prior_type == "boolean_conjunctions":
        return BooleanConjunctionPrior(hyperparameters=cfg)
    elif prior_type == "step_function":
        return StepFunctionPrior(cfg)
    else:
        raise ValueError(f"Unsupported TICL prior type: {prior_type}")


def build_tabpfn_prior(prior_type: str, max_classes: int) -> dict:
    """Build TabPFN prior configuration for regression or classification.

    Args:
        prior_type: Type of TabPFN prior ('mlp', 'gp', 'prior_bag')
        max_classes: Maximum number of classes

    Returns:
        dict with 'flexible', 'max_num_classes', and 'prior_config' keys
    """
    is_regression = max_classes == 0

    return {
        "flexible": not is_regression,  # false for regression, true for classification
        "max_num_classes": 2
        if is_regression
        else max_classes,  # library requires >=2 for both tasks
        # num_classes parameter in the library code is equated to max_num_classes
        # so its not varied separately here
        "prior_config": {
            **get_tabpfn_prior_config(prior_type),
        },
    }


def dump_prior_to_h5(
    prior,
    max_classes: int,
    batch_size: int,
    save_path: str,
    problem_type: str,
    max_seq_len: int,
    max_features: int,
):
    """Dumps synthetic prior data into an HDF5 file for later training."""

    with h5py.File(save_path, "w") as f:
        dump_X = f.create_dataset(
            "X",
            shape=(0, max_seq_len, max_features),
            maxshape=(None, max_seq_len, max_features),
            chunks=(batch_size, max_seq_len, max_features),
            compression="lzf",
        )
        dump_num_features = f.create_dataset(
            "num_features",
            shape=(0,),
            maxshape=(None,),
            chunks=(batch_size,),
            dtype="i4",
        )
        dump_num_datapoints = f.create_dataset(
            "num_datapoints",
            shape=(0,),
            maxshape=(None,),
            chunks=(batch_size,),
            dtype="i4",
        )
        dump_y = f.create_dataset(
            "y",
            shape=(0, max_seq_len),
            maxshape=(None, max_seq_len),
            chunks=(batch_size, max_seq_len),
        )
        dump_single_eval_pos = f.create_dataset(
            "single_eval_pos",
            shape=(0,),
            maxshape=(None,),
            chunks=(batch_size,),
            dtype="i4",
        )
        optional_metadata_datasets: dict[str, h5py.Dataset] = {}
        optional_metadata_dtypes: dict[str, np.dtype] = {
            "sampled_mechanism_type_id": np.dtype("i4"),
            "sampled_noise_family_id": np.dtype("i4"),
            "sampled_missing_mode_id": np.dtype("i4"),
            "sampled_kernel_family_id": np.dtype("i4"),
            "sampled_student_df": np.dtype("f4"),
            "sampled_num_vars": np.dtype("i4"),
            "sampled_num_steps": np.dtype("i4"),
            "sampled_n_train": np.dtype("i4"),
            "sampled_n_test": np.dtype("i4"),
            "sampled_pre_budget_feature_count": np.dtype("i4"),
        }

        if problem_type == "classification":
            f.create_dataset(
                "max_num_classes", data=np.array((max_classes,)), chunks=(1,)
            )
        f.create_dataset(
            "original_batch_size", data=np.array((batch_size,)), chunks=(1,)
        )
        f.create_dataset("problem_type", data=problem_type, dtype=h5py.string_dtype())

        for e in tqdm(prior):
            x = e["x"].to("cpu").numpy()
            y = e["y"].to("cpu").numpy()
            batch_n = int(x.shape[0])

            single_eval_pos_raw = e["single_eval_pos"]
            if isinstance(single_eval_pos_raw, torch.Tensor):
                single_eval_pos_np = (
                    single_eval_pos_raw.detach().to("cpu").numpy().astype(np.int32)
                )
            else:
                single_eval_pos_np = np.asarray(single_eval_pos_raw, dtype=np.int32)
            if single_eval_pos_np.ndim == 0:
                single_eval_pos = np.full(
                    (batch_n,), int(single_eval_pos_np), dtype=np.int32
                )
            else:
                single_eval_pos = single_eval_pos_np.reshape(-1).astype(np.int32)
                if single_eval_pos.shape[0] != batch_n:
                    raise ValueError(
                        "single_eval_pos length must match batch size when "
                        "provided per row."
                    )

            num_datapoints_raw = e.get("num_datapoints")
            if num_datapoints_raw is None:
                num_datapoints = np.full((batch_n,), x.shape[1], dtype=np.int32)
            elif isinstance(num_datapoints_raw, torch.Tensor):
                num_datapoints_np = (
                    num_datapoints_raw.detach().to("cpu").numpy().astype(np.int32)
                )
                if num_datapoints_np.ndim == 0:
                    num_datapoints = np.full(
                        (batch_n,), int(num_datapoints_np), dtype=np.int32
                    )
                else:
                    num_datapoints = num_datapoints_np.reshape(-1).astype(np.int32)
            else:
                num_datapoints_np = np.asarray(num_datapoints_raw, dtype=np.int32)
                if num_datapoints_np.ndim == 0:
                    num_datapoints = np.full(
                        (batch_n,), int(num_datapoints_np), dtype=np.int32
                    )
                else:
                    num_datapoints = num_datapoints_np.reshape(-1).astype(np.int32)
            if num_datapoints.shape[0] != batch_n:
                raise ValueError(
                    "num_datapoints length must match batch size when provided per row."
                )
            if np.any(num_datapoints < 1) or np.any(num_datapoints > max_seq_len):
                raise ValueError(
                    "num_datapoints must be within [1, max_seq_len] for every row."
                )
            if np.any(single_eval_pos < 0) or np.any(single_eval_pos >= num_datapoints):
                raise ValueError(
                    "single_eval_pos must satisfy "
                    "0 <= single_eval_pos < num_datapoints."
                )

            # Fast path: avoid redundant padding when tensors already match dump
            # shapes. Fall back to zero-padding for variable-size batches.
            if x.shape[1] == max_seq_len and x.shape[2] == max_features:
                x_padded = x
            else:
                x_padded = np.pad(
                    x,
                    (
                        (0, 0),
                        (0, max_seq_len - x.shape[1]),
                        (0, max_features - x.shape[2]),
                    ),
                    mode="constant",
                )
            if y.shape[1] == max_seq_len:
                y_padded = y
            else:
                y_padded = np.pad(
                    y, ((0, 0), (0, max_seq_len - y.shape[1])), mode="constant"
                )

            dump_X.resize(dump_X.shape[0] + batch_n, axis=0)
            dump_X[-batch_n:] = x_padded

            dump_y.resize(dump_y.shape[0] + batch_n, axis=0)
            dump_y[-batch_n:] = y_padded

            dump_num_features.resize(dump_num_features.shape[0] + batch_n, axis=0)
            dump_num_features[-batch_n:] = x.shape[2]

            dump_num_datapoints.resize(dump_num_datapoints.shape[0] + batch_n, axis=0)
            dump_num_datapoints[-batch_n:] = num_datapoints

            dump_single_eval_pos.resize(dump_single_eval_pos.shape[0] + batch_n, axis=0)
            dump_single_eval_pos[-batch_n:] = single_eval_pos

            for metadata_key, metadata_dtype in optional_metadata_dtypes.items():
                if metadata_key not in e:
                    if metadata_key in optional_metadata_datasets:
                        raise ValueError(
                            f"Optional metadata field {metadata_key!r} is missing in "
                            "a later batch after being present earlier."
                        )
                    continue
                metadata_values = _coerce_batch_vector(
                    e[metadata_key],
                    batch_n=batch_n,
                    dtype=metadata_dtype,
                    key=metadata_key,
                )
                if metadata_key not in optional_metadata_datasets:
                    optional_metadata_datasets[metadata_key] = f.create_dataset(
                        metadata_key,
                        shape=(0,),
                        maxshape=(None,),
                        chunks=(batch_size,),
                        dtype=metadata_dtype,
                    )
                metadata_ds = optional_metadata_datasets[metadata_key]
                metadata_ds.resize(metadata_ds.shape[0] + batch_n, axis=0)
                metadata_ds[-batch_n:] = metadata_values
