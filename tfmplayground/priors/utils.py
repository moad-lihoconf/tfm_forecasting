"""Utility functions for priors."""

import h5py
import numpy as np
import torch
from tqdm import tqdm

from .config import get_tabpfn_prior_config, get_ticl_prior_config


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
            single_eval_pos = e["single_eval_pos"]
            if isinstance(single_eval_pos, torch.Tensor):
                single_eval_pos = single_eval_pos.item()

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

            dump_X.resize(dump_X.shape[0] + batch_size, axis=0)
            dump_X[-batch_size:] = x_padded

            dump_y.resize(dump_y.shape[0] + batch_size, axis=0)
            dump_y[-batch_size:] = y_padded

            dump_num_features.resize(dump_num_features.shape[0] + batch_size, axis=0)
            dump_num_features[-batch_size:] = x.shape[2]

            dump_num_datapoints.resize(
                dump_num_datapoints.shape[0] + batch_size, axis=0
            )
            dump_num_datapoints[-batch_size:] = x.shape[1]

            dump_single_eval_pos.resize(
                dump_single_eval_pos.shape[0] + batch_size, axis=0
            )
            dump_single_eval_pos[-batch_size:] = single_eval_pos
