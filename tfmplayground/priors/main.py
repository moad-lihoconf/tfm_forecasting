"""Main module for the priors package."""

import argparse
import json
import random

import numpy as np
import torch

from .dataloader import (
    DynSCMPriorDataLoader,
    TabICLPriorDataLoader,
    TICLPriorDataLoader,
)
from .dynscm import DynSCMConfig
from .utils import build_tabpfn_prior, build_ticl_prior, dump_prior_to_h5

try:
    from .dataloader import TabPFNPriorDataLoader  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - legacy optional integration.
    TabPFNPriorDataLoader = None


def _parse_override_value(raw_value: str) -> object:
    lower = raw_value.lower()
    if lower == "true":
        return True
    if lower == "false":
        return False
    if lower == "null":
        return None
    try:
        return json.loads(raw_value)
    except json.JSONDecodeError:
        return raw_value


def _parse_dynscm_overrides(raw_overrides: list[str]) -> dict[str, object]:
    """Parse `key=value` CLI overrides into flat DynSCM config fields.

    Dotted keys such as `shape.num_variables_min=8` are accepted and reduced to
    their leaf field name (`num_variables_min`), matching DynSCM flat overrides.
    """
    overrides: dict[str, object] = {}
    for entry in raw_overrides:
        if "=" not in entry:
            raise ValueError(
                f"Each --dynscm_override must be in `key=value` format, got {entry!r}."
            )
        raw_key, raw_value = entry.split("=", maxsplit=1)
        key = raw_key.strip()
        if not key:
            raise ValueError(f"Invalid empty override key in {entry!r}.")
        leaf_key = key.rsplit(".", maxsplit=1)[-1]
        overrides[leaf_key] = _parse_override_value(raw_value.strip())
    return overrides


def _load_dynscm_config(
    config_json: str | None,
    raw_overrides: list[str],
) -> DynSCMConfig:
    cfg = DynSCMConfig.from_json(config_json) if config_json else DynSCMConfig()
    overrides = _parse_dynscm_overrides(raw_overrides)
    return cfg.with_overrides(**overrides) if overrides else cfg


def main():
    parser = argparse.ArgumentParser(
        description="Dump prior data (TICL, TabICL, TabPFN, or DynSCM) into HDF5."
    )
    parser.add_argument(
        "--lib",
        type=str,
        required=True,
        choices=["ticl", "tabicl", "tabpfn", "dynscm"],
        help="Which library to use for the prior.",
    )
    parser.add_argument(
        "--save_path", type=str, required=False, help="Path to save the HDF5 file."
    )
    parser.add_argument(
        "--num_batches", type=int, default=100, help="Number of batches to dump."
    )
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size for dumping."
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cpu", "cuda"],
        help="Device to run prior sampling on.",
    )
    parser.add_argument(
        "--prior_type",
        type=str,
        required=False,
        default=None,
        help=(
            "Type of prior to use. "
            "For TICL: mlp, gp, classification_adapter, "
            "boolean_conjunctions, step_function. "
            "For TabICL: mlp_scm, tree_scm, mix_scm, dummy. "
            "For TabPFN: mlp, gp, prior_bag."
        ),
    )
    parser.add_argument(
        "--base_prior",
        type=str,
        default="mlp",
        choices=["mlp", "gp"],
        help="Base regression prior for composite priors like classification_adapter.",
    )
    parser.add_argument(
        "--min_features", type=int, default=1, help="Minimum number of input features."
    )
    parser.add_argument(
        "--max_features",
        type=int,
        default=100,
        help="Maximum number of input features.",
    )
    parser.add_argument(
        "--min_seq_len",
        type=int,
        default=None,
        help="Minimum number of data points per function.",
    )
    parser.add_argument(
        "--max_seq_len",
        type=int,
        default=1024,
        help="Maximum number of data points per function.",
    )
    parser.add_argument(
        "--min_eval_pos",
        type=int,
        default=10,
        help="Minimum evaluation position in the sequence.",
    )
    parser.add_argument(
        "--max_classes",
        type=int,
        default=0,
        help=(
            "Maximum number of classes. Set to 0 for regression, >0 for classification."
        ),
    )
    parser.add_argument(
        "--np_seed", type=int, default=None, help="Random seed for NumPy."
    )
    parser.add_argument(
        "--torch_seed", type=int, default=None, help="Random seed for PyTorch."
    )
    parser.add_argument(
        "--dynscm_config_json",
        type=str,
        default=None,
        help=("Optional path to a DynSCM JSON config. Used only when --lib dynscm."),
    )
    parser.add_argument(
        "--dynscm_override",
        action="append",
        default=[],
        help=(
            "DynSCM override in key=value format. Can be repeated. "
            "Example: --dynscm_override num_variables_min=6 "
            "--dynscm_override features.num_kernels=2"
        ),
    )
    parser.add_argument(
        "--dynscm_seed",
        type=int,
        default=None,
        help=(
            "Optional dedicated seed for DynSCM batch sampling. "
            "Defaults to --np_seed when omitted."
        ),
    )
    parser.add_argument(
        "--dynscm_workers",
        type=int,
        default=1,
        help="Number of process workers for DynSCM batch generation.",
    )
    parser.add_argument(
        "--dynscm_worker_blas_threads",
        type=int,
        default=1,
        help="BLAS thread cap applied inside each DynSCM worker process.",
    )
    parser.add_argument(
        "--dynscm_compute_spectral_diagnostics",
        dest="dynscm_compute_spectral_diagnostics",
        action="store_true",
        default=None,
        help=(
            "Compute spectral-radius diagnostics even when spectral rescaling "
            "is disabled."
        ),
    )
    parser.add_argument(
        "--no_dynscm_compute_spectral_diagnostics",
        dest="dynscm_compute_spectral_diagnostics",
        action="store_false",
        help="Disable optional spectral-radius diagnostics for DynSCM.",
    )

    args = parser.parse_args()

    if args.np_seed is not None:
        np.random.seed(args.np_seed)
    if args.torch_seed is not None:
        torch.manual_seed(args.torch_seed)
        random.seed(args.torch_seed)

    if args.lib in {"ticl", "tabicl", "tabpfn"} and args.prior_type is None:
        parser.error("--prior_type is required for ticl/tabicl/tabpfn.")

    if args.lib == "dynscm" and args.max_classes != 0:
        parser.error("DynSCM currently supports regression only; use --max_classes 0.")
    if args.dynscm_workers < 1:
        parser.error("--dynscm_workers must be >= 1.")
    if args.dynscm_worker_blas_threads < 1:
        parser.error("--dynscm_worker_blas_threads must be >= 1.")

    device = torch.device(args.device)
    resolved_prior_type = args.prior_type if args.prior_type is not None else args.lib

    if args.save_path is None:
        args.save_path = (
            f"prior_{args.lib}_{resolved_prior_type}_{args.num_batches}"
            f"x{args.batch_size}_{args.max_seq_len}x{args.max_features}.h5"
        )

    # infer the problem_type from max_classes
    problem_type = "classification" if args.max_classes > 0 else "regression"

    if args.lib == "ticl":
        prior = TICLPriorDataLoader(
            prior=build_ticl_prior(args.prior_type, args.base_prior, args.max_classes),
            num_steps=args.num_batches,
            batch_size=args.batch_size,
            num_datapoints_max=args.max_seq_len,
            num_features=args.max_features,
            device=device,
            min_eval_pos=args.min_eval_pos,
        )
    elif args.lib == "tabpfn":
        if TabPFNPriorDataLoader is None:
            raise RuntimeError(
                "TabPFNPriorDataLoader is unavailable in this installation."
            )
        tabpfn_config = build_tabpfn_prior(args.prior_type, args.max_classes)
        prior = TabPFNPriorDataLoader(
            prior_type=args.prior_type,
            num_steps=args.num_batches,
            batch_size=args.batch_size,
            num_datapoints_max=args.max_seq_len,
            num_features=args.max_features,
            device=device,
            **tabpfn_config,
        )
    elif args.lib == "dynscm":
        try:
            dynscm_cfg = _load_dynscm_config(
                args.dynscm_config_json,
                args.dynscm_override,
            )
        except ValueError as exc:
            parser.error(f"Invalid DynSCM configuration: {exc}")
        if args.dynscm_compute_spectral_diagnostics is not None:
            dynscm_cfg = dynscm_cfg.with_overrides(
                compute_spectral_diagnostics=args.dynscm_compute_spectral_diagnostics
            )

        dynscm_seed = args.dynscm_seed
        if dynscm_seed is None:
            dynscm_seed = args.np_seed

        prior = DynSCMPriorDataLoader(
            cfg=dynscm_cfg,
            num_steps=args.num_batches,
            batch_size=args.batch_size,
            num_datapoints_max=args.max_seq_len,
            num_features=args.max_features,
            device=device,
            seed=dynscm_seed,
            workers=args.dynscm_workers,
            worker_blas_threads=args.dynscm_worker_blas_threads,
        )
    else:  # tabicl
        prior = TabICLPriorDataLoader(
            num_steps=args.num_batches,
            batch_size=args.batch_size,
            num_datapoints_min=args.min_seq_len,
            num_datapoints_max=args.max_seq_len,
            min_features=args.min_features,
            max_features=args.max_features,
            max_num_classes=args.max_classes,
            prior_type=args.prior_type,
            device=device,
        )

    dump_prior_to_h5(
        prior,
        args.max_classes,
        args.batch_size,
        args.save_path,
        problem_type,
        args.max_seq_len,
        args.max_features,
    )
