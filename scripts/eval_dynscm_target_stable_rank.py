from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Literal, TypedDict, cast

import torch

import pretrain_regression_dynscm_live as live_train
from tfmplayground.model import NanoTabPFNModel
from tfmplayground.priors.dynscm.research_profiles import (
    LiveSourceSpec,
    get_research_profile,
)
from tfmplayground.utils import get_default_device

FeatureNormalization = Literal["per_function_zscore", "none"]


class SuiteRow(TypedDict):
    profile: str
    run_name: str
    target_eval_loss: float
    stable_eval_loss: float
    target_skipped_fraction: float
    stable_skipped_fraction: float
    _pass: str


class DecisionRow(TypedDict):
    profile: str
    run_name: str
    target_eval_loss: float
    stable_eval_loss: float
    target_skipped_fraction: float
    stable_skipped_fraction: float
    rank: int
    decision: str
    stable_degradation: float
    eval_pass: str


def _load_eval_module():
    script_path = Path(__file__).resolve().with_name("eval_dynscm_synthetic_suite.py")
    spec = importlib.util.spec_from_file_location(
        "_eval_dynscm_synthetic_suite", script_path
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


eval_mod = _load_eval_module()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--dynscm_workers", type=int, default=4)
    parser.add_argument("--dynscm_worker_blas_threads", type=int, default=1)
    parser.add_argument(
        "--fast_eval_steps",
        type=int,
        default=16,
        help="Fast ranking pass step count for stable_eval and target_eval.",
    )
    parser.add_argument(
        "--full_eval_steps",
        type=int,
        default=64,
        help="Decision-grade rerun step count for the top-K runs.",
    )
    parser.add_argument(
        "--full_top_k",
        type=int,
        default=2,
        help="How many fast-pass winners to rerun at full_eval_steps.",
    )
    parser.add_argument(
        "--ranking_path",
        type=str,
        default="workdir/research/dynscm-research-20260301-024644.ranking.json",
        help="Where to write the final ranking artifact.",
    )
    parser.add_argument(
        "--run",
        action="append",
        nargs=2,
        metavar=("PROFILE", "RUN_NAME"),
        required=True,
        help="Profile name and run name pair. May be provided multiple times.",
    )
    return parser


def _feature_normalization(value: object) -> FeatureNormalization:
    if value not in {"per_function_zscore", "none"}:
        raise ValueError(f"Unsupported feature normalization: {value!r}.")
    return cast(FeatureNormalization, value)


def _load_model(checkpoint_path: Path, device: torch.device) -> NanoTabPFNModel:
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    architecture = checkpoint["architecture"]
    model = NanoTabPFNModel(
        num_attention_heads=int(architecture["num_attention_heads"]),
        embedding_size=int(architecture["embedding_size"]),
        mlp_hidden_size=int(architecture["mlp_hidden_size"]),
        num_layers=int(architecture["num_layers"]),
        num_outputs=int(architecture["num_outputs"]),
        dropout=float(architecture.get("dropout", 0.0)),
        feature_normalization=_feature_normalization(
            architecture.get("feature_normalization", "per_function_zscore")
        ),
        debug_output_clamp=architecture.get("debug_output_clamp"),
    )
    model.load_state_dict(checkpoint["model"])
    return model.to(device)


def _evaluate_run(
    *,
    profile_name: str,
    run_name: str,
    eval_steps: int,
    device: torch.device,
    dynscm_workers: int,
    dynscm_worker_blas_threads: int,
) -> dict[str, object]:
    profile = get_research_profile(profile_name)
    checkpoint_path = Path(
        f"workdir/research/checkpoints/{run_name}.best_checkpoint.pth"
    )
    model = _load_model(checkpoint_path, device)

    payload_suites: dict[str, dict[str, float | int]] = {}
    payload: dict[str, object] = {
        "checkpoint_path": str(checkpoint_path),
        "research_profile": profile.name,
        "device": str(device),
        "eval_steps": int(eval_steps),
        "suites": payload_suites,
    }
    for suite in profile.eval_suites:
        if suite.name not in {"stable_eval", "target_eval"}:
            continue
        loader = live_train._build_prior_loader(
            source=LiveSourceSpec(kind="single", cfg=suite.cfg),
            num_steps=int(eval_steps),
            batch_size=profile.training_budget.batch_size,
            num_datapoints_max=profile.max_seq_len,
            num_features=profile.max_features,
            device=device,
            seed=suite.seed,
            workers=int(dynscm_workers),
            worker_blas_threads=int(dynscm_worker_blas_threads),
            total_train_batches=max(1, int(eval_steps)),
        )
        metrics = eval_mod._suite_metrics(model=model, loader=loader, device=device)
        metrics["seed"] = suite.seed
        metrics["steps"] = int(eval_steps)
        payload_suites[suite.name] = metrics
        close_fn = getattr(getattr(loader, "get_batch_function", None), "close", None)
        if callable(close_fn):
            close_fn()
    return payload


def _row_from_payload(
    *,
    profile_name: str,
    run_name: str,
    payload: dict[str, object],
    pass_name: str,
) -> SuiteRow:
    suites = cast(dict[str, dict[str, float | int]], payload["suites"])
    target_suite = suites["target_eval"]
    stable_suite = suites["stable_eval"]
    return {
        "profile": profile_name,
        "run_name": run_name,
        "target_eval_loss": float(target_suite["loss"]),
        "stable_eval_loss": float(stable_suite["loss"]),
        "target_skipped_fraction": float(target_suite["skipped_fraction"]),
        "stable_skipped_fraction": float(stable_suite["skipped_fraction"]),
        "_pass": pass_name,
    }


def _stable_reference(rows: list[SuiteRow]) -> float:
    candidates = [
        row["stable_eval_loss"]
        for row in rows
        if row["stable_skipped_fraction"] <= 0.05
        and torch.isfinite(torch.tensor(row["stable_eval_loss"])).item()
    ]
    return min(candidates) if candidates else float("nan")


def _decision_rows(rows: list[SuiteRow]) -> list[DecisionRow]:
    stable_ref = _stable_reference(rows)
    ranked = sorted(
        rows,
        key=lambda row: (row["target_eval_loss"], row["stable_eval_loss"]),
    )
    decisions: list[DecisionRow] = []
    accept_count = 0
    for row in ranked:
        stable_deg = float("inf")
        if stable_ref > 0.0 and torch.isfinite(torch.tensor(stable_ref)).item():
            stable_deg = (row["stable_eval_loss"] / stable_ref) - 1.0
        rejected = (
            not torch.isfinite(torch.tensor(row["target_eval_loss"])).item()
            or not torch.isfinite(torch.tensor(row["stable_eval_loss"])).item()
            or row["target_skipped_fraction"] > 0.05
            or row["stable_skipped_fraction"] > 0.05
            or stable_deg > 0.03
        )
        decision = "reject"
        if not rejected:
            accept_count += 1
            decision = "promote" if accept_count <= 2 else "hold"
        decisions.append(
            {
                "profile": row["profile"],
                "run_name": row["run_name"],
                "target_eval_loss": row["target_eval_loss"],
                "stable_eval_loss": row["stable_eval_loss"],
                "target_skipped_fraction": row["target_skipped_fraction"],
                "stable_skipped_fraction": row["stable_skipped_fraction"],
                "rank": len(decisions) + 1,
                "decision": decision,
                "stable_degradation": stable_deg,
                "eval_pass": row["_pass"],
            }
        )
    return decisions


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    device = torch.device(args.device or get_default_device())
    fast_rows: list[SuiteRow] = []

    for profile_name, run_name in args.run:
        fast_payload = _evaluate_run(
            profile_name=profile_name,
            run_name=run_name,
            eval_steps=int(args.fast_eval_steps),
            device=device,
            dynscm_workers=int(args.dynscm_workers),
            dynscm_worker_blas_threads=int(args.dynscm_worker_blas_threads),
        )
        fast_output_path = Path(
            f"workdir/research/{run_name}.stable_target_eval.fast.json"
        )
        fast_output_path.parent.mkdir(parents=True, exist_ok=True)
        fast_output_path.write_text(
            json.dumps(fast_payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        row = _row_from_payload(
            profile_name=profile_name,
            run_name=run_name,
            payload=fast_payload,
            pass_name="fast",
        )
        fast_rows.append(row)
        print(json.dumps(row, sort_keys=True), flush=True)

    ranked_fast = sorted(
        fast_rows,
        key=lambda row: (row["target_eval_loss"], row["stable_eval_loss"]),
    )
    top_k_names = {
        row["run_name"] for row in ranked_fast[: max(0, int(args.full_top_k))]
    }
    final_rows: list[SuiteRow] = []
    for row in ranked_fast:
        if row["run_name"] in top_k_names:
            full_payload = _evaluate_run(
                profile_name=row["profile"],
                run_name=row["run_name"],
                eval_steps=int(args.full_eval_steps),
                device=device,
                dynscm_workers=int(args.dynscm_workers),
                dynscm_worker_blas_threads=int(args.dynscm_worker_blas_threads),
            )
            full_output_path = Path(
                f"workdir/research/{row['run_name']}.stable_target_eval.full.json"
            )
            full_output_path.parent.mkdir(parents=True, exist_ok=True)
            full_output_path.write_text(
                json.dumps(full_payload, indent=2, sort_keys=True),
                encoding="utf-8",
            )
            final_rows.append(
                _row_from_payload(
                    profile_name=row["profile"],
                    run_name=row["run_name"],
                    payload=full_payload,
                    pass_name="full",
                )
            )
        else:
            final_rows.append(row)

    ranking: list[DecisionRow] = _decision_rows(final_rows)
    ranking_path = Path(args.ranking_path)
    ranking_path.parent.mkdir(parents=True, exist_ok=True)
    ranking_path.write_text(
        json.dumps(ranking, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    print("RANKING")
    for decision_row in ranking:
        print(
            f"{decision_row['rank']}. {decision_row['profile']} "
            f"target={decision_row['target_eval_loss']:.6f} "
            f"stable={decision_row['stable_eval_loss']:.6f} "
            f"decision={decision_row['decision']}"
        )


if __name__ == "__main__":
    main()
