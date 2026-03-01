from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path
from typing import cast

import torch

import pretrain_regression_dynscm_live as live_train


def test_pretrain_regression_dynscm_live_writes_artifacts(
    monkeypatch, tmp_path: Path
) -> None:
    weights_path = tmp_path / "weights.pth"
    buckets_path = tmp_path / "buckets.pth"
    trace_path = tmp_path / "train_trace.json"
    trace_summary_path = tmp_path / "train_trace_summary.json"
    run_config_path = tmp_path / "run_config.json"
    checkpoint_path = tmp_path / "default_warm_start.pth"

    torch.save(
        {
            "epoch": 0,
            "architecture": {
                "num_layers": 6,
                "embedding_size": 192,
                "num_attention_heads": 6,
                "mlp_hidden_size": 768,
                "num_outputs": 1,
                "dropout": 0.0,
                "feature_normalization": "per_function_zscore",
            },
            "model": {},
            "optimizer": {"state": {}, "param_groups": []},
            "optimizer_name": "adamw",
            "regression_loss": "mse",
            "target_normalization": "none",
        },
        checkpoint_path,
    )

    monkeypatch.setattr(live_train, "get_default_device", lambda: torch.device("cpu"))
    base_profile = live_train.get_research_profile("medium32k_live_baseline")
    monkeypatch.setattr(
        live_train,
        "get_research_profile",
        lambda _name: replace(base_profile, warm_start_checkpoint=str(checkpoint_path)),
    )
    monkeypatch.setattr(
        live_train.NanoTabPFNModel,
        "load_state_dict",
        lambda self, state_dict, strict=True: None,
    )

    captured: dict[str, object] = {}

    def _fake_train(**kwargs):
        captured.update(kwargs)
        run_dir = Path("workdir/live-smoke")
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture": {
                    "num_layers": 6,
                    "embedding_size": 192,
                    "num_attention_heads": 6,
                    "mlp_hidden_size": 768,
                    "num_outputs": 1,
                    "dropout": 0.0,
                    "feature_normalization": "per_function_zscore",
                },
                "model": kwargs["model"].state_dict(),
                "best_epoch": 1,
                "best_metric": 0.004,
            },
            run_dir / "best_checkpoint.pth",
        )
        torch.save(
            {
                "architecture": {
                    "num_layers": 6,
                    "embedding_size": 192,
                    "num_attention_heads": 6,
                    "mlp_hidden_size": 768,
                    "num_outputs": 1,
                    "dropout": 0.0,
                    "feature_normalization": "per_function_zscore",
                },
                "model": kwargs["model"].state_dict(),
                "epoch": 1,
                "optimizer": {},
                "metrics": {"val_loss": 0.004},
            },
            run_dir / "latest_checkpoint.pth",
        )
        Path(kwargs["debug_trace_path"]).write_text(
            json.dumps(
                {
                    "metadata": {},
                    "records": [
                        {
                            "skipped": False,
                            "loss": 0.1,
                            "valid_supervised_targets": 4,
                            "train_target_std": {
                                "min": 0.1,
                                "median": 0.1,
                                "mean": 0.1,
                                "max": 0.1,
                            },
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        return kwargs["model"], {
            "best_epoch": 1,
            "best_metric": 0.004,
            "stop_reason": "completed",
        }

    monkeypatch.setattr(live_train, "train", _fake_train)

    live_train.main(
        [
            "--research_profile",
            "medium32k_live_baseline",
            "--saveweights",
            str(weights_path),
            "--savebuckets",
            str(buckets_path),
            "--debug_train_trace_json",
            str(trace_path),
            "--debug_train_trace_summary_json",
            str(trace_summary_path),
            "--run_config_json",
            str(run_config_path),
            "--epochs",
            "1",
            "--steps",
            "2",
            "--batchsize",
            "2",
            "--dynscm_workers",
            "1",
            "--runname",
            "live-smoke",
        ]
    )

    assert weights_path.exists()
    assert buckets_path.exists()
    assert Path(f"{weights_path}.best.pth").exists()
    assert trace_path.exists()
    assert trace_summary_path.exists()
    assert run_config_path.exists()
    assert cast(live_train.PriorDataLoader, captured["prior"]).num_steps == 2
    assert cast(live_train.PriorDataLoader, captured["val_prior"]).num_steps == 64
    assert captured["ckpt"] is None


def test_pretrain_regression_dynscm_live_warm_start_resets_training_state(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / "warm_start_checkpoint.pth"
    weights_path = tmp_path / "weights.pth"
    buckets_path = tmp_path / "buckets.pth"
    torch.save(
        {
            "epoch": 3,
            "architecture": {
                "num_layers": 6,
                "embedding_size": 192,
                "num_attention_heads": 6,
                "mlp_hidden_size": 768,
                "num_outputs": 1,
                "dropout": 0.0,
                "feature_normalization": "per_function_zscore",
            },
            "model": {},
            "optimizer": {"state": {}, "param_groups": []},
            "optimizer_name": "adamw",
            "regression_loss": "mse",
            "target_normalization": "none",
        },
        checkpoint_path,
    )

    monkeypatch.setattr(live_train, "get_default_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        live_train.NanoTabPFNModel,
        "load_state_dict",
        lambda self, state_dict, strict=True: None,
    )
    captured: dict[str, object] = {}

    def _fake_train(**kwargs):
        captured.update(kwargs)
        run_dir = Path("workdir/live-warm-start")
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture": {
                    "num_layers": 6,
                    "embedding_size": 192,
                    "num_attention_heads": 6,
                    "mlp_hidden_size": 768,
                    "num_outputs": 1,
                    "dropout": 0.0,
                    "feature_normalization": "per_function_zscore",
                },
                "model": kwargs["model"].state_dict(),
            },
            run_dir / "best_checkpoint.pth",
        )
        torch.save(
            {"model": kwargs["model"].state_dict()}, run_dir / "latest_checkpoint.pth"
        )
        return kwargs["model"], {
            "best_epoch": 1,
            "best_metric": 0.1,
            "stop_reason": "completed",
        }

    monkeypatch.setattr(live_train, "train", _fake_train)

    live_train.main(
        [
            "--research_profile",
            "medium32k_live_baseline",
            "--saveweights",
            str(weights_path),
            "--savebuckets",
            str(buckets_path),
            "--loadcheckpoint",
            str(checkpoint_path),
            "--warm_start",
            "--dynscm_workers",
            "1",
            "--runname",
            "live-warm-start",
        ]
    )

    assert captured["ckpt"] is None


def test_pretrain_regression_dynscm_live_defaults_to_profile_warm_start(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / "default_profile_checkpoint.pth"
    weights_path = tmp_path / "weights.pth"
    buckets_path = tmp_path / "buckets.pth"
    torch.save(
        {
            "epoch": 2,
            "architecture": {
                "num_layers": 6,
                "embedding_size": 192,
                "num_attention_heads": 6,
                "mlp_hidden_size": 768,
                "num_outputs": 1,
                "dropout": 0.0,
                "feature_normalization": "per_function_zscore",
            },
            "model": {},
            "optimizer": {"state": {}, "param_groups": []},
            "optimizer_name": "adamw",
            "regression_loss": "mse",
            "target_normalization": "none",
        },
        checkpoint_path,
    )

    monkeypatch.setattr(live_train, "get_default_device", lambda: torch.device("cpu"))
    base_profile = live_train.get_research_profile("medium32k_live_baseline")
    monkeypatch.setattr(
        live_train,
        "get_research_profile",
        lambda _name: replace(base_profile, warm_start_checkpoint=str(checkpoint_path)),
    )
    monkeypatch.setattr(
        live_train.NanoTabPFNModel,
        "load_state_dict",
        lambda self, state_dict, strict=True: None,
    )

    captured: dict[str, object] = {}

    def _fake_train(**kwargs):
        captured.update(kwargs)
        run_dir = Path("workdir/live-default-warm-start")
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture": {
                    "num_layers": 6,
                    "embedding_size": 192,
                    "num_attention_heads": 6,
                    "mlp_hidden_size": 768,
                    "num_outputs": 1,
                    "dropout": 0.0,
                    "feature_normalization": "per_function_zscore",
                },
                "model": kwargs["model"].state_dict(),
            },
            run_dir / "best_checkpoint.pth",
        )
        torch.save(
            {"model": kwargs["model"].state_dict()}, run_dir / "latest_checkpoint.pth"
        )
        return kwargs["model"], {
            "best_epoch": 1,
            "best_metric": 0.1,
            "stop_reason": "completed",
        }

    monkeypatch.setattr(live_train, "train", _fake_train)

    live_train.main(
        [
            "--research_profile",
            "medium32k_live_baseline",
            "--saveweights",
            str(weights_path),
            "--savebuckets",
            str(buckets_path),
            "--dynscm_workers",
            "1",
            "--runname",
            "live-default-warm-start",
        ]
    )

    assert captured["ckpt"] is None


def test_pretrain_regression_dynscm_live_uses_profile_target_std_floor_and_allows_warm_start_norm_mismatch(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / "norm_mismatch_checkpoint.pth"
    weights_path = tmp_path / "weights.pth"
    buckets_path = tmp_path / "buckets.pth"
    torch.save(
        {
            "epoch": 2,
            "architecture": {
                "num_layers": 6,
                "embedding_size": 192,
                "num_attention_heads": 6,
                "mlp_hidden_size": 768,
                "num_outputs": 1,
                "dropout": 0.0,
                "feature_normalization": "per_function_zscore",
            },
            "model": {},
            "optimizer": {"state": {}, "param_groups": []},
            "optimizer_name": "adamw",
            "regression_loss": "mse",
            "target_normalization": "none",
        },
        checkpoint_path,
    )

    monkeypatch.setattr(live_train, "get_default_device", lambda: torch.device("cpu"))
    base_profile = live_train.get_research_profile("medium32k_live_baseline")
    monkeypatch.setattr(
        live_train,
        "get_research_profile",
        lambda _name: replace(
            base_profile,
            warm_start_checkpoint=str(checkpoint_path),
            target_normalization="per_function_clamped",
            target_std_floor=5e-2,
        ),
    )
    monkeypatch.setattr(
        live_train.NanoTabPFNModel,
        "load_state_dict",
        lambda self, state_dict, strict=True: None,
    )

    captured: dict[str, object] = {}

    def _fake_train(**kwargs):
        captured.update(kwargs)
        run_dir = Path("workdir/live-norm-mismatch")
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture": {
                    "num_layers": 6,
                    "embedding_size": 192,
                    "num_attention_heads": 6,
                    "mlp_hidden_size": 768,
                    "num_outputs": 1,
                    "dropout": 0.0,
                    "feature_normalization": "per_function_zscore",
                },
                "model": kwargs["model"].state_dict(),
            },
            run_dir / "best_checkpoint.pth",
        )
        torch.save(
            {
                "architecture": {
                    "num_layers": 6,
                    "embedding_size": 192,
                    "num_attention_heads": 6,
                    "mlp_hidden_size": 768,
                    "num_outputs": 1,
                    "dropout": 0.0,
                    "feature_normalization": "per_function_zscore",
                },
                "model": kwargs["model"].state_dict(),
            },
            run_dir / "latest_checkpoint.pth",
        )
        return kwargs["model"], {
            "best_epoch": 1,
            "best_metric": 0.1,
            "stop_reason": "completed",
        }

    monkeypatch.setattr(live_train, "train", _fake_train)

    live_train.main(
        [
            "--research_profile",
            "medium32k_live_baseline",
            "--saveweights",
            str(weights_path),
            "--savebuckets",
            str(buckets_path),
            "--dynscm_workers",
            "1",
            "--runname",
            "live-norm-mismatch",
        ]
    )

    assert captured["ckpt"] is None
    assert captured["target_normalization"] == "per_function_clamped"
    assert captured["target_std_floor"] == 5e-2


def test_pretrain_regression_dynscm_live_no_warm_start_skips_profile_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    weights_path = tmp_path / "weights.pth"
    buckets_path = tmp_path / "buckets.pth"

    monkeypatch.setattr(live_train, "get_default_device", lambda: torch.device("cpu"))
    captured: dict[str, object] = {}

    def _fake_train(**kwargs):
        captured.update(kwargs)
        run_dir = Path("workdir/live-no-warm-start")
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture": {
                    "num_layers": 6,
                    "embedding_size": 192,
                    "num_attention_heads": 6,
                    "mlp_hidden_size": 768,
                    "num_outputs": 1,
                    "dropout": 0.0,
                    "feature_normalization": "per_function_zscore",
                },
                "model": kwargs["model"].state_dict(),
            },
            run_dir / "best_checkpoint.pth",
        )
        torch.save(
            {"model": kwargs["model"].state_dict()}, run_dir / "latest_checkpoint.pth"
        )
        return kwargs["model"], {
            "best_epoch": 1,
            "best_metric": 0.1,
            "stop_reason": "completed",
        }

    monkeypatch.setattr(live_train, "train", _fake_train)

    live_train.main(
        [
            "--research_profile",
            "medium32k_live_baseline",
            "--no-warm_start",
            "--saveweights",
            str(weights_path),
            "--savebuckets",
            str(buckets_path),
            "--dynscm_workers",
            "1",
            "--runname",
            "live-no-warm-start",
        ]
    )

    assert captured["ckpt"] is None


def test_pretrain_regression_dynscm_live_cli_debug_output_clamp_overrides_checkpoint(
    monkeypatch, tmp_path: Path
) -> None:
    checkpoint_path = tmp_path / "clamp_checkpoint.pth"
    weights_path = tmp_path / "weights.pth"
    buckets_path = tmp_path / "buckets.pth"
    torch.save(
        {
            "epoch": 1,
            "architecture": {
                "num_layers": 6,
                "embedding_size": 192,
                "num_attention_heads": 6,
                "mlp_hidden_size": 768,
                "num_outputs": 1,
                "dropout": 0.0,
                "feature_normalization": "per_function_zscore",
                "debug_output_clamp": 0.2,
            },
            "model": {},
            "optimizer": {"state": {}, "param_groups": []},
            "optimizer_name": "adamw",
            "regression_loss": "mse",
            "target_normalization": "none",
        },
        checkpoint_path,
    )

    monkeypatch.setattr(live_train, "get_default_device", lambda: torch.device("cpu"))
    monkeypatch.setattr(
        live_train.NanoTabPFNModel,
        "load_state_dict",
        lambda self, state_dict, strict=True: None,
    )

    captured: dict[str, object] = {}

    def _fake_train(**kwargs):
        captured.update(kwargs)
        run_dir = Path("workdir/live-clamp-override")
        run_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "architecture": {
                    "num_layers": 6,
                    "embedding_size": 192,
                    "num_attention_heads": 6,
                    "mlp_hidden_size": 768,
                    "num_outputs": 1,
                    "dropout": 0.0,
                    "feature_normalization": "per_function_zscore",
                    "debug_output_clamp": getattr(
                        kwargs["model"], "debug_output_clamp", None
                    ),
                },
                "model": kwargs["model"].state_dict(),
            },
            run_dir / "best_checkpoint.pth",
        )
        torch.save(
            {"model": kwargs["model"].state_dict()}, run_dir / "latest_checkpoint.pth"
        )
        return kwargs["model"], {
            "best_epoch": 1,
            "best_metric": 0.1,
            "stop_reason": "completed",
        }

    monkeypatch.setattr(live_train, "train", _fake_train)

    live_train.main(
        [
            "--research_profile",
            "medium32k_live_baseline",
            "--saveweights",
            str(weights_path),
            "--savebuckets",
            str(buckets_path),
            "--loadcheckpoint",
            str(checkpoint_path),
            "--warm_start",
            "--debug_output_clamp",
            "-1",
            "--dynscm_workers",
            "1",
            "--runname",
            "live-clamp-override",
        ]
    )

    model = cast(live_train.NanoTabPFNModel, captured["model"])
    assert model.debug_output_clamp is None
