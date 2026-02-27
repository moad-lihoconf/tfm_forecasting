from __future__ import annotations

from pathlib import Path

from tfmplayground.benchmarks.forecasting.config import ForecastBenchmarkConfig
from tfmplayground.benchmarks.forecasting.runner import BenchmarkArtifacts


def test_cli_load_config_applies_overrides(tmp_path: Path):
    from tfmplayground.benchmarks.forecasting import cli

    args = cli._build_parser().parse_args(
        [
            "--mode",
            "regression",
            "--enabled_regression_models",
            "nanotabpfn_standard",
            "nanotabpfn_dynscm",
            "nicl_regression",
            "--model_standard_ckpt",
            "std.pth",
            "--model_dynscm_ckpt",
            "dyn.pth",
            "--tabicl_checkpoint_version",
            "tabicl-custom.ckpt",
            "--output_dir",
            str(tmp_path / "out"),
        ]
    )

    cfg = cli._load_config(args)
    assert isinstance(cfg, ForecastBenchmarkConfig)
    assert cfg.mode == "regression"
    assert cfg.models.enabled_regression_models == (
        "nanotabpfn_standard",
        "nanotabpfn_dynscm",
        "nicl_regression",
    )
    assert cfg.models.model_standard_ckpt == "std.pth"
    assert cfg.models.model_dynscm_ckpt == "dyn.pth"
    assert cfg.models.tabicl_checkpoint_version == "tabicl-custom.ckpt"
    assert str(cfg.output_dir).endswith("out")


def test_cli_main_invokes_runner(monkeypatch, tmp_path: Path, capsys):
    from tfmplayground.benchmarks.forecasting import cli

    out_dir = tmp_path / "results"

    def _fake_run(cfg, device="cpu"):
        out_dir.mkdir(parents=True, exist_ok=True)
        report = out_dir / "report.md"
        report.write_text("ok", encoding="utf-8")
        return BenchmarkArtifacts(
            regression_rows_path=out_dir / "regression_rows.csv",
            regression_summary_path=out_dir / "regression_summary.json",
            proxy_rows_path=None,
            proxy_summary_path=None,
            report_path=report,
            regression_rows=None,
            regression_summary=None,
            proxy_rows=None,
            proxy_summary=None,
        )

    monkeypatch.setattr(cli, "run_benchmark", _fake_run)
    monkeypatch.setattr(
        "sys.argv",
        [
            "forecast-cli",
            "--mode",
            "regression",
            "--output_dir",
            str(out_dir),
        ],
    )

    cli.main()
    captured = capsys.readouterr()
    assert "Report:" in captured.out
