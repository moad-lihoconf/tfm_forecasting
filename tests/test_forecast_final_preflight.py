from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest


def _load_script_module(script_name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


preflight_mod = _load_script_module("check_forecast_final_ready.py")


class _FakeCfg:
    class _Datasets:
        dataset_names = preflight_mod.FINAL_DATASETS

    class _Models:
        enabled_regression_models = preflight_mod.FINAL_MODELS
        nicl_regression_mode = "quantized_proxy"
        nicl_regression_endpoint = "https://example.com/predict"
        nicl_api_key_env = "NEURALK_API_KEY"
        nicl_max_rows_budget = 100
        model_dynscm_ckpt = "dyn.pth"
        model_dynscm_dist = "dyn_dist.pth"

    class _Protocol:
        context_rows = 32
        test_rows = 16
        horizons = (1, 3, 6, 12)

    datasets = _Datasets()
    models = _Models()
    protocol = _Protocol()


class _FakeCfgBadModels(_FakeCfg):
    class _Models(_FakeCfg._Models):
        enabled_regression_models = (
            "nanotabpfn_standard",
            "nanotabpfn_dynscm",
            "tabicl_regressor",
        )

    models = _Models()


def test_check_config_accepts_final_contract():
    detail = preflight_mod._check_config(_FakeCfg())
    assert "final benchmark contract" in detail


def test_check_config_rejects_wrong_model_set():
    with pytest.raises(ValueError, match="enabled_regression_models"):
        preflight_mod._check_config(_FakeCfgBadModels())


def test_check_nicl_accepts_token_from_dotenv(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.delenv("NEURALK_API_KEY", raising=False)
    detail = preflight_mod._check_nicl(
        _FakeCfg(),
        {"NEURALK_API_KEY": "secret"},
    )
    assert "token found" in detail


def test_main_exits_success_when_all_checks_pass(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        preflight_mod.ForecastBenchmarkConfig, "from_json", lambda path: _FakeCfg()
    )
    monkeypatch.setattr(preflight_mod, "_check_config", lambda cfg: "ok")
    monkeypatch.setattr(preflight_mod, "_check_datasets", lambda cfg: "ok")
    monkeypatch.setattr(preflight_mod, "_check_standard_model", lambda: "ok")
    monkeypatch.setattr(preflight_mod, "_check_dynscm_model", lambda cfg: "ok")
    monkeypatch.setattr(preflight_mod, "_check_nicl", lambda cfg, dotenv: "ok")
    monkeypatch.setattr("sys.argv", ["check", "--config", str(cfg_path)])

    preflight_mod.main()
    captured = capsys.readouterr()
    assert captured.out.splitlines()[0] == "PASS"


def test_main_exits_nonzero_on_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys
):
    cfg_path = tmp_path / "cfg.json"
    cfg_path.write_text("{}", encoding="utf-8")

    monkeypatch.setattr(
        preflight_mod.ForecastBenchmarkConfig, "from_json", lambda path: _FakeCfg()
    )
    monkeypatch.setattr(
        preflight_mod,
        "_check_config",
        lambda cfg: (_ for _ in ()).throw(ValueError("bad config")),
    )
    monkeypatch.setattr(preflight_mod, "_check_datasets", lambda cfg: "ok")
    monkeypatch.setattr(preflight_mod, "_check_standard_model", lambda: "ok")
    monkeypatch.setattr(preflight_mod, "_check_dynscm_model", lambda cfg: "ok")
    monkeypatch.setattr(preflight_mod, "_check_nicl", lambda cfg, dotenv: "ok")
    monkeypatch.setattr("sys.argv", ["check", "--config", str(cfg_path)])

    with pytest.raises(SystemExit, match="1"):
        preflight_mod.main()
    captured = capsys.readouterr()
    assert captured.out.splitlines()[0] == "FAIL"
    assert "bad config" in captured.out
