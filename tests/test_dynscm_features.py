"""Concise tests for DynSCM phase-7 forecast featurization."""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


def _load_module(fullname: str, filepath: Path):
    spec = importlib.util.spec_from_file_location(fullname, filepath)
    if spec is None or spec.loader is None:
        raise RuntimeError(
            f"Could not create module spec for {fullname} from {filepath}"
        )
    module = importlib.util.module_from_spec(spec)
    sys.modules[fullname] = module
    spec.loader.exec_module(module)
    return module


def _load_dynscm_api():
    repo_root = Path(__file__).resolve().parents[1]
    dyn_dir = repo_root / "tfmplayground" / "priors" / "dynscm"

    for pkg_name, pkg_path in (
        ("tfmplayground", repo_root / "tfmplayground"),
        ("tfmplayground.priors", repo_root / "tfmplayground" / "priors"),
        ("tfmplayground.priors.dynscm", dyn_dir),
    ):
        pkg = types.ModuleType(pkg_name)
        pkg.__path__ = [str(pkg_path)]
        sys.modules[pkg_name] = pkg

    config_mod = _load_module(
        "tfmplayground.priors.dynscm.config", dyn_dir / "config.py"
    )
    features_mod = _load_module(
        "tfmplayground.priors.dynscm.features", dyn_dir / "features.py"
    )
    missing_mod = _load_module(
        "tfmplayground.priors.dynscm.missingness", dyn_dir / "missingness.py"
    )
    return config_mod, features_mod, missing_mod


@pytest.fixture(scope="module")
def dynscm_api():
    return _load_dynscm_api()


def test_sample_origins_and_horizons_enforces_split_and_index_safety(dynscm_api):
    config_mod, features_mod, _ = dynscm_api
    cfg = config_mod.DynSCMConfig()

    t_idx, h_idx = features_mod.sample_origins_and_horizons(
        cfg,
        batch_size=6,
        num_steps=220,
        n_train=18,
        n_test=9,
        seed=5,
    )

    assert t_idx.shape == (6, 27)
    assert h_idx.shape == (6, 27)
    assert t_idx.dtype == np.int64
    assert h_idx.dtype == np.int64

    required_lag = max(cfg.max_feature_lag, max(cfg.explicit_lags))
    assert np.all(t_idx >= required_lag)
    assert np.all(t_idx + h_idx < 220)
    assert np.all(np.max(t_idx[:, :18], axis=1) < np.min(t_idx[:, 18:], axis=1))


def test_build_forecasting_table_extracts_exact_labels_and_metadata(dynscm_api):
    config_mod, features_mod, _ = dynscm_api
    cfg = config_mod.DynSCMConfig()

    batch_size, num_steps, num_vars = 4, 210, 5
    series = np.random.default_rng(11).normal(size=(batch_size, num_steps, num_vars))
    y_idx = np.array([0, 1, 2, 3], dtype=np.int64)

    x, y, meta = features_mod.build_forecasting_table(
        cfg,
        series,
        y_idx,
        n_train=16,
        n_test=8,
        seed=7,
    )

    t_idx = meta["t_idx"]
    h_idx = meta["h_idx"]
    label_time = t_idx + h_idx
    expected_y = series[
        np.arange(batch_size, dtype=np.int64)[:, None],
        label_time,
        y_idx[:, None],
    ]

    assert x.shape[:2] == (batch_size, 24)
    assert y.shape == (batch_size, 24)
    assert np.allclose(y, expected_y)
    assert meta["single_eval_pos"] == 16
    assert meta["F_used"] == x.shape[2]
    assert "feature_slices" in meta
    assert np.isfinite(x).all()
    assert np.isfinite(y).all()


def test_missingness_aware_imputation_and_mask_channels(dynscm_api):
    config_mod, features_mod, missing_mod = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "missing_mode": "mix",
            "missing_rate_min": 0.45,
            "missing_rate_max": 0.45,
            "block_missing_prob": 1.0,
            "block_len_min": 5,
            "block_len_max": 5,
            "add_mask_channels": True,
            "num_kernels": 2,
        }
    )

    batch_size, num_steps, num_vars = 3, 200, 4
    series = np.random.default_rng(13).normal(size=(batch_size, num_steps, num_vars))
    y_idx = np.array([0, 1, 2], dtype=np.int64)
    t_idx, h_idx = features_mod.sample_origins_and_horizons(
        cfg,
        batch_size=batch_size,
        num_steps=num_steps,
        n_train=14,
        n_test=6,
        seed=19,
    )
    obs_mask = missing_mod.sample_observation_mask(
        cfg,
        series,
        seed=23,
        label_times=t_idx + h_idx,
        label_var_indices=y_idx,
    )

    x, y, meta = features_mod.build_forecasting_table(
        cfg,
        series,
        y_idx,
        n_train=14,
        n_test=6,
        t_idx=t_idx,
        h_idx=h_idx,
        obs_mask=obs_mask,
        seed=31,
    )

    assert np.isfinite(x).all()
    assert np.isfinite(y).all()
    assert "data_mask" in meta["feature_slices"]

    start, end = meta["feature_slices"]["data_mask"]
    mask_channels = x[:, :, start:end]
    assert np.isin(mask_channels, np.array([0.0, 1.0], dtype=np.float64)).all()
    assert np.any(mask_channels == 0.0)
    assert np.any(mask_channels == 1.0)


def test_build_forecasting_table_is_seed_deterministic(dynscm_api):
    config_mod, features_mod, _ = dynscm_api
    cfg = config_mod.DynSCMConfig()
    series = np.random.default_rng(17).normal(size=(2, 180, 3))
    y_idx = np.array([0, 2], dtype=np.int64)

    x1, y1, m1 = features_mod.build_forecasting_table(
        cfg,
        series,
        y_idx,
        n_train=12,
        n_test=7,
        seed=41,
    )
    x2, y2, m2 = features_mod.build_forecasting_table(
        cfg,
        series,
        y_idx,
        n_train=12,
        n_test=7,
        seed=41,
    )

    assert np.array_equal(x1, x2)
    assert np.array_equal(y1, y2)
    assert np.array_equal(m1["t_idx"], m2["t_idx"])
    assert np.array_equal(m1["h_idx"], m2["h_idx"])


def test_validation_errors(dynscm_api):
    config_mod, features_mod, _ = dynscm_api
    cfg = config_mod.DynSCMConfig()

    with pytest.raises(ValueError, match="shape"):
        features_mod.build_forecasting_table(
            cfg, np.zeros((4, 32)), np.array([0]), n_train=5, n_test=3, seed=0
        )

    with pytest.raises(ValueError, match="y_idx"):
        features_mod.build_forecasting_table(
            cfg,
            np.zeros((2, 200, 4)),
            np.array([99, 0], dtype=np.int64),
            n_train=10,
            n_test=5,
            seed=0,
        )

    with pytest.raises(ValueError, match="obs_mask"):
        features_mod.build_forecasting_table(
            cfg,
            np.zeros((2, 200, 4)),
            np.array([0, 1], dtype=np.int64),
            n_train=10,
            n_test=5,
            obs_mask=np.ones((2, 100, 4), dtype=bool),
            seed=0,
        )

    with pytest.raises(ValueError, match="batch_size"):
        features_mod.sample_origins_and_horizons(
            cfg, batch_size=0, num_steps=200, n_train=10, n_test=5, seed=0
        )

    with pytest.raises(ValueError, match="max_origin"):
        features_mod.sample_origins_and_horizons(
            cfg, batch_size=2, num_steps=20, n_train=5, n_test=3, seed=0
        )


def test_extract_feature_block_and_sub_slices(dynscm_api):
    config_mod, features_mod, _ = dynscm_api
    cfg = config_mod.DynSCMConfig()

    series = np.random.default_rng(50).normal(size=(2, 200, 5))
    y_idx = np.array([0, 1], dtype=np.int64)
    x, y, meta = features_mod.build_forecasting_table(
        cfg, series, y_idx, n_train=12, n_test=6, seed=51
    )

    lags_block = features_mod.extract_feature_block(x, meta, "lags_value")
    kern_block = features_mod.extract_feature_block(x, meta, "kern_value")
    assert lags_block.shape[2] > 0
    assert kern_block.shape[2] > 0
    assert (
        lags_block.shape[2] + kern_block.shape[2]
        == meta["feature_slices"]["data_values"][1]
        - meta["feature_slices"]["data_values"][0]
    )

    with pytest.raises(KeyError, match="nonexistent"):
        features_mod.extract_feature_block(x, meta, "nonexistent")


def test_deterministic_channels_present_with_expected_widths(dynscm_api):
    config_mod, features_mod, _ = dynscm_api
    cfg = config_mod.DynSCMConfig.from_dict(
        {
            "add_time_feature": True,
            "add_horizon_feature": True,
            "add_log_horizon": True,
            "add_seasonality": True,
        }
    )
    series = np.random.default_rng(60).normal(size=(2, 200, 4))
    y_idx = np.array([0, 1], dtype=np.int64)
    x, y, meta = features_mod.build_forecasting_table(
        cfg, series, y_idx, n_train=10, n_test=5, seed=61
    )
    slices = meta["feature_slices"]

    assert "time" in slices
    t_start, t_end = slices["time"]
    assert t_end - t_start == 1

    assert "horizon" in slices
    h_start, h_end = slices["horizon"]
    assert h_end - h_start == 1

    assert "log_horizon" in slices
    lh_start, lh_end = slices["log_horizon"]
    assert lh_end - lh_start == 1

    assert "seasonality" in slices
    s_start, s_end = slices["seasonality"]
    assert s_end - s_start == 2  # sin + cos
