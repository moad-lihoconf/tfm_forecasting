from __future__ import annotations

import importlib.util
import io
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest


def _load_script_module(script_name: str):
    path = Path(__file__).resolve().parents[1] / "scripts" / script_name
    spec = importlib.util.spec_from_file_location(path.stem, path)
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


prepare_mod = _load_script_module("prepare_forecast_datasets.py")


def test_build_m4_weekly_panel_combines_train_and_test(monkeypatch):
    train_csv = 'id,v1,v2\n"W1",1,2\n"W2",3,4\n'
    test_csv = 'id,v1\n"W1",5\n"W2",6\n'

    responses = [train_csv.encode("utf-8"), test_csv.encode("utf-8")]
    monkeypatch.setattr(
        prepare_mod,
        "_download_bytes",
        lambda url, timeout: responses.pop(0),
    )

    panel = prepare_mod.build_m4_weekly_panel(timeout=1.0)

    assert panel.shape == (2, 3)
    assert np.allclose(panel[0], [1.0, 2.0, 5.0], equal_nan=True)
    assert np.allclose(panel[1], [3.0, 4.0, 6.0], equal_nan=True)


def test_build_tourism_monthly_panel_parses_tsf(monkeypatch):
    tsf_text = """
@relation tourism
@attribute series_name string
@attribute start_timestamp date
@frequency monthly
@horizon 24
@missing false
@equallength false
@data
T1:1979-01-01 00-00-00:1,2,3
T2:1979-01-01 00-00-00:4,5
""".strip()

    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as zf:
        zf.writestr("tourism_monthly_dataset.tsf", tsf_text)
    monkeypatch.setattr(
        prepare_mod,
        "_download_bytes",
        lambda url, timeout: buffer.getvalue(),
    )

    panel = prepare_mod.build_tourism_monthly_panel(timeout=1.0)

    assert panel.shape == (2, 3)
    assert np.allclose(panel[0], [1.0, 2.0, 3.0], equal_nan=True)
    assert np.allclose(panel[1], [4.0, 5.0, np.nan], equal_nan=True)


def test_prepare_dataset_writes_npz(tmp_path: Path, monkeypatch):
    panel = np.array([[1.0, 2.0], [3.0, np.nan]], dtype=np.float64)
    monkeypatch.setattr(prepare_mod, "build_m4_weekly_panel", lambda timeout: panel)

    out_path = prepare_mod.prepare_dataset(
        "m4_weekly",
        cache_dir=tmp_path,
        timeout=1.0,
        force=True,
    )

    saved = np.load(out_path, allow_pickle=False)["series"]
    assert out_path.exists()
    assert saved.shape == (2, 2)
    assert np.isnan(saved[1, 1])


def test_prepare_dataset_skips_existing_without_force(tmp_path: Path, monkeypatch):
    existing = tmp_path / "m4_weekly.npz"
    np.savez_compressed(existing, series=np.array([[1.0, 2.0]], dtype=np.float64))
    monkeypatch.setattr(
        prepare_mod,
        "build_m4_weekly_panel",
        lambda timeout: (_ for _ in ()).throw(AssertionError("should not rebuild")),
    )

    out_path = prepare_mod.prepare_dataset(
        "m4_weekly",
        cache_dir=tmp_path,
        timeout=1.0,
        force=False,
    )

    assert out_path == existing


def test_build_tourism_monthly_panel_fails_on_missing_tsf(monkeypatch):
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, mode="w") as zf:
        zf.writestr("readme.txt", "no tsf here")
    monkeypatch.setattr(
        prepare_mod,
        "_download_bytes",
        lambda url, timeout: buffer.getvalue(),
    )

    with pytest.raises(ValueError, match=".tsf"):
        prepare_mod.build_tourism_monthly_panel(timeout=1.0)
