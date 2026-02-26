from __future__ import annotations

import sys
from pathlib import Path

from tfmplayground.priors import main as priors_main


def test_priors_cli_uploads_when_save_path_is_gcs(monkeypatch, tmp_path: Path) -> None:
    uploaded: dict[str, str] = {}

    class _DummyPrior:
        pass

    def _fake_dump_prior_to_h5(
        _prior,
        _max_classes,
        _batch_size,
        save_path,
        _problem_type,
        _max_seq_len,
        _max_features,
    ) -> None:
        Path(save_path).write_text("dump", encoding="utf-8")

    def _fake_upload(local_path: str | Path, uri: str) -> None:
        uploaded["local_path"] = str(local_path)
        uploaded["uri"] = uri
        assert Path(local_path).exists()

    monkeypatch.setattr(
        priors_main,
        "DynSCMPriorDataLoader",
        lambda **kwargs: _DummyPrior(),
    )
    monkeypatch.setattr(priors_main, "dump_prior_to_h5", _fake_dump_prior_to_h5)
    monkeypatch.setattr(priors_main, "upload_local_file_to_gcs", _fake_upload)

    monkeypatch.setattr(
        sys,
        "argv",
        [
            "priors-main",
            "--lib",
            "dynscm",
            "--num_batches",
            "1",
            "--batch_size",
            "1",
            "--max_seq_len",
            "8",
            "--max_features",
            "8",
            "--max_classes",
            "0",
            "--save_path",
            "gs://demo-bucket/tfm_forecasting/priors/example.h5",
        ],
    )

    priors_main.main()

    assert uploaded["uri"] == "gs://demo-bucket/tfm_forecasting/priors/example.h5"
    assert uploaded["local_path"].endswith("example.h5")
