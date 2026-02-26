from __future__ import annotations

from pathlib import Path

import pytest

from tfmplayground import gcs_utils


class _FakeBlob:
    def __init__(self, exists: bool = True):
        self._exists = exists
        self.uploaded_from: str | None = None
        self.downloaded_to: str | None = None

    def exists(self, _client) -> bool:
        return self._exists

    def download_to_filename(self, filename: str) -> None:
        Path(filename).write_text("payload", encoding="utf-8")
        self.downloaded_to = filename

    def upload_from_filename(self, filename: str) -> None:
        self.uploaded_from = filename


class _FakeBucket:
    def __init__(self, blob: _FakeBlob):
        self._blob = blob

    def blob(self, _blob_name: str) -> _FakeBlob:
        return self._blob


class _FakeClient:
    def __init__(self, blob: _FakeBlob):
        self._blob = blob

    def bucket(self, _bucket_name: str) -> _FakeBucket:
        return _FakeBucket(self._blob)


def test_parse_gcs_uri_ok() -> None:
    bucket, blob = gcs_utils.parse_gcs_uri("gs://demo-bucket/a/b/file.h5")
    assert bucket == "demo-bucket"
    assert blob == "a/b/file.h5"


def test_parse_gcs_uri_invalid() -> None:
    with pytest.raises(ValueError):
        gcs_utils.parse_gcs_uri("/tmp/file.h5")
    with pytest.raises(ValueError):
        gcs_utils.parse_gcs_uri("gs://bucket")


def test_download_gcs_to_local_uses_storage_client(tmp_path: Path, monkeypatch) -> None:
    blob = _FakeBlob(exists=True)
    monkeypatch.setattr(gcs_utils, "_get_storage_client", lambda: _FakeClient(blob))

    local_path = gcs_utils.download_gcs_to_local(
        "gs://bucket/path/file.h5",
        local_dir=tmp_path,
    )

    assert local_path.exists()
    assert local_path.read_text(encoding="utf-8") == "payload"
    assert blob.downloaded_to == str(local_path)


def test_upload_local_file_to_gcs_uses_storage_client(
    tmp_path: Path,
    monkeypatch,
) -> None:
    source = tmp_path / "weights.pth"
    source.write_text("weights", encoding="utf-8")

    blob = _FakeBlob(exists=True)
    monkeypatch.setattr(gcs_utils, "_get_storage_client", lambda: _FakeClient(blob))

    gcs_utils.upload_local_file_to_gcs(source, "gs://bucket/path/weights.pth")
    assert blob.uploaded_from == str(source)
