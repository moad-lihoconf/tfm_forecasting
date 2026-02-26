"""Utilities for handling local and Google Cloud Storage paths."""

from __future__ import annotations

import hashlib
import os
import tempfile
from pathlib import Path


def is_gcs_uri(path: str) -> bool:
    """Return True when `path` is a Google Cloud Storage URI."""
    return path.startswith("gs://")


def parse_gcs_uri(uri: str) -> tuple[str, str]:
    """Parse a `gs://bucket/object` URI into bucket and object path."""
    if not is_gcs_uri(uri):
        raise ValueError(f"Expected gs:// URI, got: {uri!r}")

    payload = uri[5:]
    if not payload:
        raise ValueError(f"GCS URI is missing bucket name: {uri!r}")

    bucket, sep, blob_name = payload.partition("/")
    if not bucket:
        raise ValueError(f"GCS URI is missing bucket name: {uri!r}")
    if not sep or not blob_name:
        raise ValueError(f"GCS URI is missing object path: {uri!r}")
    return bucket, blob_name


def _get_storage_client():
    try:
        from google.cloud import storage
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "google-cloud-storage is required for gs:// paths. "
            "Install it or run inside the GPU container."
        ) from exc
    return storage.Client()


def _safe_local_name(uri: str) -> str:
    digest = hashlib.sha256(uri.encode("utf-8")).hexdigest()[:12]
    base = Path(uri).name or "artifact"
    return f"{base}.{digest}"


def download_gcs_to_local(
    uri: str,
    *,
    local_dir: str | os.PathLike[str] | None = None,
    overwrite: bool = False,
) -> Path:
    """Download a GCS object to a local cache file and return its path."""
    bucket_name, blob_name = parse_gcs_uri(uri)

    if local_dir is None:
        local_root = Path(tempfile.gettempdir()) / "tfmplayground_gcs_cache"
    else:
        local_root = Path(local_dir)
    local_root.mkdir(parents=True, exist_ok=True)

    local_path = local_root / _safe_local_name(uri)
    if local_path.exists() and not overwrite:
        return local_path

    client = _get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    if not blob.exists(client):
        raise FileNotFoundError(f"GCS object does not exist: {uri}")

    blob.download_to_filename(str(local_path))
    return local_path


def upload_local_file_to_gcs(local_path: str | os.PathLike[str], uri: str) -> None:
    """Upload a local file to a GCS URI."""
    source = Path(local_path)
    if not source.exists() or not source.is_file():
        raise FileNotFoundError(f"Local file does not exist: {source}")

    bucket_name, blob_name = parse_gcs_uri(uri)
    client = _get_storage_client()
    bucket = client.bucket(bucket_name)
    blob = bucket.blob(blob_name)
    blob.upload_from_filename(str(source))


def path_for_read(path: str) -> str:
    """Return a local path suitable for reading with file-only libraries."""
    if is_gcs_uri(path):
        return str(download_gcs_to_local(path))
    return path


def maybe_upload(path: str, *, local_source: str | os.PathLike[str]) -> None:
    """Upload `local_source` when destination `path` is gs://, else no-op."""
    if is_gcs_uri(path):
        upload_local_file_to_gcs(local_source, path)
