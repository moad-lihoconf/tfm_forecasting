#!/usr/bin/env python3
"""Small NICL API smoke script sending a tiny in-context classification request."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tarfile
from typing import Any

import numpy as np
import requests

_ZSTD_MAGIC = b"\x28\xb5\x2f\xfd"

try:  # pragma: no cover - optional dependency
    import zstandard as _zstd  # type: ignore[import-not-found]
except Exception:  # pragma: no cover
    _zstd = None


def _resolve_token(env_name: str) -> tuple[str, str]:
    token = os.getenv(env_name)
    if token:
        return token, env_name
    fallback = os.getenv("NICL_API_TOKEN")
    if fallback:
        return fallback, "NICL_API_TOKEN"
    raise RuntimeError(f"Missing API token in {env_name} or NICL_API_TOKEN.")


def _add_tar_json(archive: tarfile.TarFile, name: str, payload: dict[str, Any]) -> None:
    data = json.dumps(payload).encode("utf-8")
    _add_tar_bytes(archive, name, data)


def _add_tar_npy(archive: tarfile.TarFile, name: str, arr: np.ndarray) -> None:
    buffer = io.BytesIO()
    np.save(buffer, arr, allow_pickle=False)
    _add_tar_bytes(archive, name, buffer.getvalue())


def _add_tar_bytes(archive: tarfile.TarFile, name: str, data: bytes) -> None:
    info = tarfile.TarInfo(name=name)
    info.size = len(data)
    archive.addfile(info, io.BytesIO(data))


def _build_request_body() -> tuple[bytes, int]:
    # Tiny toy classification request with a few rows.
    x_train = np.array(
        [
            [0.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [1.0, 1.0],
            [2.0, 2.0],
            [2.0, 3.0],
        ],
        dtype=np.float32,
    )
    y_train = np.array([0, 0, 0, 1, 1, 1], dtype=np.int64)
    x_test = np.array([[0.1, 0.2], [1.5, 1.7], [2.2, 2.6]], dtype=np.float32)

    metadata = {
        "version": 1,
        "method": "fit_predict",
        "model": "nicl-small",
        "dataset": "nicl_api_smoke",
        "prompter_config": None,
        "memory_optimization": False,
        "preprocess": True,
        "metadata": {"task": "classification", "num_classes": 2},
        "user": "",
    }

    tar_buffer = io.BytesIO()
    with tarfile.open(fileobj=tar_buffer, mode="w") as archive:
        _add_tar_json(archive, "metadata.json", metadata)
        _add_tar_npy(archive, "X_train.npy", x_train)
        _add_tar_npy(archive, "y_train.npy", y_train)
        _add_tar_npy(archive, "X_test.npy", x_test)

    tar_buffer.seek(0)
    return tar_buffer.read(), x_test.shape[0]


def _decode_tar_payload(payload: bytes) -> dict[str, Any]:
    if payload.startswith(_ZSTD_MAGIC):
        if _zstd is None:
            raise RuntimeError(
                "Response is zstd-compressed but zstandard is not installed."
            )
        with _zstd.ZstdDecompressor().stream_reader(io.BytesIO(payload)) as reader:
            stream = io.BytesIO(reader.read())
    else:
        stream = io.BytesIO(payload)

    out: dict[str, Any] = {}
    with tarfile.open(fileobj=stream, mode="r:*") as archive:
        for member in archive:
            if not member.isfile():
                continue
            member_file = archive.extractfile(member)
            if member_file is None:
                continue
            name = member.name
            stem = name.rsplit(".", 1)[0]
            data = member_file.read()
            if name.endswith(".npy"):
                out[stem] = np.load(io.BytesIO(data), allow_pickle=False)
            elif name.endswith(".json"):
                out[stem] = json.loads(data.decode("utf-8"))
            else:
                out[name] = data
    return out


def _normalize_prediction_payload(raw: dict[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    if "predictions" in raw:
        result["predictions"] = raw["predictions"]
    elif "predict" in raw:
        result["predictions"] = raw["predict"]
    if "probabilities" in raw:
        result["probabilities"] = raw["probabilities"]
    elif "predict_proba" in raw:
        result["probabilities"] = raw["predict_proba"]
    if "metadata" in raw and isinstance(raw["metadata"], dict):
        result.update(raw["metadata"])
    return result


def _decode_response(response: requests.Response) -> dict[str, Any]:
    body = response.content
    # Some error paths return JSON body directly.
    if response.headers.get("Content-Type", "").startswith("application/json"):
        payload = response.json()
        if not isinstance(payload, dict):
            raise RuntimeError("Expected JSON object response.")
        return payload
    if body.startswith(b"{"):
        payload = json.loads(body.decode("utf-8"))
        if not isinstance(payload, dict):
            raise RuntimeError("Expected JSON object response.")
        return payload
    extracted = _decode_tar_payload(body)
    return _normalize_prediction_payload(extracted)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Send a tiny NICL request to verify API access."
    )
    parser.add_argument(
        "--url",
        default="https://api.prediction.neuralk-ai.com/api/v1/inference",
        help="NICL inference endpoint URL.",
    )
    parser.add_argument(
        "--token_env",
        default="NEURALK_API_KEY",
        help="Environment variable name holding the API key.",
    )
    parser.add_argument(
        "--timeout_seconds",
        type=float,
        default=30.0,
        help="HTTP timeout in seconds.",
    )
    return parser


def main() -> None:
    args = _build_parser().parse_args()
    token, token_source = _resolve_token(args.token_env)
    body, expected_rows = _build_request_body()
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/x-tar",
        "Accept": "application/x-tar",
        "X-Content-SHA256": hashlib.sha256(body).hexdigest(),
    }

    print(f"Endpoint: {args.url}")
    print(f"Token source: {token_source}")
    print(f"Request bytes: {len(body)}")

    response = requests.post(
        args.url,
        data=body,
        headers=headers,
        timeout=args.timeout_seconds,
    )

    if response.status_code >= 400:
        detail = response.text[:500].strip()
        raise RuntimeError(
            f"NICL API request failed ({response.status_code}): {detail}"
        )

    decoded = _decode_response(response)
    pred = np.asarray(decoded.get("predictions", []))
    proba = np.asarray(decoded.get("probabilities", []))

    print("Response keys:", sorted(decoded.keys()))
    print("Predictions:", pred.tolist())
    if proba.size:
        print("Probabilities shape:", tuple(proba.shape))
        print("Probabilities[0]:", proba[0].tolist())
    print("Request id:", response.headers.get("x-request-id"))

    if pred.shape[0] != expected_rows:
        raise RuntimeError(
            f"Prediction row mismatch: expected {expected_rows}, got {pred.shape[0]}."
        )
    print("Smoke test passed.")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1) from exc
