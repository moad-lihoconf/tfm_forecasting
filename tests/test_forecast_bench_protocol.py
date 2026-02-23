from __future__ import annotations

import numpy as np
import pytest

from tfmplayground.benchmarks.forecasting.protocol import (
    generate_rolling_origin_indices,
    validate_indices,
)


def test_generate_rolling_origin_indices_enforces_chronology_and_bounds():
    out = generate_rolling_origin_indices(
        series_length=200,
        horizons=(1, 3, 6, 12),
        n_train=32,
        n_test=16,
        required_lag=32,
        seed=42,
    )

    split = out.split_index
    assert out.t_idx.shape == (48,)
    assert out.h_idx.shape == (48,)
    assert np.max(out.t_idx[:split]) < np.min(out.t_idx[split:])
    assert np.all(out.t_idx >= 32)
    assert np.all(out.t_idx + out.h_idx < 200)


def test_validate_indices_raises_on_leakage_or_bad_order():
    with pytest.raises(ValueError, match="Strict chronology"):
        validate_indices(
            t_idx=np.array([10, 11, 9, 12], dtype=np.int64),
            h_idx=np.array([1, 1, 1, 1], dtype=np.int64),
            n_train=2,
            series_length=20,
            required_lag=0,
        )

    with pytest.raises(ValueError, match="outside timeline"):
        validate_indices(
            t_idx=np.array([10, 11, 15, 16], dtype=np.int64),
            h_idx=np.array([1, 1, 10, 10], dtype=np.int64),
            n_train=2,
            series_length=20,
            required_lag=0,
        )


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            {
                "series_length": 1,
                "horizons": (1,),
                "n_train": 1,
                "n_test": 1,
                "required_lag": 0,
                "seed": 0,
            },
            "series_length",
        ),
        (
            {
                "series_length": 200,
                "horizons": (1,),
                "n_train": 0,
                "n_test": 1,
                "required_lag": 0,
                "seed": 0,
            },
            "n_train",
        ),
        (
            {
                "series_length": 200,
                "horizons": (1,),
                "n_train": 1,
                "n_test": 0,
                "required_lag": 0,
                "seed": 0,
            },
            "n_test",
        ),
        (
            {
                "series_length": 200,
                "horizons": (1,),
                "n_train": 1,
                "n_test": 1,
                "required_lag": -1,
                "seed": 0,
            },
            "required_lag",
        ),
        (
            {
                "series_length": 200,
                "horizons": (),
                "n_train": 1,
                "n_test": 1,
                "required_lag": 0,
                "seed": 0,
            },
            "non-empty",
        ),
        (
            {
                "series_length": 200,
                "horizons": (-1,),
                "n_train": 1,
                "n_test": 1,
                "required_lag": 0,
                "seed": 0,
            },
            "positive",
        ),
        (
            {
                "series_length": 10,
                "horizons": (8,),
                "n_train": 1,
                "n_test": 1,
                "required_lag": 5,
                "seed": 0,
            },
            "Insufficient",
        ),
    ],
)
def test_generate_rolling_origin_indices_rejects_invalid_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        generate_rolling_origin_indices(**kwargs)


@pytest.mark.parametrize(
    "kwargs, match",
    [
        (
            {
                "t_idx": np.array([1, 2, 3]),
                "h_idx": np.array([1, 2]),
                "n_train": 1,
                "series_length": 10,
                "required_lag": 0,
            },
            "same shape",
        ),
        (
            {
                "t_idx": np.array([[1, 2], [3, 4]]),
                "h_idx": np.array([[1, 1], [1, 1]]),
                "n_train": 1,
                "series_length": 10,
                "required_lag": 0,
            },
            "1D",
        ),
        (
            {
                "t_idx": np.array([5], dtype=np.int64),
                "h_idx": np.array([1], dtype=np.int64),
                "n_train": 1,
                "series_length": 10,
                "required_lag": 0,
            },
            "at least one test row",
        ),
        (
            {
                "t_idx": np.array([1, 5], dtype=np.int64),
                "h_idx": np.array([1, 1], dtype=np.int64),
                "n_train": 1,
                "series_length": 10,
                "required_lag": 3,
            },
            "lag floor",
        ),
        (
            {
                "t_idx": np.array([5, 6], dtype=np.int64),
                "h_idx": np.array([0, 1], dtype=np.int64),
                "n_train": 1,
                "series_length": 10,
                "required_lag": 0,
            },
            "positive",
        ),
    ],
)
def test_validate_indices_rejects_invalid_inputs(kwargs, match):
    with pytest.raises(ValueError, match=match):
        validate_indices(**kwargs)
