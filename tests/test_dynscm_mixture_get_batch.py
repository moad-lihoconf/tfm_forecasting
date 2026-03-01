from __future__ import annotations

from typing import cast

import numpy as np
import torch

from tfmplayground.priors.dynscm.research import LinearPhaseSchedule, MixtureGetBatch


class _FakeGetBatch:
    def __init__(self, value: int) -> None:
        self.value = value

    def __call__(self, batch_size: int, num_datapoints_max: int, num_features: int):
        tensor = torch.full(
            (batch_size, num_datapoints_max, num_features),
            float(self.value),
        )
        return {
            "x": tensor,
            "y": torch.full((batch_size, num_datapoints_max), float(self.value)),
            "target_y": torch.full((batch_size, num_datapoints_max), float(self.value)),
            "single_eval_pos": 1,
            "num_datapoints": num_datapoints_max,
            "target_mask": torch.ones(
                (batch_size, num_datapoints_max), dtype=torch.bool
            ),
        }


def test_mixture_get_batch_is_deterministic_under_fixed_seed() -> None:
    schedule = LinearPhaseSchedule(
        [
            (0.3, (1.0, 0.0, 0.0), None),
            (0.7, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            (1.0, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ]
    )
    mix_a = MixtureGetBatch(
        children=(_FakeGetBatch(1), _FakeGetBatch(2), _FakeGetBatch(3)),
        child_names=("stable", "temporal", "target"),
        schedule=schedule,
        seed=17,
        total_batches=10,
    )
    mix_b = MixtureGetBatch(
        children=(_FakeGetBatch(1), _FakeGetBatch(2), _FakeGetBatch(3)),
        child_names=("stable", "temporal", "target"),
        schedule=schedule,
        seed=17,
        total_batches=10,
    )

    outputs_a = [cast(torch.Tensor, mix_a(1, 2, 1)["x"]).clone() for _ in range(10)]
    outputs_b = [cast(torch.Tensor, mix_b(1, 2, 1)["x"]).clone() for _ in range(10)]

    for batch_a, batch_b in zip(outputs_a, outputs_b, strict=True):
        assert torch.equal(batch_a, batch_b)
    assert mix_a.selection_counts == mix_b.selection_counts


def test_phase_schedule_shifts_mass_toward_later_children() -> None:
    schedule = LinearPhaseSchedule(
        [
            (0.3, (1.0, 0.0, 0.0), None),
            (0.7, (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
            (1.0, (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
        ]
    )
    early = schedule.weights_at(0.0)
    mid = schedule.weights_at(0.5)
    late = schedule.weights_at(1.0)

    assert np.allclose(early, np.array([1.0, 0.0, 0.0]))
    assert mid[1] > mid[0]
    assert np.allclose(late, np.array([0.0, 0.0, 1.0]))
