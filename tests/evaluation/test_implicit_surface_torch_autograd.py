from __future__ import annotations

import pytest
from pyrecest.evaluation import surface_band_probability_from_signed_distance

torch = pytest.importorskip("torch")


def test_surface_band_probability_preserves_torch_autograd() -> None:
    distance = torch.tensor([0.0, 0.2], dtype=torch.float64, requires_grad=True)
    distance_std = torch.tensor([0.05, 0.05], dtype=torch.float64)

    probability = surface_band_probability_from_signed_distance(
        distance,
        distance_std,
        0.1,
    )

    assert torch.is_tensor(probability)
    assert probability.device == distance.device
    assert probability.dtype == distance.dtype
    assert probability.requires_grad
    assert torch.all(torch.isfinite(probability))
    assert torch.all((0.0 <= probability) & (probability <= 1.0))

    probability.sum().backward()

    assert distance.grad is not None
    assert torch.all(torch.isfinite(distance.grad))
    assert distance.grad[1] < 0.0
