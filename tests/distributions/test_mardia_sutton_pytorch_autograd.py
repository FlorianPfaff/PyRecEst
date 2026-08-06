from __future__ import annotations

import pytest

import pyrecest.backend
from pyrecest.distributions.cart_prod.mardia_sutton_distribution import (
    MardiaSuttonDistribution,
)

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "pytorch",
    reason="PyTorch backend regression",
)


def test_mardia_sutton_pdf_preserves_pytorch_autograd() -> None:
    distribution = MardiaSuttonDistribution(
        mu=2.0,
        mu0=1.0,
        kappa=0.7,
        rho1=0.5,
        rho2=0.3,
        sigma=1.5,
    )
    points = torch.tensor(
        [[0.8, 1.4], [1.3, 2.7]],
        dtype=torch.float64,
        requires_grad=True,
    )

    density = distribution.pdf(points)

    assert torch.is_tensor(density)
    assert density.device == points.device
    assert density.dtype == points.dtype
    assert density.requires_grad
    assert torch.all(torch.isfinite(density))
    assert torch.all(density > 0.0)

    density.sum().backward()

    assert points.grad is not None
    assert torch.all(torch.isfinite(points.grad))
    assert torch.any(points.grad != 0.0)
