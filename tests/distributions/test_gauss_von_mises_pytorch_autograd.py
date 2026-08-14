from __future__ import annotations

import pyrecest.backend
import pytest
from pyrecest.distributions.cart_prod.gauss_von_mises_distribution import (
    GaussVonMisesDistribution,
)

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "pytorch",
    reason="PyTorch backend regression",
)


def test_single_point_pdf_preserves_pytorch_autograd() -> None:
    distribution = GaussVonMisesDistribution(
        mu=2.0,
        P=1.3,
        alpha=3.0,
        beta=0.0,
        Gamma=0.001,
        kappa=0.7,
    )
    point = torch.tensor(
        [0.8, 1.4],
        dtype=torch.float64,
        requires_grad=True,
    )

    density = distribution.pdf(point)

    assert torch.is_tensor(density)
    assert density.ndim == 0
    assert density.dtype == point.dtype
    assert density.device == point.device
    assert density.requires_grad
    assert torch.isfinite(density)
    assert density > 0.0

    density.backward()

    assert point.grad is not None
    assert torch.all(torch.isfinite(point.grad))
    assert torch.all(point.grad != 0.0)
