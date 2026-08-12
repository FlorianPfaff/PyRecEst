from __future__ import annotations

import pyrecest.backend
import pytest
from pyrecest.distributions.cart_prod.partially_wrapped_normal_distribution import (
    PartiallyWrappedNormalDistribution,
)

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "pytorch",
    reason="PyTorch backend regression",
)


def test_partially_wrapped_normal_pdf_preserves_pytorch_autograd() -> None:
    dtype = torch.float64
    distribution = PartiallyWrappedNormalDistribution(
        torch.tensor([0.4, -0.2], dtype=dtype),
        torch.tensor([[0.8, 0.1], [0.1, 1.2]], dtype=dtype),
        bound_dim=1,
    )
    points = torch.tensor(
        [[0.7, 0.3], [1.1, -0.5]],
        dtype=dtype,
        requires_grad=True,
    )

    density = distribution.pdf(points, m=1)

    assert torch.is_tensor(density)
    assert density.shape == (2,)
    assert density.device == points.device
    assert density.dtype == points.dtype
    assert density.requires_grad
    assert torch.all(torch.isfinite(density))
    assert torch.all(density > 0.0)

    density.sum().backward()

    assert points.grad is not None
    assert torch.all(torch.isfinite(points.grad))
    assert torch.any(points.grad != 0.0)
