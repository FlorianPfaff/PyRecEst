from __future__ import annotations

import pyrecest.backend
import pytest
from pyrecest.distributions import ComplexBinghamDistribution

torch = pytest.importorskip("torch")

pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "pytorch",
    reason="PyTorch backend regression",
)


def test_single_point_pdf_preserves_pytorch_autograd() -> None:
    distribution = ComplexBinghamDistribution(
        torch.tensor(
            [[-3.0, 0.0], [0.0, 0.0]],
            dtype=torch.complex128,
        )
    )
    point = torch.tensor(
        [0.5 + 0.5j, 0.5 - 0.5j],
        dtype=torch.complex128,
        requires_grad=True,
    )

    density = distribution.pdf(point)

    assert torch.is_tensor(density)
    assert density.ndim == 0
    assert density.dtype == torch.float64
    assert density.device == point.device
    assert density.requires_grad
    assert torch.isfinite(density)
    assert density > 0.0

    density.backward()

    assert point.grad is not None
    assert torch.all(torch.isfinite(point.grad))
    assert torch.any(point.grad != 0.0)
