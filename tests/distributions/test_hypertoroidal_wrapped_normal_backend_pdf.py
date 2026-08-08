import numpy as np
import pyrecest.backend as backend
import pytest
from pyrecest.distributions import HypertoroidalWrappedNormalDistribution


def _distribution():
    return HypertoroidalWrappedNormalDistribution(
        [0.2, -0.1],
        [[0.5, 0.1], [0.1, 0.8]],
    )


def test_hypertoroidal_wrapped_normal_pdf_matches_reference_values():
    values = _distribution().pdf(
        backend.array([[0.1, 0.2], [1.0, -2.0]]),
        m=2,
    )

    np.testing.assert_allclose(
        backend.to_numpy(values),
        np.array([0.23628565059916362, 0.00885153202932088]),
        rtol=1e-6,
        atol=1e-9,
    )


def test_pytorch_hypertoroidal_wrapped_normal_pdf_preserves_gradients():
    if backend.__backend_name__ != "pytorch":
        pytest.skip("PyTorch-specific autodiff regression")

    torch = pytest.importorskip("torch")
    distribution = _distribution()
    points = torch.tensor(
        [[0.1, 0.2], [1.0, -2.0]],
        dtype=distribution.C.dtype,
        device=distribution.C.device,
        requires_grad=True,
    )

    values = distribution.pdf(points, m=2)
    values.sum().backward()

    assert points.grad is not None
    assert points.grad.shape == points.shape
    assert bool(torch.isfinite(points.grad).all())
    assert bool(torch.count_nonzero(points.grad) > 0)
