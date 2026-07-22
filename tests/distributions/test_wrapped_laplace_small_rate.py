"""Regression coverage for small-rate wrapped Laplace densities."""

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array, pi
from pyrecest.distributions.circle.wrapped_laplace_distribution import (
    WrappedLaplaceDistribution,
)


def test_tiny_rate_pdf_approaches_uniform_density():
    distribution = WrappedLaplaceDistribution(array(1.0e-18), array(3.0))

    density = pyrecest.backend.to_numpy(distribution.pdf(array([0.0, pi])))

    assert np.isfinite(density).all()
    npt.assert_allclose(
        density,
        np.full(2, 1.0 / (2.0 * np.pi)),
        rtol=5.0e-7,
        atol=5.0e-7,
    )
