"""Regression tests for piecewise-constant input validation."""

import numpy as np
import pyrecest.backend
import pytest
from pyrecest.distributions.circle.piecewise_constant_distribution import (
    PiecewiseConstantDistribution,
)


def test_piecewise_constant_distribution_rejects_complex_weights():
    with pytest.raises(ValueError, match="Weights must contain real values"):
        PiecewiseConstantDistribution(np.asarray([1.0 + 0.0j, 2.0 + 0.0j]))


def test_piecewise_constant_pdf_rejects_complex_angles():
    distribution = PiecewiseConstantDistribution([1.0, 2.0])

    with pytest.raises(ValueError, match="xs must contain real values"):
        distribution.pdf(np.asarray([0.0 + 1.0j]))


@pytest.mark.skipif(
    pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
    reason="Not supported on JAX backend",
)
@pytest.mark.parametrize(
    "order",
    [
        1.5,
        True,
        "1",
        np.asarray([1]),
        np.timedelta64(1, "ns"),
        np.asarray(1.0 + 0.0j),
    ],
)
def test_piecewise_constant_moment_rejects_noninteger_orders(order):
    distribution = PiecewiseConstantDistribution([1.0, 2.0])

    with pytest.raises(ValueError, match="n must be a scalar integer"):
        distribution.trigonometric_moment(order)


def test_piecewise_constant_moment_accepts_numpy_integer_order():
    if pyrecest.backend.__backend_name__ == "jax":  # pylint: disable=no-member
        pytest.skip("Not supported on JAX backend")

    distribution = PiecewiseConstantDistribution([1.0, 2.0])

    assert np.isfinite(distribution.trigonometric_moment(np.int64(1)))
