"""Regression tests for piecewise-constant input validation."""

from fractions import Fraction

import numpy as np
import pytest
import pyrecest.backend

from pyrecest.distributions.circle.piecewise_constant_distribution import (
    PiecewiseConstantDistribution,
    _validate_positive_sample_count,
)


def test_piecewise_constant_distribution_rejects_complex_weights():
    with pytest.raises(ValueError, match="Weights must contain real values"):
        PiecewiseConstantDistribution(np.asarray([1.0 + 0.0j, 2.0 + 0.0j]))


def test_piecewise_constant_pdf_rejects_complex_angles():
    distribution = PiecewiseConstantDistribution([1.0, 2.0])

    with pytest.raises(ValueError, match="xs must contain real values"):
        distribution.pdf(np.asarray([0.0 + 1.0j]))


def test_piecewise_constant_sample_count_rejects_fraction_rounded_by_binary64():
    fractional_count = Fraction(2**54 + 1, 2)

    assert float(fractional_count).is_integer()
    with pytest.raises(ValueError, match="n must be a finite integer"):
        _validate_positive_sample_count(fractional_count)


def test_piecewise_constant_sample_count_accepts_exact_rational_integer():
    assert _validate_positive_sample_count(Fraction(8, 2)) == 4


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
