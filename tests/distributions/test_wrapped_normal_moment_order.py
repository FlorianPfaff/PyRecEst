"""Regression coverage for wrapped-normal trigonometric-moment orders."""

import pytest
from pyrecest.backend import allclose, array, conj
from pyrecest.distributions import WrappedNormalDistribution


@pytest.mark.parametrize("order", [0.5, True, "1"])
def test_wrapped_normal_rejects_non_integer_moment_orders(order):
    distribution = WrappedNormalDistribution(array(0.3), array(0.7))

    with pytest.raises(ValueError, match="n must be an integer"):
        distribution.trigonometric_moment(order)


def test_wrapped_normal_accepts_negative_integer_moment_orders():
    distribution = WrappedNormalDistribution(array(0.3), array(0.7))

    positive_moment = distribution.trigonometric_moment(2)
    negative_moment = distribution.trigonometric_moment(-2)

    assert bool(allclose(negative_moment, conj(positive_moment)))
