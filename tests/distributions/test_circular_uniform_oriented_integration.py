import pytest

from pyrecest.backend import array, pi
from pyrecest.distributions import CircularUniformDistribution


def test_reversed_circular_interval_preserves_signed_integral():
    dist = CircularUniformDistribution()

    value = dist.integrate(array([2.0 * pi, -1.0]))

    assert float(value) == pytest.approx((-1.0 - 2.0 * pi) / (2.0 * pi))
