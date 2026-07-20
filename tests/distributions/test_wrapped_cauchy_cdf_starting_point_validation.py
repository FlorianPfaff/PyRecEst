import pytest

from pyrecest.backend import array
from pyrecest.distributions.circle.wrapped_cauchy_distribution import (
    WrappedCauchyDistribution,
)


@pytest.mark.parametrize("starting_point", ([0.0, 1.0], [[0.0]]))
def test_wrapped_cauchy_cdf_rejects_non_scalar_starting_points(starting_point):
    distribution = WrappedCauchyDistribution(0.0, 0.5)

    with pytest.raises(ValueError, match="starting_point must be a scalar"):
        distribution.cdf(array([0.5]), starting_point=starting_point)


@pytest.mark.parametrize(
    "starting_point", (float("nan"), float("inf"), float("-inf"))
)
def test_wrapped_cauchy_cdf_rejects_nonfinite_starting_points(starting_point):
    distribution = WrappedCauchyDistribution(0.0, 0.5)

    with pytest.raises(ValueError, match="starting_point must be finite"):
        distribution.cdf(array([0.5]), starting_point=starting_point)
