import pytest
from pyrecest.backend import array
from pyrecest.distributions.circle.wrapped_cauchy_distribution import (
    WrappedCauchyDistribution,
)


@pytest.mark.parametrize(
    "mu",
    [
        array([0.0, 1.0]),
        array([[0.0]]),
    ],
)
def test_wrapped_cauchy_rejects_non_scalar_mean(mu):
    with pytest.raises(ValueError, match="mu must be a scalar"):
        WrappedCauchyDistribution(mu, 0.5)


@pytest.mark.parametrize("mu", [float("nan"), float("inf"), -float("inf")])
def test_wrapped_cauchy_rejects_nonfinite_mean(mu):
    with pytest.raises(ValueError, match="mu must be finite"):
        WrappedCauchyDistribution(mu, 0.5)
