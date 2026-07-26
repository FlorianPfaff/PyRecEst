import pytest
from pyrecest.backend import array
from pyrecest.distributions import WrappedNormalDistribution


@pytest.mark.parametrize("starting_point", ([0.0, 1.0], [[0.0]]))
def test_wrapped_normal_cdf_rejects_non_scalar_starting_points(starting_point):
    distribution = WrappedNormalDistribution(0.0, 0.5)

    with pytest.raises(ValueError, match="starting_point must be a scalar"):
        distribution.cdf(array([0.5]), starting_point=starting_point)


@pytest.mark.parametrize("starting_point", (float("nan"), float("inf"), float("-inf")))
def test_wrapped_normal_cdf_rejects_nonfinite_starting_points(starting_point):
    distribution = WrappedNormalDistribution(0.0, 0.5)

    with pytest.raises(ValueError, match="starting_point must be finite"):
        distribution.cdf(array([0.5]), starting_point=starting_point)


def test_wrapped_normal_cdf_accepts_singleton_starting_point():
    distribution = WrappedNormalDistribution(0.0, 0.5)

    scalar_result = distribution.cdf(array([0.5]), starting_point=0.25)
    singleton_result = distribution.cdf(array([0.5]), starting_point=array([0.25]))

    assert float(scalar_result) == pytest.approx(float(singleton_result))
