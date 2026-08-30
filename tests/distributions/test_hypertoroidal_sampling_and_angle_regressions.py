import pytest
from pyrecest.backend import array
from pyrecest.distributions import AbstractHypertoroidalDistribution
from pyrecest.distributions.hypertorus.hypertoroidal_uniform_distribution import (
    HypertoroidalUniformDistribution,
)


def test_uniform_metropolis_hastings_does_not_require_unique_mean():
    dist = HypertoroidalUniformDistribution(2)

    samples = dist.sample_metropolis_hastings(4, burn_in=0, skipping=1)

    assert samples.shape == (4, 2)


@pytest.mark.parametrize(
    ("alpha", "beta"),
    [
        (float("inf"), 0.0),
        (float("-inf"), 0.0),
        (0.0, float("inf")),
        (0.0, float("-inf")),
    ],
)
def test_angular_error_rejects_infinite_inputs(alpha, beta):
    with pytest.raises(ValueError, match="finite"):
        AbstractHypertoroidalDistribution.angular_error(array(alpha), array(beta))
