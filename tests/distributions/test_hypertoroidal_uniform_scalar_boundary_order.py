import pytest
from pyrecest.distributions.hypertorus.hypertoroidal_uniform_distribution import (
    HypertoroidalUniformDistribution,
)


@pytest.mark.parametrize(
    "integration_boundaries",
    [
        (1.0, 0.0),
        ([1.0], [0.0]),
    ],
)
def test_integrate_rejects_reversed_one_dimensional_boundaries(
    integration_boundaries,
):
    distribution = HypertoroidalUniformDistribution(1)

    with pytest.raises(ValueError, match="increasing"):
        distribution.integrate(integration_boundaries)
