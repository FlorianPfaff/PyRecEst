import pytest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions.nonperiodic.hyperrectangular_uniform_distribution import (
    HyperrectangularUniformDistribution,
)


@pytest.mark.parametrize(
    "bounds",
    (
        [[0.0 + 0.0j, 1.0 + 0.0j]],
        [[0.0 + 0.0j, 1.0 + 1.0j]],
    ),
)
def test_constructor_rejects_complex_bounds(bounds):
    with pytest.raises(ValueError, match="finite real values"):
        HyperrectangularUniformDistribution(array(bounds))


@pytest.mark.parametrize(
    "integration_boundaries",
    (
        [[0.0 + 0.0j, 0.5 + 0.0j]],
        [[0.0 + 0.0j, 0.5 + 0.5j]],
    ),
)
def test_integrate_rejects_complex_boundaries(integration_boundaries):
    distribution = HyperrectangularUniformDistribution(array([[0.0, 1.0]]))

    with pytest.raises(ValueError, match="finite real values"):
        distribution.integrate(array(integration_boundaries))
