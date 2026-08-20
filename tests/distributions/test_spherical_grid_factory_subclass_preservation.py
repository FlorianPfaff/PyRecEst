from pyrecest.backend import ones
from pyrecest.distributions.conversion import convert_distribution
from pyrecest.distributions.hypersphere_subset.custom_hyperspherical_distribution import (
    CustomHypersphericalDistribution,
)
from pyrecest.distributions.hypersphere_subset.spherical_grid_distribution import (
    SphericalGridDistribution,
)


class _SphericalGridSubclass(SphericalGridDistribution):
    pass


def _uniform_density(xs):
    return ones(xs.shape[0])


def test_convert_distribution_preserves_spherical_grid_subclass():
    source = CustomHypersphericalDistribution(_uniform_density, dim=2)

    converted = convert_distribution(
        source,
        _SphericalGridSubclass,
        no_of_grid_points=12,
    )

    assert type(converted) is _SphericalGridSubclass


def test_from_function_preserves_spherical_grid_subclass():
    converted = _SphericalGridSubclass.from_function(
        _uniform_density,
        no_of_grid_points=12,
    )

    assert type(converted) is _SphericalGridSubclass
