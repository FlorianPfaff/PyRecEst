import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, ones
from pyrecest.distributions.conversion import convert_distribution
from pyrecest.distributions.hypersphere_subset.hyperspherical_grid_distribution import (
    HypersphericalGridDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperspherical_uniform_distribution import (
    HypersphericalUniformDistribution,
)
from pyrecest.distributions.hypersphere_subset.spherical_grid_distribution import (
    SphericalGridDistribution,
)


class _HypersphericalGridSubclass(HypersphericalGridDistribution):
    pass


class HypersphericalGridFactorySubclassPreservationTest(unittest.TestCase):
    def test_conversion_factory_preserves_requested_subclass(self):
        source = HypersphericalUniformDistribution(2)

        converted = convert_distribution(
            source, _HypersphericalGridSubclass, no_of_grid_points=12
        )

        self.assertIsInstance(converted, _HypersphericalGridSubclass)

    def test_from_function_preserves_requested_subclass(self):
        converted = _HypersphericalGridSubclass.from_function(
            lambda xs: ones(xs.shape[0]), no_of_grid_points=12, dim=2
        )

        self.assertIsInstance(converted, _HypersphericalGridSubclass)

    def test_symmetrize_preserves_builtin_spherical_subclass(self):
        dist = SphericalGridDistribution(
            array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]),
            array([1.0, 2.0]),
        )

        symmetrized = dist.symmetrize()

        self.assertIsInstance(symmetrized, SphericalGridDistribution)


if __name__ == "__main__":
    unittest.main()
