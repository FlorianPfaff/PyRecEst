import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import ones
from pyrecest.distributions.conversion import convert_distribution
from pyrecest.distributions.hypersphere_subset.hyperspherical_grid_distribution import (
    HypersphericalGridDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperspherical_uniform_distribution import (
    HypersphericalUniformDistribution,
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


if __name__ == "__main__":
    unittest.main()
