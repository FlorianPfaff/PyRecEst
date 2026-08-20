import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions.conversion import convert_distribution
from pyrecest.distributions.hypersphere_subset.hyperhemispherical_grid_distribution import (
    HyperhemisphericalGridDistribution,
)
from pyrecest.distributions.hypersphere_subset.hyperspherical_grid_distribution import (
    HypersphericalGridDistribution,
)
from pyrecest.distributions.hypersphere_subset.spherical_grid_distribution import (
    SphericalGridDistribution,
)
from pyrecest.distributions.hypersphere_subset.von_mises_fisher_distribution import (
    VonMisesFisherDistribution,
)
from pyrecest.distributions.hypersphere_subset.watson_distribution import (
    WatsonDistribution,
)


class _HypersphericalGridSubclass(HypersphericalGridDistribution):
    pass


class _HyperhemisphericalGridSubclass(HyperhemisphericalGridDistribution):
    pass


class _SphericalGridSubclass(SphericalGridDistribution):
    pass


class GridFactorySubclassPreservationTest(unittest.TestCase):
    def test_hyperspherical_grid_conversion_preserves_requested_subclass(self):
        source = VonMisesFisherDistribution(array([0.0, 0.0, 1.0]), 1.0)

        converted = convert_distribution(
            source, _HypersphericalGridSubclass, no_of_grid_points=12
        )

        self.assertIsInstance(converted, _HypersphericalGridSubclass)

    def test_hyperhemispherical_grid_conversion_preserves_requested_subclass(self):
        source = WatsonDistribution(array([0.0, 0.0, 1.0]), 1.0)

        converted = convert_distribution(
            source, _HyperhemisphericalGridSubclass, no_of_grid_points=12
        )

        self.assertIsInstance(converted, _HyperhemisphericalGridSubclass)

    def test_spherical_grid_conversion_preserves_requested_subclass(self):
        source = VonMisesFisherDistribution(array([0.0, 0.0, 1.0]), 1.0)

        converted = convert_distribution(
            source, _SphericalGridSubclass, no_of_grid_points=12
        )

        self.assertIsInstance(converted, _SphericalGridSubclass)


if __name__ == "__main__":
    unittest.main()
