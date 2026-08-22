import unittest

import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import ones
from pyrecest.distributions.conversion import convert_distribution
from pyrecest.distributions.hypersphere_subset.hyperspherical_uniform_distribution import (
    HypersphericalUniformDistribution,
)
from pyrecest.distributions.hypersphere_subset.spherical_grid_distribution import (
    SphericalGridDistribution,
)


class _SphericalGridSubclass(SphericalGridDistribution):
    pass


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",
    reason="LeopardiSampler is not supported on the JAX backend",
)
class SphericalGridFactorySubclassPreservationTest(unittest.TestCase):
    def test_conversion_factory_preserves_requested_subclass(self):
        source = HypersphericalUniformDistribution(2)

        converted = convert_distribution(
            source, _SphericalGridSubclass, no_of_grid_points=12
        )

        self.assertIsInstance(converted, _SphericalGridSubclass)

    def test_from_function_preserves_requested_subclass(self):
        converted = _SphericalGridSubclass.from_function(
            lambda xs: ones(xs.shape[0]), no_of_grid_points=12
        )

        self.assertIsInstance(converted, _SphericalGridSubclass)


if __name__ == "__main__":
    unittest.main()
