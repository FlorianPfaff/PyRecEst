import unittest

import pyrecest
from pyrecest.backend import ones
from pyrecest.distributions.conversion import convert_distribution
from pyrecest.distributions.hypersphere_subset.hyperspherical_uniform_distribution import (
    HypersphericalUniformDistribution,
)
from pyrecest.distributions.hypersphere_subset.spherical_grid_distribution import (
    SphericalGridDistribution,
)


class SphericalGridSubclass(SphericalGridDistribution):
    pass


class SphericalGridFactorySubclassPreservationTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Spherical grid construction is not supported on this backend",
    )
    def test_conversion_to_subclass_preserves_requested_type(self):
        source = HypersphericalUniformDistribution(2)

        converted = convert_distribution(
            source,
            SphericalGridSubclass,
            no_of_grid_points=42,
        )

        self.assertIs(type(converted), SphericalGridSubclass)
        self.assertEqual(converted.dim, 2)
        self.assertEqual(converted.input_dim, 3)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Spherical grid construction is not supported on this backend",
    )
    def test_inherited_from_function_preserves_requested_type(self):
        converted = SphericalGridSubclass.from_function(
            lambda xs: ones(xs.shape[0]),
            42,
        )

        self.assertIs(type(converted), SphericalGridSubclass)
        self.assertEqual(converted.dim, 2)
        self.assertEqual(converted.input_dim, 3)


if __name__ == "__main__":
    unittest.main()
