import unittest

import pyrecest.backend
from pyrecest.backend import array, pi
from pyrecest.distributions import VonMisesFisherDistribution
from pyrecest.distributions.hypersphere_subset.hyperspherical_uniform_distribution import (
    HypersphericalUniformDistribution,
)


class HypersphereSubsetDistanceValidationTest(unittest.TestCase):
    def test_hellinger_distance_rejects_dimension_mismatch(self):
        dist = HypersphericalUniformDistribution(2)
        other = HypersphericalUniformDistribution(3)

        with self.assertRaisesRegex(ValueError, "different number of dimensions"):
            dist.hellinger_distance_numerical(other)

    def test_total_variation_distance_rejects_dimension_mismatch(self):
        dist = HypersphericalUniformDistribution(2)
        other = HypersphericalUniformDistribution(3)

        with self.assertRaisesRegex(ValueError, "different number of dimensions"):
            dist.total_variation_distance_numerical(other)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Numerical hyperspherical integration is not supported on JAX.",
    )
    def test_distances_respect_custom_integration_boundaries(self):
        dist = VonMisesFisherDistribution(array([1.0, 0.0]), 2.0)
        other = VonMisesFisherDistribution(array([0.0, 1.0]), 1.0)
        partial_boundaries = array([[0.0, pi / 2.0]])

        for distance_name in (
            "hellinger_distance_numerical",
            "total_variation_distance_numerical",
        ):
            with self.subTest(distance=distance_name):
                distance = getattr(dist, distance_name)
                full_distance = float(distance(other))
                partial_distance = float(
                    distance(other, integration_boundaries=partial_boundaries)
                )

                self.assertGreater(partial_distance, 0.0)
                self.assertLess(partial_distance, full_distance)


if __name__ == "__main__":
    unittest.main()
