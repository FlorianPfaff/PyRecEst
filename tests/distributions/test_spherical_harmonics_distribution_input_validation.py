import unittest

import pyrecest.backend
from pyrecest.distributions.hypersphere_subset.hyperspherical_uniform_distribution import (
    HypersphericalUniformDistribution,
)
from pyrecest.distributions.hypersphere_subset.spherical_harmonics_distribution_complex import (
    SphericalHarmonicsDistributionComplex,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
    reason="Spherical harmonics distributions are not supported on JAX.",
)
class SphericalHarmonicsDistributionInputValidationTest(unittest.TestCase):
    def test_from_distribution_via_integral_rejects_non_s2_distribution(self):
        circle_dist = HypersphericalUniformDistribution(1)

        with self.assertRaisesRegex(ValueError, "sphere"):
            SphericalHarmonicsDistributionComplex.from_distribution_via_integral(
                circle_dist, degree=0
            )


if __name__ == "__main__":
    unittest.main()
