import unittest

import pyrecest.backend
from pyrecest.backend import array, pi, sqrt
from pyrecest.distributions.hypersphere_subset.spherical_harmonics_distribution_complex import (
    SphericalHarmonicsDistributionComplex,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
    reason="Spherical harmonics distributions are not supported on JAX.",
)
class SphericalHarmonicsComplexValidationTest(unittest.TestCase):
    def test_multiply_rejects_invalid_type(self):
        dist = SphericalHarmonicsDistributionComplex(array([[1.0 / sqrt(4.0 * pi)]]))

        with self.assertRaisesRegex(TypeError, "SphericalHarmonicsDistributionComplex"):
            dist.multiply(object())

    def test_multiply_rejects_mixed_transformations(self):
        identity_dist = SphericalHarmonicsDistributionComplex(
            array([[1.0 / sqrt(4.0 * pi)]]), "identity"
        )
        sqrt_dist = SphericalHarmonicsDistributionComplex(array([[1.0]]), "sqrt")

        with self.assertRaisesRegex(ValueError, "same transformation"):
            identity_dist.multiply(sqrt_dist)


if __name__ == "__main__":
    unittest.main()
