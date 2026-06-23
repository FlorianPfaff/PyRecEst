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
class SphericalHarmonicsConvolutionValidationTest(unittest.TestCase):
    def test_convolve_rejects_invalid_type(self):
        dist = SphericalHarmonicsDistributionComplex(array([[1.0 / sqrt(4.0 * pi)]]))

        with self.assertRaisesRegex(TypeError, "SphericalHarmonicsDistributionComplex"):
            dist.convolve(object())


if __name__ == "__main__":
    unittest.main()
