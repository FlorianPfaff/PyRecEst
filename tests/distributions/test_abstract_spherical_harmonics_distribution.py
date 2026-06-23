import unittest

import pyrecest.backend
from pyrecest.backend import array
from pyrecest.distributions.hypersphere_subset.spherical_harmonics_distribution_complex import (
    SphericalHarmonicsDistributionComplex,
)
from pyrecest.distributions.hypersphere_subset.spherical_harmonics_distribution_real import (
    SphericalHarmonicsDistributionReal,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
    reason="Spherical harmonics distributions are not supported on JAX.",
)
class AbstractSphericalHarmonicsDistributionTest(unittest.TestCase):
    def test_constructor_rejects_incompatible_coefficient_matrix_size(self):
        with self.assertRaisesRegex(ValueError, "CoefficientMatrix"):
            SphericalHarmonicsDistributionReal(array([[1.0, 0.0], [0.0, 0.0]]))

    def test_constructor_rejects_complex_integral(self):
        with self.assertRaisesRegex(ValueError, "imaginary"):
            SphericalHarmonicsDistributionComplex(
                array([[1.0 + 1.0j]]), assert_real=False
            )


if __name__ == "__main__":
    unittest.main()
