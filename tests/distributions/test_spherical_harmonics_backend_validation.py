import unittest
from unittest.mock import patch

import numpy as np
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.distributions.hypersphere_subset.spherical_harmonics_distribution_complex import (
    SphericalHarmonicsDistributionComplex,
)


class SphericalHarmonicsBackendValidationTest(unittest.TestCase):
    def test_constructor_rejects_jax_backend(self):
        with patch.object(pyrecest.backend, "__backend_name__", "jax"):
            with self.assertRaisesRegex(NotImplementedError, "JAX backend"):
                SphericalHarmonicsDistributionComplex(array([[1.0]]))

    def test_fit_from_grid_rejects_non_numpy_backend(self):
        with patch.object(pyrecest.backend, "__backend_name__", "pytorch"):
            with self.assertRaisesRegex(NotImplementedError, "numpy backend"):
                SphericalHarmonicsDistributionComplex._fit_from_grid(
                    array([[1.0]]), degree=0, transformation="sqrt"
                )

    def test_sqrt_convolve_rejects_non_numpy_backend(self):
        dist = SphericalHarmonicsDistributionComplex.__new__(
            SphericalHarmonicsDistributionComplex
        )
        dist.coeff_mat = np.array([[1.0]])
        dist.transformation = "sqrt"

        with patch.object(pyrecest.backend, "__backend_name__", "pytorch"):
            with self.assertRaisesRegex(NotImplementedError, "numpy backend"):
                dist.convolve(dist)


if __name__ == "__main__":
    unittest.main()
