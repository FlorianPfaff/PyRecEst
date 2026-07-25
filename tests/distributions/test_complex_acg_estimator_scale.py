import unittest

import numpy.testing as npt
import pyrecest.backend

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, complex128, eye, real
from pyrecest.distributions import ComplexAngularCentralGaussianDistribution


@unittest.skipIf(
    pyrecest.backend.__backend_name__ == "jax",
    reason="Not supported on JAX backend",
)
class TestComplexAcgEstimatorScale(unittest.TestCase):
    def test_orthonormal_samples_do_not_contract_parameter_scale(self):
        samples = eye(2, dtype=complex128)

        estimated = ComplexAngularCentralGaussianDistribution.estimate_parameter_matrix(
            samples, n_iterations=100
        )

        npt.assert_allclose(real(estimated), eye(2), atol=1e-12)

    def test_fit_supports_one_complex_dimension(self):
        samples = array([[1.0 + 0.0j], [1.0j]])

        fitted = ComplexAngularCentralGaussianDistribution.fit(samples, n_iterations=3)

        self.assertEqual(fitted.dim, 1)
        npt.assert_allclose(real(fitted.C), eye(1), atol=1e-12)


if __name__ == "__main__":
    unittest.main()
