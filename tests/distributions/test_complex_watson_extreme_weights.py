# pylint: disable=no-name-in-module,no-member
import unittest

import numpy as np
import pyrecest.backend
from pyrecest.backend import abs, allclose, array, complex128, conj, float64, sum
from pyrecest.distributions import ComplexWatsonDistribution


class TestComplexWatsonExtremeWeights(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",  # pylint: disable=no-member
        reason="Complex Watson parameter estimation is not tested on JAX",
    )
    def test_weight_rescaling_remains_finite_and_invariant(self):
        inv_sqrt_two = 1.0 / np.sqrt(2.0)
        samples = array(
            [
                [1.0 + 0.0j, 0.0 + 0.0j],
                [0.0 + 0.0j, 1.0 + 0.0j],
                [inv_sqrt_two + 0.0j, inv_sqrt_two + 0.0j],
                [inv_sqrt_two + 0.0j, 1j * inv_sqrt_two],
            ],
            dtype=complex128,
        )
        reference_weights = array([8.0, 4.0, 2.0, 1.0], dtype=float64)
        extreme_weights = reference_weights * (np.finfo(np.float64).max / 8.0)

        reference_mu, reference_kappa = ComplexWatsonDistribution.estimate_parameters(
            samples, weights=reference_weights
        )
        extreme_mu, extreme_kappa = ComplexWatsonDistribution.estimate_parameters(
            samples, weights=extreme_weights
        )

        phase_invariant_overlap = abs(sum(conj(reference_mu) * extreme_mu))
        self.assertTrue(
            bool(allclose(phase_invariant_overlap, 1.0, rtol=1e-12, atol=1e-12))
        )
        self.assertTrue(
            bool(allclose(extreme_kappa, reference_kappa, rtol=1e-12, atol=1e-12))
        )


if __name__ == "__main__":
    unittest.main()
