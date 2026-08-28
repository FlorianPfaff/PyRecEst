"""Regression tests for numerical hyperspherical moments."""

import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.distributions import HypersphericalUniformDistribution


class HypersphereMomentJacobianTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        "Numerical hyperspherical integration is not supported on JAX.",
    )
    def test_uniform_s2_second_moment_uses_surface_jacobian_once(self):
        """Uniform S2 has E[x x^T] = I/3 and therefore unit trace."""
        dist = HypersphericalUniformDistribution(2)

        moment = dist.moment_numerical()

        npt.assert_allclose(moment, np.eye(3) / 3.0, atol=1e-6)
        npt.assert_allclose(np.trace(moment), 1.0, atol=1e-6)


if __name__ == "__main__":
    unittest.main()
