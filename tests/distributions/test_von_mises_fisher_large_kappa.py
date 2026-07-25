import math
import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, to_numpy
from pyrecest.distributions import VonMisesFisherDistribution
from scipy.special import ive


class TestVonMisesFisherLargeKappa(unittest.TestCase):
    def test_pdf_remains_finite_at_mode_for_large_kappa(self):
        kappa = 1000.0

        for input_dim in (3, 4):
            with self.subTest(input_dim=input_dim):
                mu_np = np.zeros(input_dim, dtype=float)
                mu_np[0] = 1.0
                distribution = VonMisesFisherDistribution(array(mu_np), kappa)

                points = array(np.stack((mu_np, -mu_np)))
                densities = np.asarray(to_numpy(distribution.pdf(points)), dtype=float)

                order = input_dim / 2.0 - 1.0
                expected_mode_density = kappa**order / (
                    (2.0 * math.pi) ** (input_dim / 2.0) * ive(order, kappa)
                )

                self.assertTrue(np.all(np.isfinite(densities)))
                self.assertGreater(densities[0], 0.0)
                self.assertGreaterEqual(densities[1], 0.0)
                self.assertLess(densities[1], densities[0])
                npt.assert_allclose(
                    densities[0], expected_mode_density, rtol=1.0e-7, atol=0.0
                )


if __name__ == "__main__":
    unittest.main()
