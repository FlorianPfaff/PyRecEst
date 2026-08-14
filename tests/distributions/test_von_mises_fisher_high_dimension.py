import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, to_numpy
from pyrecest.distributions import VonMisesFisherDistribution
from scipy.special import ive


class TestVonMisesFisherHighDimension(unittest.TestCase):
    def test_mean_resultant_survives_scaled_bessel_underflow(self):
        input_dim = 1000
        kappa = 100.0
        order = input_dim / 2.0 - 1.0
        self.assertEqual(float(ive(order, kappa)), 0.0)
        self.assertEqual(float(ive(order + 1.0, kappa)), 0.0)

        mu = np.zeros(input_dim, dtype=float)
        mu[0] = 1.0
        distribution = VonMisesFisherDistribution(array(mu), kappa)
        resultant = np.asarray(
            to_numpy(distribution.mean_resultant_vector()), dtype=float
        )

        self.assertTrue(np.all(np.isfinite(resultant)))
        npt.assert_allclose(
            resultant[0],
            0.099021395665281644,
            rtol=2.0e-13,
            atol=0.0,
        )
        npt.assert_allclose(resultant[1:], 0.0, atol=0.0)

    def test_high_dimensional_mean_resultant_round_trip(self):
        mean_resultant = np.zeros(1000, dtype=float)
        mean_resultant[0] = 0.1

        distribution = VonMisesFisherDistribution.from_mean_resultant_vector(
            array(mean_resultant)
        )
        recovered = np.asarray(
            to_numpy(distribution.mean_resultant_vector()), dtype=float
        )

        self.assertTrue(np.isfinite(distribution.kappa))
        self.assertGreater(distribution.kappa, 0.0)
        npt.assert_allclose(recovered, mean_resultant, rtol=1.0e-11, atol=1.0e-13)


if __name__ == "__main__":
    unittest.main()
