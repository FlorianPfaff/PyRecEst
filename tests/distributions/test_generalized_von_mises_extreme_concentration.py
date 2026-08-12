import unittest

import numpy as np
import numpy.testing as npt

import pyrecest.backend
from pyrecest.backend import array
from pyrecest.distributions import GvMDistribution, VonMisesDistribution


@unittest.skipUnless(
    pyrecest.backend.__backend_name__ == "numpy",
    reason="Strict NumPy floating-point regression",
)
class TestGvMExtremeConcentration(unittest.TestCase):
    def test_order_one_pdf_avoids_unnormalized_exponential_overflow(self):
        mu = np.pi
        kappa = 1000.0
        gvm = GvMDistribution(array([mu]), array([kappa]))
        vm = VonMisesDistribution(mu, kappa)
        xs = array([mu, mu + 0.1])

        with np.errstate(over="raise", divide="raise", invalid="raise"):
            actual = np.asarray(gvm.pdf(xs), dtype=float)
            expected = np.asarray(vm.pdf(xs), dtype=float)

        self.assertTrue(np.all(np.isfinite(actual)))
        npt.assert_allclose(actual, expected, rtol=1.0e-12, atol=0.0)


if __name__ == "__main__":
    unittest.main()
