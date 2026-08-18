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

    def test_higher_order_pdf_avoids_unnormalized_exponential_overflow(self):
        gvm = GvMDistribution(
            array([0.0, 0.0]),
            array([1000.0, 500.0]),
        )
        xs = array([0.0, 0.01, 0.1])

        with np.errstate(over="raise", divide="raise", invalid="raise"):
            actual = np.asarray(gvm.pdf(xs), dtype=float)

        self.assertTrue(np.all(np.isfinite(actual)))
        self.assertTrue(np.all(actual >= 0.0))
        self.assertGreater(actual[0], actual[-1])

        from scipy.integrate import quad

        integral, _ = quad(
            lambda x: float(gvm.pdf(array([x]))[0]),
            0.0,
            2.0 * np.pi,
        )
        npt.assert_allclose(integral, 1.0, rtol=1.0e-8, atol=1.0e-10)


if __name__ == "__main__":
    unittest.main()
