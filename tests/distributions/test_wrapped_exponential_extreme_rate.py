import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array
from pyrecest.distributions.circle.wrapped_exponential_distribution import (
    WrappedExponentialDistribution,
)


@unittest.skipUnless(
    pyrecest.backend.__backend_name__ == "numpy",
    "Extreme binary64 regression is specific to the NumPy backend.",
)
class WrappedExponentialExtremeRateTest(unittest.TestCase):
    def test_pdf_and_entropy_remain_finite_for_extreme_rate(self):
        lambda_ = 1.0e308

        with np.errstate(over="raise", invalid="raise"):
            distribution = WrappedExponentialDistribution(array(lambda_))
            density_at_zero = np.asarray(
                pyrecest.backend.to_numpy(distribution.pdf(array(0.0)))
            )
            entropy = np.asarray(pyrecest.backend.to_numpy(distribution.entropy()))

        self.assertTrue(np.isfinite(density_at_zero).all())
        self.assertTrue(np.isfinite(entropy).all())
        npt.assert_allclose(density_at_zero, lambda_, rtol=5e-7)
        npt.assert_allclose(entropy, 1.0 - np.log(lambda_), rtol=5e-7)


if __name__ == "__main__":
    unittest.main()
