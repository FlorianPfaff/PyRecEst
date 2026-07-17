import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend
from pyrecest.backend import array, to_numpy
from pyrecest.distributions.circle.wrapped_exponential_distribution import (
    WrappedExponentialDistribution,
)


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Subnormal float64 regression is specific to the NumPy backend.",
)
class WrappedExponentialTinyRateTest(unittest.TestCase):
    def test_pdf_stays_finite_for_subnormal_positive_rate(self):
        distribution = WrappedExponentialDistribution(array(1.0e-310))
        density = to_numpy(distribution.pdf(array([0.0, np.pi])))

        self.assertTrue(np.isfinite(density).all())
        npt.assert_allclose(
            density,
            np.full(2, 1.0 / (2.0 * np.pi)),
            rtol=1.0e-14,
            atol=0.0,
        )


if __name__ == "__main__":
    unittest.main()
