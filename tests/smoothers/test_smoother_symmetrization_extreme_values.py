import unittest

import numpy as np
import numpy.testing as npt

from pyrecest.smoothers.abstract_smoother import AbstractSmoother


class SmootherSymmetrizationExtremeValuesTest(unittest.TestCase):
    def test_preserves_extreme_finite_symmetric_covariance(self):
        covariance = np.diag([1e308, 2e307])

        with np.errstate(over="raise", invalid="raise"):
            symmetrized = AbstractSmoother._symmetrize(covariance)

        npt.assert_array_equal(symmetrized, covariance)
        self.assertTrue(np.isfinite(symmetrized).all())


if __name__ == "__main__":
    unittest.main()
