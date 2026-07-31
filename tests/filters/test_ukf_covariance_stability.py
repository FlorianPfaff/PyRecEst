import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, float64
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.unscented_kalman_filter import UnscentedKalmanFilter


class UnscentedKalmanFilterCovarianceStabilityTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_predict_preserves_extreme_finite_process_covariance(self):
        largest = np.finfo(np.float64).max
        ukf = UnscentedKalmanFilter(
            GaussianDistribution(
                array([0.0], dtype=float64),
                array([[1.0]], dtype=float64),
            )
        )

        with np.errstate(over="raise", invalid="raise"):
            ukf.predict_identity(array([[largest]], dtype=float64))

        covariance = ukf.filter_state.covariance()
        self.assertTrue(np.isfinite(covariance).all())
        npt.assert_array_equal(covariance, array([[largest]], dtype=float64))


if __name__ == "__main__":
    unittest.main()
