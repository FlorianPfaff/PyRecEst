import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, diag
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.unscented_kalman_filter import UnscentedKalmanFilter


class UnscentedKalmanFilterProcessNoiseShapeTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_vector_process_covariance_is_rejected_without_state_mutation(self):
        initial_mean = array([0.5, -0.25])
        initial_covariance = diag(array([1.2, 0.8]))
        ukf = UnscentedKalmanFilter(
            GaussianDistribution(initial_mean, initial_covariance)
        )

        with self.assertRaisesRegex(
            ValueError,
            r"process noise covariance Q has shape .* expected \(2, 2\)",
        ):
            ukf.predict_identity(array([0.4, 0.2]))

        npt.assert_allclose(ukf.get_point_estimate(), initial_mean)
        npt.assert_allclose(ukf.filter_state.covariance(), initial_covariance)


if __name__ == "__main__":
    unittest.main()
