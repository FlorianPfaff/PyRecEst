import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.axial_kalman_filter import AxialKalmanFilter


class TestAxialKalmanFilterComplexMeasurement(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="AxialKalmanFilter is not supported on this backend",
    )
    def test_update_rejects_complex_measurement_without_mutating_state(self):
        current_filter = AxialKalmanFilter()
        mean = array([1.0, 0.0, 0.0, 0.0])
        covariance = 0.3 * eye(4)
        current_filter.filter_state = GaussianDistribution(mean, covariance)
        measurement_noise = GaussianDistribution(mean, covariance)

        with self.assertRaisesRegex(ValueError, "real-valued"):
            current_filter.update_identity(
                measurement_noise,
                [1.0 + 1.0j, 0.0, 0.0, 0.0],
            )

        npt.assert_allclose(current_filter.filter_state.mu, mean)
        npt.assert_allclose(current_filter.filter_state.C, covariance)


if __name__ == "__main__":
    unittest.main()
