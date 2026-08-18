import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, eye, sqrt
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.abstract_axial_filter import _quaternion_multiplication
from pyrecest.filters.axial_kalman_filter import AxialKalmanFilter


class TestAxialKalmanFilterQuaternionMeasurementOrder(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="Not supported on this backend",  # pylint: disable=no-member
    )
    def test_update_removes_right_composed_quaternion_noise(self):
        """A measurement z=x*v must be corrected as z*inv(v), not inv(v)*z."""
        root_half = sqrt(0.5)
        state_mean = array([root_half, root_half, 0.0, 0.0])
        noise_mean = array([root_half, 0.0, root_half, 0.0])
        measurement = _quaternion_multiplication(state_mean, noise_mean)
        covariance = 0.3 * eye(4)

        filt = AxialKalmanFilter()
        filt.filter_state = GaussianDistribution(state_mean, covariance)
        filt.update_identity(
            GaussianDistribution(noise_mean, covariance),
            measurement,
        )

        npt.assert_allclose(filt.get_point_estimate(), state_mean, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
