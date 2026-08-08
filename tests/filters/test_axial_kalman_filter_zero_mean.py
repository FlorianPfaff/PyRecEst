import unittest

import numpy.testing as npt

import pyrecest.backend
from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.axial_kalman_filter import AxialKalmanFilter


class TestAxialKalmanFilterZeroMean(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="Not supported on this backend",  # pylint: disable=no-member
    )
    def test_update_rejects_zero_length_posterior_mean_atomically(self):
        inv_sqrt_two = 2.0**-0.5
        state_cov = array([[5.0, 2.0], [2.0, 1.0]])
        noise_cov = array(
            [
                [2.0 - inv_sqrt_two, 1.0],
                [1.0, (1.0 + inv_sqrt_two) / 2.0],
            ]
        )

        axial_filter = AxialKalmanFilter()
        axial_filter.filter_state = GaussianDistribution(
            array([1.0, 0.0]), state_cov
        )
        prior_mu = axial_filter.filter_state.mu.copy()
        prior_cov = axial_filter.filter_state.C.copy()
        noise = GaussianDistribution(array([1.0, 0.0]), noise_cov)
        measurement = array([inv_sqrt_two, inv_sqrt_two])

        with self.assertRaisesRegex(ValueError, "zero-length posterior mean"):
            axial_filter.update_identity(noise, measurement)

        npt.assert_array_equal(axial_filter.filter_state.mu, prior_mu)
        npt.assert_array_equal(axial_filter.filter_state.C, prior_cov)


if __name__ == "__main__":
    unittest.main()
