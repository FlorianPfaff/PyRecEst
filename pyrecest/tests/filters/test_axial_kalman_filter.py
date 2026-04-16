import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, eye, linalg
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters.abstract_axial_filter import (
    _complex_multiplication,
    _quaternion_multiplication,
)
from pyrecest.filters.axial_kalman_filter import AxialKalmanFilter


class TestAxialKalmanFilter4D(unittest.TestCase):
    def setUp(self):
        mu = array([1.0, 2.0, 3.0, 4.0])
        mu = mu / linalg.norm(mu)
        C = 0.3 * eye(4)
        self.mu = mu
        self.C = C
        self.filter = AxialKalmanFilter()

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="Not supported on this backend",  # pylint: disable=no-member
    )
    def test_set_state_and_get_estimate(self):
        self.filter.filter_state = GaussianDistribution(self.mu, self.C)
        est = self.filter.get_point_estimate()
        npt.assert_array_equal(self.mu, est)
        npt.assert_array_equal(self.C, self.filter.filter_state.C)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="Not supported on this backend",  # pylint: disable=no-member
    )
    def test_predict_identity_zero_mean(self):
        """Predicting with identity-rotation noise should not change the mean."""
        self.filter.filter_state = GaussianDistribution(self.mu, self.C)
        noise_mu = array([1.0, 0.0, 0.0, 0.0])
        self.filter.predict_identity(GaussianDistribution(noise_mu, 0.1 * eye(4)))
        est = self.filter.get_point_estimate()
        npt.assert_allclose(self.mu, est, atol=1e-10)
        # Covariance should increase
        self.assertTrue(
            (self.filter.filter_state.C >= self.C).all()
            if hasattr((self.filter.filter_state.C >= self.C), "all")
            else all(self.filter.filter_state.C.flatten() >= self.C.flatten())
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="Not supported on this backend",  # pylint: disable=no-member
    )
    def test_predict_identity_nonzero_mean(self):
        """Predicting with non-identity rotation noise updates the mean correctly."""
        self.filter.filter_state = GaussianDistribution(self.mu, self.C)
        self.filter.predict_identity(GaussianDistribution(self.mu, 0.1 * eye(4)))
        est = self.filter.get_point_estimate()
        expected = _quaternion_multiplication(self.mu, self.mu)
        npt.assert_allclose(est, expected, atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="Not supported on this backend",  # pylint: disable=no-member
    )
    def test_update_identity_at_mode(self):
        """Updating with z=mu and identity noise should keep the mean and reduce C."""
        self.filter.filter_state = GaussianDistribution(self.mu, self.C)
        z = self.mu
        self.filter.update_identity(
            GaussianDistribution(array([1.0, 0.0, 0.0, 0.0]), self.C), z
        )
        est = self.filter.get_point_estimate()
        npt.assert_allclose(self.mu, est, atol=1e-6)
        # Covariance should decrease
        self.assertTrue(
            (self.filter.filter_state.C <= self.C).all()
            if hasattr((self.filter.filter_state.C <= self.C), "all")
            else all(self.filter.filter_state.C.flatten() <= self.C.flatten())
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="Not supported on this backend",  # pylint: disable=no-member
    )
    def test_update_identity_antipodal_symmetry(self):
        """z and -z (antipodal) should produce the same result."""
        self.filter.filter_state = GaussianDistribution(self.mu, self.C)
        noise = GaussianDistribution(array([1.0, 0.0, 0.0, 0.0]), self.C)
        self.filter.update_identity(noise, self.mu)
        mu4 = self.filter.get_point_estimate()
        C4 = self.filter.filter_state.C

        self.filter.filter_state = GaussianDistribution(self.mu, self.C)
        self.filter.update_identity(noise, -self.mu)
        mu5 = self.filter.get_point_estimate()
        C5 = self.filter.filter_state.C

        npt.assert_allclose(mu4, mu5, atol=1e-10)
        npt.assert_allclose(C4, C5, atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="Not supported on this backend",  # pylint: disable=no-member
    )
    def test_update_identity_unit_norm(self):
        """After updating with a non-mode measurement the mean is a unit vector."""
        self.filter.filter_state = GaussianDistribution(self.mu, self.C)
        z = array([0.0, 0.0, 0.0, 1.0])
        self.filter.update_identity(GaussianDistribution(self.mu, self.C), z)
        est = self.filter.get_point_estimate()
        npt.assert_allclose(linalg.norm(est), 1.0, atol=1e-6)


class TestAxialKalmanFilter2D(unittest.TestCase):
    def setUp(self):
        mu = array([1.0, 2.0])
        mu = mu / linalg.norm(mu)
        C = 0.3 * eye(2)
        self.mu = mu
        self.C = C
        self.filter = AxialKalmanFilter()

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="Not supported on this backend",  # pylint: disable=no-member
    )
    def test_set_state_and_get_estimate(self):
        self.filter.filter_state = GaussianDistribution(self.mu, self.C)
        est = self.filter.get_point_estimate()
        npt.assert_array_equal(self.mu, est)
        npt.assert_array_equal(self.C, self.filter.filter_state.C)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="Not supported on this backend",  # pylint: disable=no-member
    )
    def test_predict_identity_zero_mean(self):
        """Predicting with identity-rotation noise should not change the mean."""
        self.filter.filter_state = GaussianDistribution(self.mu, self.C)
        noise_mu = array([1.0, 0.0])
        self.filter.predict_identity(GaussianDistribution(noise_mu, 0.1 * eye(2)))
        est = self.filter.get_point_estimate()
        npt.assert_allclose(self.mu, est, atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="Not supported on this backend",  # pylint: disable=no-member
    )
    def test_predict_identity_nonzero_mean(self):
        """Predicting with non-identity rotation noise updates the mean correctly."""
        self.filter.filter_state = GaussianDistribution(self.mu, self.C)
        self.filter.predict_identity(GaussianDistribution(self.mu, 0.1 * eye(2)))
        est = self.filter.get_point_estimate()
        expected = _complex_multiplication(self.mu, self.mu)
        npt.assert_allclose(est, expected, atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="Not supported on this backend",  # pylint: disable=no-member
    )
    def test_update_identity_at_mode(self):
        """Updating with z=mu and identity noise should keep the mean and reduce C."""
        self.filter.filter_state = GaussianDistribution(self.mu, self.C)
        z = self.mu
        self.filter.update_identity(GaussianDistribution(array([1.0, 0.0]), self.C), z)
        est = self.filter.get_point_estimate()
        npt.assert_allclose(self.mu, est, atol=1e-6)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="Not supported on this backend",  # pylint: disable=no-member
    )
    def test_update_identity_antipodal_symmetry(self):
        """z and -z (antipodal) should produce the same result."""
        self.filter.filter_state = GaussianDistribution(self.mu, self.C)
        noise = GaussianDistribution(array([1.0, 0.0]), self.C)
        self.filter.update_identity(noise, self.mu)
        mu4 = self.filter.get_point_estimate()
        C4 = self.filter.filter_state.C

        self.filter.filter_state = GaussianDistribution(self.mu, self.C)
        self.filter.update_identity(noise, -self.mu)
        mu5 = self.filter.get_point_estimate()
        C5 = self.filter.filter_state.C

        npt.assert_allclose(mu4, mu5, atol=1e-10)
        npt.assert_allclose(C4, C5, atol=1e-10)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "pytorch",
        reason="Not supported on this backend",  # pylint: disable=no-member
    )
    def test_update_identity_unit_norm(self):
        """After updating with a non-mode measurement the mean is a unit vector."""
        self.filter.filter_state = GaussianDistribution(self.mu, self.C)
        z = array([0.0, 1.0])
        self.filter.update_identity(GaussianDistribution(self.mu, self.C), z)
        est = self.filter.get_point_estimate()
        npt.assert_allclose(linalg.norm(est), 1.0, atol=1e-10)


if __name__ == "__main__":
    unittest.main()
