import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member,protected-access
import pyrecest.backend
from pyrecest.backend import array, diag, isfinite
from pyrecest.filters import GGIWTracker


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="GGIW tracker tests currently use numpy.testing assertions",
)
class TestGGIWLogLikelihoodStability(unittest.TestCase):
    def setUp(self):
        self.tracker = GGIWTracker(
            kinematic_state=array([0.0, 0.0]),
            covariance=diag(array([1.0, 1.0])),
            extent=diag(array([1.0, 1.0])),
            extent_degrees_of_freedom=8.0,
            gamma_shape=1.0,
            gamma_rate=1.0,
        )

    def test_valid_covariance_with_underflowing_determinant_is_finite(self):
        covariance = diag(array([1e-200, 1e-200]))
        innovation = array([0.0, 0.0])

        self.assertEqual(float(np.linalg.det(np.asarray(covariance))), 0.0)

        log_likelihood = self.tracker._gaussian_log_likelihood(
            innovation, covariance
        )
        expected = 200.0 * np.log(10.0) - np.log(2.0 * np.pi)

        self.assertTrue(bool(isfinite(log_likelihood)))
        npt.assert_allclose(float(log_likelihood), expected, rtol=1e-12)

    def test_positive_determinant_does_not_accept_indefinite_covariance(self):
        covariance = diag(array([-1.0, -1.0]))
        innovation = array([0.0, 0.0])

        self.assertEqual(float(np.linalg.det(np.asarray(covariance))), 1.0)

        log_likelihood = self.tracker._gaussian_log_likelihood(
            innovation, covariance
        )

        self.assertTrue(np.isneginf(float(log_likelihood)))


if __name__ == "__main__":
    unittest.main()
