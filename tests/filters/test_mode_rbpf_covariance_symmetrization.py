"""Regression tests for overflow-safe Mode-RBPF covariance symmetrization."""

import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend as pyrecest_backend
from pyrecest.filters import ModeRBPFManifoldUKFTracker


@unittest.skipIf(
    pyrecest_backend.__backend_name__ != "numpy",
    reason="ModeRBPFManifoldUKFTracker is currently NumPy-backend only",
)
class TestModeRBPFCovarianceSymmetrization(unittest.TestCase):
    def setUp(self):
        self.covariance = np.diag([1.0e308, 2.0e307])

    def test_extreme_covariance_validation_does_not_overflow(self):
        with np.errstate(over="raise", invalid="raise"):
            validated = ModeRBPFManifoldUKFTracker._as_covariance(
                self.covariance,
                2,
                "covariance",
            )

        npt.assert_allclose(validated, self.covariance, rtol=0.0, atol=0.0)
        self.assertTrue(np.all(np.isfinite(validated)))

    def test_stabilization_and_logpdf_do_not_overflow(self):
        tracker = object.__new__(ModeRBPFManifoldUKFTracker)
        tracker.minimum_covariance_eigenvalue = 0.0

        with np.errstate(over="raise", invalid="raise", divide="raise"):
            stabilized = tracker._stabilize_covariance(
                self.covariance,
                floor=0.0,
            )
            log_pdf = tracker._gaussian_logpdf(
                np.zeros(2),
                self.covariance,
            )

        npt.assert_allclose(stabilized, self.covariance, rtol=1.0e-15, atol=0.0)
        self.assertTrue(np.all(np.isfinite(stabilized)))
        self.assertTrue(np.isfinite(log_pdf))


if __name__ == "__main__":
    unittest.main()
