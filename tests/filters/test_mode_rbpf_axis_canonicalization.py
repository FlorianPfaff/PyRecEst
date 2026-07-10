"""Regression tests for mode-RBPF extent canonicalization."""

import unittest

import numpy as np
import numpy.testing as npt
import pyrecest.backend as pyrecest_backend
from pyrecest.filters import ModeRBPFManifoldUKFTracker


@unittest.skipIf(
    pyrecest_backend.__backend_name__ != "numpy",
    reason="ModeRBPFManifoldUKFTracker is currently NumPy-backend only",
)
class TestModeRBPFAxisCanonicalization(unittest.TestCase):
    def test_axis_swap_applies_covariance_congruence_permutation(self):
        tracker = ModeRBPFManifoldUKFTracker(
            np.zeros(4),
            np.eye(4),
            np.array([0.2, 2.0, 1.0]),
            np.eye(3),
            n_particles=2,
            rng=0,
            canonicalize_extent=False,
        )
        tracker.canonicalize_extent = True
        tracker.mu[:, 5:7] = np.log([[1.0, 2.0], [2.0, 1.0]])

        covariance = np.diag([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        covariance[0, 5] = covariance[5, 0] = 0.5
        covariance[0, 6] = covariance[6, 0] = 0.75
        covariance[5, 6] = covariance[6, 5] = 0.25
        tracker.covariances[0] = covariance
        tracker.covariances[1] = 2.0 * covariance
        untouched_covariance = tracker.covariances[1].copy()
        untouched_angle = tracker.mu[1, 4]

        permutation = np.array([0, 1, 2, 3, 4, 6, 5])
        expected_covariance = covariance[np.ix_(permutation, permutation)]
        expected_angle = (tracker.mu[0, 4] + np.pi / 2.0) % np.pi

        tracker._canonicalize_particles()

        npt.assert_allclose(np.exp(tracker.mu[0, 5:7]), [2.0, 1.0])
        npt.assert_allclose(tracker.covariances[0], expected_covariance)
        npt.assert_allclose(tracker.covariances[0], tracker.covariances[0].T)
        npt.assert_allclose(tracker.mu[0, 4], expected_angle)
        npt.assert_allclose(tracker.covariances[1], untouched_covariance)
        npt.assert_allclose(tracker.mu[1, 4], untouched_angle)


if __name__ == "__main__":
    unittest.main()
