import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member,protected-access
from pyrecest.backend import array, diag
from tests.filters.spline_tracker_test_cases import SplineTrackerCommonTests


class TestUKFSplineTracker(SplineTrackerCommonTests, unittest.TestCase):
    tracker_key = "ukf"

    def test_sigma_point_weights_are_normalized(self):
        mean_weights, _, spread = self.tracker._sigma_point_weights(self.tracker.state_dim)

        npt.assert_allclose(np.sum(mean_weights), 1.0, atol=1e-12)
        self.assertGreater(spread, 0.0)

    def test_orientation_correction_can_be_disabled(self):
        tracker = self.make_tracker(
            covariance=diag(array([0.05, 0.05, 0.2, 0.01, 0.01, 0.05, 0.05])),
            orientation_correction=False,
        )

        tracker.update(array([2.5, 0.7]))

        npt.assert_allclose(tracker.kinematic_state[2], 0.0, atol=1e-10)
