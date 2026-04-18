import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, eye, log
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import KalmanFilter
from pyrecest.filters.multi_hypothesis_tracker import MultiHypothesisTracker


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Currently supported for numpy backend only",
)
class MultiHypothesisTrackerTest(unittest.TestCase):
    def setUp(self):
        self.kfs_init = [
            KalmanFilter(GaussianDistribution(array([0.0, 0.0]), eye(2))),
            KalmanFilter(GaussianDistribution(array([10.0, 0.0]), eye(2))),
        ]
        self.meas_mat = eye(2)
        self.meas_noise = eye(2)
        self.association_param = {
            "gating_distance_threshold": 1.0e6,
            "max_global_hypotheses": 5,
            "max_hypotheses_per_global_hypothesis": 5,
            "max_measurements_per_track": 5,
            "detection_probability": 0.9,
            "clutter_intensity": 1.0e-6,
        }

    def test_setting_state_sets_single_global_hypothesis(self):
        tracker = MultiHypothesisTracker(
            association_param=self.association_param,
            log_prior_estimates=False,
            log_posterior_estimates=False,
        )
        tracker.filter_state = self.kfs_init

        self.assertEqual(tracker.get_number_of_targets(), 2)
        self.assertEqual(tracker.get_number_of_global_hypotheses(), 1)
        self.assertEqual(tracker.get_point_estimate().shape, (2, 2))
        self.assertEqual(tracker.get_point_estimate(True).shape, (4,))

    def test_setting_multiple_global_hypotheses_and_weighted_estimate(self):
        tracker = MultiHypothesisTracker(
            log_prior_estimates=False,
            log_posterior_estimates=False,
        )
        tracker.set_global_hypotheses(
            [
                [KalmanFilter(GaussianDistribution(array([0.0, 0.0]), eye(2)))],
                [KalmanFilter(GaussianDistribution(array([2.0, 0.0]), eye(2)))],
            ],
            log_weights=log(array([0.75, 0.25])),
        )

        npt.assert_allclose(
            tracker.get_global_hypothesis_weights(), array([0.75, 0.25])
        )
        npt.assert_allclose(
            tracker.get_point_estimate(weighted_average=True),
            array([[0.5], [0.0]]),
        )

    def test_update_linear_prefers_correct_global_hypothesis(self):
        tracker = MultiHypothesisTracker(
            association_param=self.association_param,
            log_prior_estimates=False,
            log_posterior_estimates=False,
        )
        tracker.filter_state = self.kfs_init

        measurements = array([[9.9, 0.1], [0.2, -0.1]])
        tracker.update_linear(measurements, self.meas_mat, self.meas_noise)

        best_index = tracker.get_best_hypothesis_index()
        best_history = tracker.global_hypothesis_histories[best_index][-1]
        self.assertEqual(best_history, (1, 0))
        self.assertGreater(tracker.get_global_hypothesis_weights()[best_index], 0.99)
        npt.assert_allclose(
            tracker.get_point_estimate(),
            array([[0.05, 9.95], [-0.05, 0.1]]),
            atol=1.0e-12,
        )
        for filter_state in tracker.filter_state:
            npt.assert_allclose(filter_state.C, 0.5 * eye(2), atol=1.0e-12)

    def test_update_with_no_measurements_keeps_state(self):
        tracker = MultiHypothesisTracker(
            association_param=self.association_param,
            log_prior_estimates=False,
            log_posterior_estimates=False,
        )
        tracker.filter_state = self.kfs_init
        previous_estimate = tracker.get_point_estimate().copy()

        tracker.update_linear(array([[], []]), self.meas_mat, self.meas_noise)

        npt.assert_allclose(tracker.get_point_estimate(), previous_estimate)
        self.assertEqual(
            tracker.global_hypothesis_histories[tracker.get_best_hypothesis_index()][
                -1
            ],
            (-1, -1),
        )
        for filter_state in tracker.filter_state:
            npt.assert_allclose(filter_state.C, eye(2), atol=1.0e-12)


if __name__ == "__main__":
    unittest.main()
