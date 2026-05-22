import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, eye, log
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import KalmanFilter
from pyrecest.filters import MultiHypothesisTracker as ExportedMultiHypothesisTracker
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

    def test_filter_namespace_exports_multi_hypothesis_tracker(self):
        self.assertIs(ExportedMultiHypothesisTracker, MultiHypothesisTracker)

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

    def test_setting_gaussian_states_wraps_kalman_filters(self):
        tracker = MultiHypothesisTracker(
            association_param=self.association_param,
            log_prior_estimates=False,
            log_posterior_estimates=False,
        )
        tracker.filter_state = [
            GaussianDistribution(array([0.0, 0.0]), eye(2)),
            GaussianDistribution(array([10.0, 0.0]), eye(2)),
        ]

        self.assertEqual(tracker.get_number_of_targets(), 2)
        self.assertIsInstance(tracker.filter_bank[0], KalmanFilter)
        npt.assert_allclose(
            tracker.get_point_estimate(), array([[0.0, 10.0], [0.0, 0.0]])
        )

    def test_setting_inconsistent_global_hypotheses_raises(self):
        tracker = MultiHypothesisTracker(
            log_prior_estimates=False,
            log_posterior_estimates=False,
        )
        with self.assertRaisesRegex(ValueError, "same number of tracks"):
            tracker.set_global_hypotheses(
                [
                    [KalmanFilter(GaussianDistribution(array([0.0, 0.0]), eye(2)))],
                    [
                        KalmanFilter(GaussianDistribution(array([0.0, 0.0]), eye(2))),
                        KalmanFilter(GaussianDistribution(array([1.0, 0.0]), eye(2))),
                    ],
                ]
            )

    def test_update_linear_accepts_array_like_inputs(self):
        tracker = MultiHypothesisTracker(
            association_param=self.association_param,
            log_prior_estimates=False,
            log_posterior_estimates=False,
        )
        tracker.filter_state = [GaussianDistribution(array([0.0, 0.0]), eye(2))]

        tracker.update_linear([[0.2], [-0.2]], [[1.0, 0.0], [0.0, 1.0]], eye(2))

        npt.assert_allclose(
            tracker.get_point_estimate(),
            array([[0.1], [-0.1]]),
            atol=1.0e-12,
        )

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

    def test_update_accepts_array_like_inputs(self):
        tracker = MultiHypothesisTracker(
            association_param=self.association_param,
            log_prior_estimates=False,
            log_posterior_estimates=False,
        )
        tracker.filter_state = self.kfs_init

        tracker.update_linear(
            [[9.9, 0.1], [0.2, -0.1]],
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        )

        best_history = tracker.global_hypothesis_histories[
            tracker.get_best_hypothesis_index()
        ][-1]
        self.assertEqual(best_history, (1, 0))

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

    def test_assignment_marginals_and_lagged_commitment(self):
        tracker = MultiHypothesisTracker(
            association_param=self.association_param,
            log_prior_estimates=False,
            log_posterior_estimates=False,
        )
        tracker.filter_state = self.kfs_init
        tracker.update_linear(
            array([[9.9, 0.1], [0.2, -0.1]]), self.meas_mat, self.meas_noise
        )

        distribution = tracker.get_assignment_distribution(lag=0)
        self.assertAlmostEqual(sum(distribution.values()), 1.0)
        self.assertGreater(distribution[(1, 0)], 0.99)

        marginals = tracker.get_assignment_marginals(lag=0)
        self.assertGreater(marginals[0][1], 0.99)
        self.assertGreater(marginals[1][0], 0.99)

        commitment = tracker.get_lagged_assignment_commitment(lag=0, mass_threshold=0.9)
        self.assertEqual(commitment["assignment"], (1, 0))
        self.assertGreater(commitment["probability"], 0.99)

        track_commitment = tracker.get_track_assignment_commitments(
            lag=0, mass_threshold=0.9
        )
        self.assertEqual(track_commitment["assignments"], (1, 0))
        npt.assert_allclose(
            tracker.get_lagged_point_estimate(lag=0),
            tracker.get_point_estimate(),
        )
        self.assertGreater(tracker.get_best_hypothesis_margin(), 0.0)
        self.assertGreaterEqual(tracker.get_hypothesis_entropy(), 0.0)

    def test_hypothesis_reranker_can_lift_non_top_branch(self):
        def prefer_identity_assignment(
            _filter_bank, assignment_history, _base_log_weight
        ):
            if assignment_history and assignment_history[-1] == (0, 1):
                return 100.0
            return 0.0

        tracker = MultiHypothesisTracker(
            association_param={
                **self.association_param,
                "max_hypotheses_per_global_hypothesis": 10,
            },
            log_prior_estimates=False,
            log_posterior_estimates=False,
            hypothesis_reranker=prefer_identity_assignment,
        )
        tracker.filter_state = self.kfs_init
        tracker.update_linear(
            array([[9.9, 0.1], [0.2, -0.1]]), self.meas_mat, self.meas_noise
        )

        best_history = tracker.global_hypothesis_histories[
            tracker.get_best_hypothesis_index()
        ][-1]
        self.assertEqual(best_history, (0, 1))
        self.assertEqual(
            tracker.get_top_hypotheses(1, include_weights=True, include_histories=True)[
                0
            ]["history"][-1],
            (0, 1),
        )

    def test_score_temperature_softens_effective_weights(self):
        tracker_cold = MultiHypothesisTracker(
            association_param={**self.association_param, "score_temperature": 1.0},
            log_prior_estimates=False,
            log_posterior_estimates=False,
        )
        tracker_warm = MultiHypothesisTracker(
            association_param={**self.association_param, "score_temperature": 5.0},
            log_prior_estimates=False,
            log_posterior_estimates=False,
        )
        for tracker in (tracker_cold, tracker_warm):
            tracker.filter_state = self.kfs_init
            tracker.update_linear(
                array([[9.9, 0.1], [0.2, -0.1]]),
                self.meas_mat,
                self.meas_noise,
            )

        self.assertGreater(
            tracker_warm.get_hypothesis_entropy(),
            tracker_cold.get_hypothesis_entropy(),
        )

    def test_diverse_pruning_keeps_distinct_recent_assignment_signatures(self):
        tracker = MultiHypothesisTracker(
            association_param={
                **self.association_param,
                "max_global_hypotheses": 2,
                "pruning_strategy": "diverse_top_k",
                "diversity_history_length": 1,
                "max_hypotheses_per_signature": 1,
            },
            log_prior_estimates=False,
            log_posterior_estimates=False,
        )
        tracker.filter_state = self.kfs_init
        tracker.update_linear(
            array([[9.9, 0.1], [0.2, -0.1]]), self.meas_mat, self.meas_noise
        )

        signatures = {
            tuple(history[-1:]) for history in tracker.global_hypothesis_histories
        }
        self.assertEqual(len(signatures), tracker.get_number_of_global_hypotheses())
        self.assertLessEqual(tracker.get_number_of_global_hypotheses(), 2)


if __name__ == "__main__":
    unittest.main()
