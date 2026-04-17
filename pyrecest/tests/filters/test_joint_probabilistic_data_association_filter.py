import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import allclose, array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import JPDAF, KalmanFilter


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Only supported on numpy backend",
)
class JointProbabilisticDataAssociationFilterTest(unittest.TestCase):
    def setUp(self):
        self.kfs_init = [
            KalmanFilter(
                GaussianDistribution(
                    array([-10.0, 0.0]),
                    0.5 * eye(2),
                )
            ),
            KalmanFilter(
                GaussianDistribution(
                    array([10.0, 0.0]),
                    0.5 * eye(2),
                )
            ),
        ]
        self.meas_mat = eye(2)
        self.meas_cov = eye(2)
        self.association_param = {
            "detection_probability": 0.95,
            "clutter_intensity": 1e-3,
            "gating_distance_threshold": 100.0,
        }

    def test_setting_state_sets_correct_state(self):
        tracker = JPDAF(association_param=self.association_param)
        tracker.filter_state = self.kfs_init
        self.assertEqual(len(tracker.filter_state), len(self.kfs_init))

    def test_get_state_returns_correct_shape(self):
        tracker = JPDAF(self.kfs_init, association_param=self.association_param)
        self.assertEqual(tracker.get_point_estimate().shape, (2, 2))
        self.assertEqual(tracker.get_point_estimate(True).shape, (4,))

    def test_find_association_probabilities_and_map_assignment(self):
        tracker = JPDAF(self.kfs_init, association_param=self.association_param)
        perfect_meas_ordered = (
            self.meas_mat @ array([kf.get_point_estimate() for kf in self.kfs_init]).T
        )
        association_probabilities, map_association = (
            tracker.find_association_probabilities(
                perfect_meas_ordered,
                self.meas_mat,
                self.meas_cov,
            )
        )

        npt.assert_allclose(association_probabilities.sum(axis=1), array([1.0, 1.0]))
        self.assertTrue(all(association_probabilities.diagonal(offset=1) > 0.99))
        npt.assert_array_equal(map_association, array([0, 1]))

        shuffled_meas = perfect_meas_ordered[:, [1, 0]]
        _, shuffled_map_association = tracker.find_association_probabilities(
            shuffled_meas,
            self.meas_mat,
            self.meas_cov,
        )
        npt.assert_array_equal(shuffled_map_association, array([1, 0]))

    def test_update_linear_is_robust_to_far_away_clutter(self):
        tracker_no_clutter = JPDAF(
            self.kfs_init, association_param=self.association_param
        )
        tracker_clutter = JPDAF(self.kfs_init, association_param=self.association_param)

        perfect_meas_ordered = (
            self.meas_mat @ array([kf.get_point_estimate() for kf in self.kfs_init]).T
        )
        cluttered_measurements = array([[-10.0, 10.0, 100.0], [0.0, 0.0, 100.0]])

        tracker_no_clutter.update_linear(
            perfect_meas_ordered, self.meas_mat, self.meas_cov
        )
        tracker_clutter.update_linear(
            cluttered_measurements, self.meas_mat, self.meas_cov
        )

        self.assertTrue(
            allclose(
                tracker_no_clutter.get_point_estimate(),
                tracker_clutter.get_point_estimate(),
            )
        )
        self.assertTrue(
            allclose(
                tracker_clutter.latest_association_probabilities[:, -1],
                array([0.0, 0.0]),
            )
        )

        prior_covariances = [kf.filter_state.C for kf in self.kfs_init]
        posterior_covariances = [dist.C for dist in tracker_no_clutter.filter_state]
        for posterior_covariance, prior_covariance in zip(
            posterior_covariances,
            prior_covariances,
        ):
            self.assertTrue(
                (posterior_covariance.diagonal() < prior_covariance.diagonal()).all()
            )

    def test_symmetric_ambiguous_case_leads_to_symmetric_probabilities(self):
        tracker = JPDAF(
            [
                KalmanFilter(GaussianDistribution(array([-1.0, 0.0]), eye(2))),
                KalmanFilter(GaussianDistribution(array([1.0, 0.0]), eye(2))),
            ],
            association_param={
                "detection_probability": 0.9,
                "clutter_intensity": 1e-3,
                "gating_distance_threshold": 100.0,
            },
        )
        measurements = array([[-0.5, 0.5], [0.0, 0.0]])
        association_probabilities, _ = tracker.find_association_probabilities(
            measurements,
            self.meas_mat,
            self.meas_cov,
        )

        npt.assert_allclose(
            association_probabilities[0, 1],
            association_probabilities[1, 2],
        )
        npt.assert_allclose(
            association_probabilities[0, 2],
            association_probabilities[1, 1],
        )

        tracker.update_linear(measurements, self.meas_mat, self.meas_cov)
        point_estimates = tracker.get_point_estimate()
        npt.assert_allclose(point_estimates[:, 0], -point_estimates[:, 1])


if __name__ == "__main__":
    unittest.main()
