import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
import scipy
from parameterized import parameterized

# pylint: disable=redefined-builtin,no-name-in-module,no-member
# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    all,
    allclose,
    array,
    column_stack,
    diag,
    dstack,
    eye,
    real,
    roll,
    sort,
    zeros,
)
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import KalmanFilter
from pyrecest.filters.global_nearest_neighbor import GlobalNearestNeighbor


class GlobalNearestNeighborTest(unittest.TestCase):
    """Test case for the GlobalNearestNeighbor class."""

    def setUp(self):
        """Initialize test variables before each test is run."""
        self.kfs_init = [
            KalmanFilter(
                GaussianDistribution(zeros(4), diag(array([1.0, 2.0, 3.0, 4.0])))
            ),
            KalmanFilter(
                GaussianDistribution(
                    array([1.0, 2.0, 3.0, 4.0]), diag(array([2.0, 2.0, 2.0, 2.0]))
                )
            ),
            KalmanFilter(
                GaussianDistribution(
                    -array([1.0, 2.0, 3.0, 4.0]), diag(array([4.0, 3.0, 2.0, 1.0]))
                )
            ),
        ]
        self.meas_mat = array([[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]])
        self.sys_mat = array(
            scipy.linalg.block_diag(
                array([[1.0, 1.0], [0.0, 1.0]]), array([[1.0, 1.0], [0.0, 1.0]])
            )
        )
        self.all_different_meas_covs = dstack(
            [
                diag(array([1.0, 2.0])),
                array([[5.0, 0.1], [0.1, 3.0]]),
                array([[2.0, -0.5], [-0.5, 0.5]]),
            ]
        )
        self.all_different_meas_covs_4 = dstack(
            (self.all_different_meas_covs, array([[2.0, -0.5], [-0.5, 0.5]]))
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_setting_state_sets_correct_state(self):
        tracker = GlobalNearestNeighbor()
        tracker.filter_state = self.kfs_init
        self.assertEqual(
            len(tracker.filter_state),
            len(self.kfs_init),
            "State was not set correctly.",
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ == "jax",
        reason="Not supported on this backend",
    )
    def test_get_state_returns_correct_shape(self):
        tracker = GlobalNearestNeighbor()
        tracker.filter_state = self.kfs_init
        self.assertEqual(tracker.get_point_estimate().shape, (4, 3))
        self.assertEqual(tracker.get_point_estimate(True).shape, (12,))

    @parameterized.expand(
        [("no_inputs", zeros(4)), ("with_inputs", array([1.0, -1.0, 1.0, -1.0]))]
    )
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_predict_linear(self, name, sys_input):
        import numpy as _np

        # Can use scipy.linalg.block_diag instead of native backend functions here because the efficiency does not matter
        # for the test.
        C_matrices = array(
            [
                scipy.linalg.block_diag(
                    [[3.0, 2.0], [2.0, 2.0]], [[7.0, 4.0], [4.0, 4.0]]
                )
                + _np.eye(4),
                scipy.linalg.block_diag(
                    [[4.0, 2.0], [2.0, 2.0]], [[4.0, 2.0], [2.0, 2.0]]
                )
                + _np.eye(4),
                scipy.linalg.block_diag(
                    [[7.0, 3.0], [3.0, 3.0]], [[3.0, 1.0], [1.0, 1.0]]
                )
                + _np.eye(4),
            ]
        )

        tracker = GlobalNearestNeighbor()
        tracker.filter_state = self.kfs_init
        if name == "no_inputs":
            tracker.predict_linear(self.sys_mat, eye(4))
        else:
            tracker.predict_linear(self.sys_mat, eye(4), sys_input)

        for i in range(3):
            with self.subTest(i=i):
                npt.assert_array_equal(
                    tracker.filter_bank[i].get_point_estimate(),
                    self.sys_mat @ self.kfs_init[i].get_point_estimate() + sys_input,
                )
                npt.assert_array_equal(
                    tracker.filter_bank[i].filter_state.C, C_matrices[i]
                )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_predict_linear_accepts_gaussian_process_noise(self):
        tracker = GlobalNearestNeighbor()
        tracker.filter_state = [self.kfs_init[0]]
        sys_noise = GaussianDistribution(zeros(4), 2.0 * eye(4), check_validity=False)
        prior_covariance = tracker.filter_bank[0].filter_state.C

        tracker.predict_linear(eye(4), sys_noise)

        npt.assert_array_equal(
            tracker.filter_bank[0].filter_state.C,
            prior_covariance + 2.0 * eye(4),
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_predict_linear_different_mats_and_inputs(self):
        tracker = GlobalNearestNeighbor()
        tracker.filter_state = self.kfs_init

        sys_mats = dstack(
            (
                scipy.linalg.block_diag(
                    [[1.0, 1.0], [0.0, 1.0]], [[1.0, 1.0], [0.0, 1.0]]
                ),
                eye(4),
                array(
                    [
                        [0.0, 0.0, 1.0, 1.0],
                        [0.0, 0.0, 0.0, 1.0],
                        [1.0, 1.0, 0.0, 0.0],
                        [0.0, 1.0, 0.0, 0.0],
                    ]
                ),
            )
        )
        sys_noises = dstack(
            (eye(4), diag([10.0, 11.0, 12.0, 13.0]), diag([1.0, 5.0, 3.0, 5.0]))
        )
        sys_inputs = array(
            [[-1.0, 1.0, -1.0, 1.0], [1.0, 2.0, 3.0, 4.0], -array([4.0, 3.0, 2.0, 1.0])]
        ).T

        tracker.predict_linear(sys_mats, sys_noises, sys_inputs)

        npt.assert_array_equal(
            tracker.filter_bank[0].filter_state.mu, array([-1.0, 1.0, -1.0, 1.0])
        )
        npt.assert_array_equal(
            tracker.filter_bank[1].filter_state.mu, array([2.0, 4.0, 6.0, 8.0])
        )
        npt.assert_array_equal(
            tracker.filter_bank[2].filter_state.mu, array([-11.0, -7.0, -5.0, -3.0])
        )
        npt.assert_array_equal(
            tracker.filter_bank[0].filter_state.C,
            scipy.linalg.block_diag([[4.0, 2.0], [2.0, 3.0]], [[8.0, 4.0], [4.0, 5.0]]),
        )
        npt.assert_array_equal(
            tracker.filter_bank[1].filter_state.C, diag([12.0, 13.0, 14.0, 15.0])
        )
        npt.assert_array_equal(
            tracker.filter_bank[2].filter_state.C,
            scipy.linalg.block_diag(
                [[4.0, 1.0], [1.0, 6.0]], [[10.0, 3.0], [3.0, 8.0]]
            ),
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_association_no_clutter(self):
        tracker = GlobalNearestNeighbor()
        tracker.filter_state = self.kfs_init
        all_gaussians = [kf.filter_state for kf in self.kfs_init]

        # Generate perfect measurements, association should then be
        # optimal.
        perfect_meas_ordered = (
            self.meas_mat @ array([gaussian.mu for gaussian in all_gaussians]).T
        )
        association = tracker.find_association(
            perfect_meas_ordered, self.meas_mat, eye(2)
        )
        npt.assert_array_equal(association, [0, 1, 2])

        # Shift them
        measurements = roll(perfect_meas_ordered, 1, axis=1)
        association = tracker.find_association(measurements, self.meas_mat, eye(2))
        npt.assert_array_equal(measurements[:, association], perfect_meas_ordered)

        # Shift them and add a bit of noise
        measurements = roll(perfect_meas_ordered, 1, axis=1) + 0.1
        association = tracker.find_association(measurements, self.meas_mat, eye(2))
        npt.assert_array_equal(measurements[:, association], perfect_meas_ordered + 0.1)

        # Use different covariances
        association = tracker.find_association(
            roll(perfect_meas_ordered, 1, axis=1) + 0.1,
            self.meas_mat,
            self.all_different_meas_covs,
        )
        npt.assert_array_equal(measurements[:, association], perfect_meas_ordered + 0.1)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_mahalanobis_uses_inverse_covariance(self):
        tracker = GlobalNearestNeighbor(
            association_param={
                "distance_metric_pos": "Mahalanobis",
                "square_dist": True,
                "gating_distance_threshold": 4.0,
            }
        )
        tracker.filter_state = [
            KalmanFilter(
                GaussianDistribution(
                    array([0.0, 0.0]),
                    diag(array([100.0, 1.0])),
                )
            )
        ]

        association = tracker.find_association(
            array([[10.0], [0.0]]),
            eye(2),
            zeros((2, 2)),
            warn_on_no_meas_for_track=False,
        )

        npt.assert_array_equal(association, array([0]))

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_square_dist_controls_gate_scale(self):
        tracker = GlobalNearestNeighbor(
            association_param={
                "distance_metric_pos": "Euclidean",
                "square_dist": True,
                "gating_distance_threshold": 3.0,
            }
        )
        tracker.filter_state = [KalmanFilter(GaussianDistribution(zeros(2), eye(2)))]

        association = tracker.find_association(
            array([[2.0], [0.0]]), eye(2), eye(2), warn_on_no_meas_for_track=False
        )
        self.assertGreaterEqual(int(association[0]), 1)

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_association_with_clutter(self):
        tracker = GlobalNearestNeighbor()
        tracker.filter_state = self.kfs_init
        all_gaussians = [kf.filter_state for kf in self.kfs_init]

        # Generate perfect measurements, association should then be
        # optimal.
        perfect_meas_ordered = self.meas_mat @ column_stack(
            [gaussian.mu for gaussian in all_gaussians]
        )
        measurements = column_stack([perfect_meas_ordered, array([3, 2])])
        association = tracker.find_association(measurements, self.meas_mat, eye(2))
        npt.assert_array_equal(association, [0, 1, 2])

        # Shift them and add one measurement
        measurements = column_stack(
            [
                perfect_meas_ordered[:, 1],
                perfect_meas_ordered[:, 2],
                array([2, 2]),
                perfect_meas_ordered[:, 0],
            ]
        )
        association = tracker.find_association(measurements, self.meas_mat, eye(2))
        npt.assert_array_equal(measurements[:, association], perfect_meas_ordered)

        # Shift them, add one add one meausurement, and add a bit of noise
        association = tracker.find_association(
            measurements + 0.1, self.meas_mat, eye(2)
        )
        npt.assert_array_equal(
            measurements[:, association] + 0.1, perfect_meas_ordered + 0.1
        )

        # Use different covariances
        association = tracker.find_association(
            measurements + 0.1, self.meas_mat, self.all_different_meas_covs_4
        )
        npt.assert_array_equal(
            measurements[:, association] + 0.1, perfect_meas_ordered + 0.1
        )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_update_with_and_without_clutter(self):
        tracker_no_clut = GlobalNearestNeighbor()
        tracker_clut = GlobalNearestNeighbor()
        tracker_no_clut.filter_state = self.kfs_init
        tracker_clut.filter_state = self.kfs_init
        all_gaussians = [kf.filter_state for kf in self.kfs_init]

        perfect_meas_ordered = self.meas_mat @ column_stack(
            [gaussian.mu for gaussian in all_gaussians]
        )
        measurements_no_clut = perfect_meas_ordered
        tracker_no_clut.update_linear(measurements_no_clut, self.meas_mat, eye(2))

        self.assertTrue(
            allclose(
                [dist.mu for dist in tracker_no_clut.filter_state],
                [dist.mu for dist in all_gaussians],
            )
        )
        curr_covs = dstack([dist.C for dist in tracker_no_clut.filter_state])
        self.assertTrue(all(curr_covs <= dstack([dist.C for dist in all_gaussians])))

        measurements_clut = column_stack(
            [measurements_no_clut, array([2, 2]).reshape(-1, 1)]
        )
        tracker_clut.update_linear(measurements_clut, self.meas_mat, eye(2))
        self.assertTrue(
            allclose(
                tracker_clut.get_point_estimate(), tracker_no_clut.get_point_estimate()
            )
        )

        measurements_no_clut = perfect_meas_ordered[:, [1, 2, 0]]
        tracker_no_clut.update_linear(measurements_no_clut, self.meas_mat, eye(2))
        self.assertTrue(
            allclose(
                [dist.mu for dist in tracker_no_clut.filter_state],
                [dist.mu for dist in all_gaussians],
            )
        )
        previous_covs = curr_covs
        curr_covs = dstack([dist.C for dist in tracker_no_clut.filter_state])
        self.assertTrue(all(curr_covs <= previous_covs))

        measurements_clut = column_stack(
            [
                perfect_meas_ordered[:, [1, 2]],
                array([2, 2]).reshape(-1, 1),
                perfect_meas_ordered[:, 0],
            ]
        )
        tracker_clut.update_linear(measurements_clut, self.meas_mat, eye(2))
        self.assertTrue(
            allclose(
                tracker_clut.get_point_estimate(), tracker_no_clut.get_point_estimate()
            )
        )

        measurements_no_clut += 0.1
        tracker_no_clut.update_linear(measurements_no_clut, self.meas_mat, eye(2))
        curr_means = [dist.mu for dist in tracker_no_clut.filter_state]
        self.assertFalse(allclose(curr_means, [dist.mu for dist in all_gaussians]))
        previous_covs = curr_covs
        curr_covs = dstack([dist.C for dist in tracker_no_clut.filter_state])
        self.assertTrue(all(curr_covs <= previous_covs))

        measurements_clut += 0.1
        tracker_clut.update_linear(measurements_clut, self.meas_mat, eye(2))
        self.assertTrue(
            allclose(
                tracker_clut.get_point_estimate(), tracker_no_clut.get_point_estimate()
            )
        )

        tracker_no_clut.update_linear(
            measurements_no_clut, self.meas_mat, self.all_different_meas_covs
        )
        previous_means = curr_means
        self.assertFalse(
            allclose([dist.mu for dist in tracker_no_clut.filter_state], previous_means)
        )
        previous_covs = curr_covs
        curr_covs = dstack([dist.C for dist in tracker_no_clut.filter_state])
        for i in range(curr_covs.shape[2]):
            self.assertTrue(
                all(
                    sort(real(scipy.linalg.eigvals(curr_covs[:, :, i])))
                    <= sort(real(scipy.linalg.eigvals(previous_covs[:, :, i])))
                )
            )

        tracker_clut.update_linear(
            measurements_clut, self.meas_mat, self.all_different_meas_covs_4
        )
        self.assertTrue(
            allclose(
                tracker_clut.get_point_estimate(), tracker_no_clut.get_point_estimate()
            )
        )


if __name__ == "__main__":
    unittest.main()
