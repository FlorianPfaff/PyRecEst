import unittest

import numpy.testing as npt

# pylint: disable=no-member,duplicate-code
import pyrecest.backend
from pyrecest.backend import array, column_stack, diag, eye, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import KalmanFilter
from pyrecest.filters.global_nearest_neighbor import GlobalNearestNeighbor


class GlobalNearestNeighborPairwiseCostTest(unittest.TestCase):
    """Regression tests for pairwise-cost fusion in GNN."""

    def setUp(self):
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

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_association_supports_shared_2d_measurement_covariance(self):
        tracker = GlobalNearestNeighbor()
        tracker.filter_state = [
            KalmanFilter(
                GaussianDistribution(zeros(4), diag(array([1.0, 2.0, 3.0, 4.0])))
            ),
            KalmanFilter(
                GaussianDistribution(
                    array([1.0, 2.0, 3.0, 4.0]), diag(array([1.0, 2.0, 3.0, 4.0]))
                )
            ),
            KalmanFilter(
                GaussianDistribution(
                    -array([1.0, 2.0, 3.0, 4.0]), diag(array([1.0, 2.0, 3.0, 4.0]))
                )
            ),
        ]
        all_gaussians = [kf.filter_state for kf in tracker.filter_bank]
        perfect_meas_ordered = self.meas_mat @ column_stack(
            [gaussian.mu for gaussian in all_gaussians]
        )
        association = tracker.find_association(
            perfect_meas_ordered, self.meas_mat, eye(2)
        )
        npt.assert_array_equal(association, [0, 1, 2])

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_pairwise_cost_matrix_can_override_geometric_assignment(self):
        tracker = GlobalNearestNeighbor()
        tracker.filter_state = self.kfs_init
        all_gaussians = [kf.filter_state for kf in self.kfs_init]
        perfect_meas_ordered = self.meas_mat @ column_stack(
            [gaussian.mu for gaussian in all_gaussians]
        )

        pairwise_cost_matrix = array(
            [[10.0, -10.0, 10.0], [10.0, 10.0, -10.0], [-10.0, 10.0, 10.0]]
        )
        association = tracker.find_association(
            perfect_meas_ordered,
            self.meas_mat,
            eye(2),
            pairwise_cost_matrix=pairwise_cost_matrix,
        )
        npt.assert_array_equal(association, [1, 2, 0])

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_pairwise_cost_matrix_shape_is_validated(self):
        tracker = GlobalNearestNeighbor()
        tracker.filter_state = self.kfs_init
        all_gaussians = [kf.filter_state for kf in self.kfs_init]
        perfect_meas_ordered = self.meas_mat @ column_stack(
            [gaussian.mu for gaussian in all_gaussians]
        )

        with self.assertRaises(ValueError):
            tracker.find_association(
                perfect_meas_ordered,
                self.meas_mat,
                eye(2),
                pairwise_cost_matrix=zeros((2, 3)),
            )

    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_update_linear_accepts_pairwise_cost_matrix(self):
        tracker_manual = GlobalNearestNeighbor()
        tracker_pairwise = GlobalNearestNeighbor()
        tracker_manual.filter_state = self.kfs_init
        tracker_pairwise.filter_state = self.kfs_init

        all_gaussians = [kf.filter_state for kf in self.kfs_init]
        perfect_meas_ordered = self.meas_mat @ column_stack(
            [gaussian.mu for gaussian in all_gaussians]
        )
        forced_permutation = [1, 2, 0]

        for track_index, meas_index in enumerate(forced_permutation):
            tracker_manual.filter_bank[track_index].update_linear(
                perfect_meas_ordered[:, meas_index],
                self.meas_mat,
                eye(2),
            )

        pairwise_cost_matrix = array(
            [[10.0, -10.0, 10.0], [10.0, 10.0, -10.0], [-10.0, 10.0, 10.0]]
        )
        tracker_pairwise.update_linear(
            perfect_meas_ordered,
            self.meas_mat,
            eye(2),
            pairwise_cost_matrix=pairwise_cost_matrix,
        )

        npt.assert_allclose(
            tracker_pairwise.get_point_estimate(),
            tracker_manual.get_point_estimate(),
        )


if __name__ == "__main__":
    unittest.main()
