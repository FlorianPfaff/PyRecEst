import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-member
import pyrecest.backend
from pyrecest.backend import array, diag, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import KalmanFilter
from pyrecest.filters.global_nearest_neighbor import GlobalNearestNeighbor


@unittest.skipIf(
    pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
    reason="GlobalNearestNeighbor array-like input regression uses numpy fixtures.",
)
class GlobalNearestNeighborArrayLikeInputTest(unittest.TestCase):
    def _tracker(self):
        tracker = GlobalNearestNeighbor()
        tracker.filter_state = [
            KalmanFilter(
                GaussianDistribution(zeros(4), diag(array([1.0, 2.0, 3.0, 4.0])))
            ),
            KalmanFilter(
                GaussianDistribution(
                    array([1.0, 2.0, 3.0, 4.0]), diag(array([2.0, 2.0, 2.0, 2.0]))
                )
            ),
        ]
        return tracker

    def test_find_association_accepts_array_like_inputs(self):
        tracker = self._tracker()

        association = tracker.find_association(
            measurements=[[0.0, 1.0], [0.0, 3.0]],
            measurement_matrix=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            cov_mats_meas=[[1.0, 0.0], [0.0, 1.0]],
            warn_on_no_meas_for_track=False,
        )

        npt.assert_array_equal(association, np.array([0, 1]))

    def test_update_linear_accepts_array_like_inputs(self):
        tracker = self._tracker()

        tracker.update_linear(
            measurements=[[0.0, 1.0], [0.0, 3.0]],
            measurement_matrix=[
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
            ],
            covMatsMeas=[[1.0, 0.0], [0.0, 1.0]],
        )

        point_estimate = tracker.get_point_estimate()
        self.assertEqual(point_estimate.shape, (4, 2))
        self.assertTrue(np.all(np.isfinite(point_estimate)))


if __name__ == "__main__":
    unittest.main()
