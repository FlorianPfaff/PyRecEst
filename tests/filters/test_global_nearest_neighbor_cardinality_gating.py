import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, eye, zeros
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import KalmanFilter
from pyrecest.filters.global_nearest_neighbor import GlobalNearestNeighbor


class GlobalNearestNeighborCardinalityGatingTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_positive_pairwise_cost_does_not_remove_geometrically_gated_edge(self):
        tracker = GlobalNearestNeighbor(
            association_param={
                "distance_metric_pos": "Euclidean",
                "square_dist": False,
                "gating_distance_threshold": 1.0,
                "maximize_cardinality": True,
            }
        )
        tracker.filter_state = [
            KalmanFilter(GaussianDistribution(zeros(2), eye(2)))
        ]

        association = tracker.find_association(
            array([[0.0], [0.0]]),
            eye(2),
            eye(2),
            warn_on_no_meas_for_track=False,
            pairwise_cost_matrix=array([[100.0]]),
        )

        npt.assert_array_equal(association, array([0]))


if __name__ == "__main__":
    unittest.main()
