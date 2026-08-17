import unittest

import numpy.testing as npt

import pyrecest.backend
from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import KalmanFilter
from pyrecest.filters.global_nearest_neighbor import GlobalNearestNeighbor


class GlobalNearestNeighborPairwiseGatingTest(unittest.TestCase):
    @unittest.skipIf(
        pyrecest.backend.__backend_name__ in ("pytorch", "jax"),
        reason="Not supported on this backend",
    )
    def test_pairwise_cost_does_not_reduce_maximum_cardinality(self):
        tracker = GlobalNearestNeighbor(
            association_param={
                "distance_metric_pos": "Euclidean",
                "square_dist": False,
                "gating_distance_threshold": 2.05,
                "max_new_tracks": 2,
                "maximize_cardinality": True,
            }
        )
        tracker.filter_state = [
            KalmanFilter(GaussianDistribution(array([0.0, 0.0]), eye(2))),
            KalmanFilter(GaussianDistribution(array([0.0, -0.1]), eye(2))),
        ]

        association = tracker.find_association(
            array([[0.0, 0.0], [1.0, 2.0]]),
            eye(2),
            eye(2),
            warn_on_no_meas_for_track=False,
            pairwise_cost_matrix=array([[0.0, 1.0], [0.0, 0.0]]),
        )

        # Track 0 -> measurement 1 is geometrically inside the 2.05 gate, even
        # though its auxiliary cost raises the combined cost from 2.0 to 3.0.
        # Maximum-cardinality association must therefore keep both matches.
        npt.assert_array_equal(association, array([1, 0]))


if __name__ == "__main__":
    unittest.main()
