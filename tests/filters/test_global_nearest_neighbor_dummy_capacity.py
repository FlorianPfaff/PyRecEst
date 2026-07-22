import unittest

import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
import pyrecest.backend
from pyrecest.backend import array, eye
from pyrecest.distributions import GaussianDistribution
from pyrecest.filters import GlobalNearestNeighbor, KalmanFilter


@unittest.skipIf(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="Only supported on numpy backend",
)
class GlobalNearestNeighborDummyCapacityTest(unittest.TestCase):
    def test_all_out_of_gate_tracks_can_use_dummy_assignments(self):
        tracker = GlobalNearestNeighbor(
            association_param={
                "distance_metric_pos": "Euclidean",
                "square_dist": False,
                "gating_distance_threshold": 1.0,
                "max_new_tracks": 1,
            }
        )
        tracker.filter_state = [
            KalmanFilter(GaussianDistribution(array([0.0, 0.0]), eye(2)))
            for _ in range(3)
        ]

        association = tracker.find_association(
            array([[10.0, 20.0, 30.0], [0.0, 0.0, 0.0]]),
            eye(2),
            eye(2),
            warn_on_no_meas_for_track=False,
        )

        npt.assert_array_equal(association, array([3, 3, 3]))


if __name__ == "__main__":
    unittest.main()
