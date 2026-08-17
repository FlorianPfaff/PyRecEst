import unittest

from scipy.stats import chi2

from pyrecest.filters import GlobalNearestNeighbor


class GlobalNearestNeighborGatingTest(unittest.TestCase):
    def test_default_gate_matches_distance_representation(self):
        squared_tracker = GlobalNearestNeighbor(
            association_param={"square_dist": True}
        )
        unsquared_tracker = GlobalNearestNeighbor(
            association_param={"square_dist": False}
        )

        squared_threshold = chi2.ppf(0.999, 2)
        self.assertAlmostEqual(
            squared_tracker.association_param["gating_distance_threshold"],
            squared_threshold,
        )
        self.assertAlmostEqual(
            unsquared_tracker.association_param["gating_distance_threshold"],
            squared_threshold**0.5,
        )

    def test_explicit_unsquared_gate_is_preserved(self):
        tracker = GlobalNearestNeighbor(
            association_param={
                "square_dist": False,
                "gating_distance_threshold": 7.5,
            }
        )

        self.assertEqual(tracker.association_param["gating_distance_threshold"], 7.5)


if __name__ == "__main__":
    unittest.main()
