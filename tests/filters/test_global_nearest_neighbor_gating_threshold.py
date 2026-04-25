import unittest

from pyrecest.filters.global_nearest_neighbor import GlobalNearestNeighbor
from scipy.stats import chi2


class GlobalNearestNeighborGatingThresholdTest(unittest.TestCase):
    def test_default_gating_threshold_uses_unsquared_chi_square_quantile(self):
        tracker = GlobalNearestNeighbor()

        self.assertAlmostEqual(
            tracker.association_param["gating_distance_threshold"],
            chi2.ppf(0.999, 2),
        )


if __name__ == "__main__":
    unittest.main()
