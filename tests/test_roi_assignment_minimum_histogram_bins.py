import unittest

from pyrecest.backend import array
from pyrecest.utils.roi_assignment import (
    minimum_similarity_threshold,
    otsu_similarity_threshold,
)


class TestRoiThresholdMinimumHistogramBins(unittest.TestCase):
    def test_thresholds_reject_single_bin_histograms(self):
        score_cases = (
            ("nondegenerate", array([0.2, 0.8])),
            ("constant", array([0.5, 0.5])),
            ("empty", array([])),
        )

        for threshold_fn in (otsu_similarity_threshold, minimum_similarity_threshold):
            for case_name, scores in score_cases:
                with self.subTest(threshold_fn=threshold_fn.__name__, case=case_name):
                    with self.assertRaisesRegex(ValueError, "nbins"):
                        threshold_fn(scores, nbins=1)


if __name__ == "__main__":
    unittest.main()
