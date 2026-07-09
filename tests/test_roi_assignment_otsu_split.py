import unittest

from pyrecest.backend import array, concatenate
from pyrecest.utils.roi_assignment import otsu_similarity_threshold


class TestRoiOtsuStrictSplit(unittest.TestCase):
    def test_skewed_two_bin_histogram_uses_strict_foreground_split(self):
        scores = concatenate([array([0.0] * 90), array([1.0] * 10)])

        threshold = otsu_similarity_threshold(scores, nbins=2)

        self.assertAlmostEqual(threshold, 0.25)


if __name__ == "__main__":
    unittest.main()
