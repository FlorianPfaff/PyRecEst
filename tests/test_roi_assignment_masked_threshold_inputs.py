import unittest

import numpy as np
from pyrecest.utils.roi_assignment import (
    minimum_similarity_threshold,
    otsu_similarity_threshold,
)


class TestRoiAssignmentMaskedThresholdInputs(unittest.TestCase):
    def test_thresholds_reject_masked_similarity_entries(self):
        similarities = np.ma.array(
            [0.05, 0.08, 0.81, 0.9],
            mask=[False, True, False, False],
        )

        for threshold_fn in (otsu_similarity_threshold, minimum_similarity_threshold):
            with self.subTest(threshold_fn=threshold_fn.__name__):
                with self.assertRaisesRegex(
                    ValueError,
                    "similarities must not contain masked values",
                ):
                    threshold_fn(similarities)

    def test_thresholds_reject_nested_masked_similarity_entries(self):
        similarities = [0.05, np.ma.masked, 0.81, 0.9]

        for threshold_fn in (otsu_similarity_threshold, minimum_similarity_threshold):
            with self.subTest(threshold_fn=threshold_fn.__name__):
                with self.assertRaisesRegex(
                    ValueError,
                    "similarities must not contain masked values",
                ):
                    threshold_fn(similarities)

    def test_thresholds_reject_masked_bin_count(self):
        similarities = np.array([0.05, 0.08, 0.81, 0.9])
        masked_nbins = np.ma.array(32, mask=True)

        for threshold_fn in (otsu_similarity_threshold, minimum_similarity_threshold):
            with self.subTest(threshold_fn=threshold_fn.__name__):
                with self.assertRaisesRegex(ValueError, "nbins"):
                    threshold_fn(similarities, nbins=masked_nbins)

    def test_thresholds_accept_masked_array_without_masked_entries(self):
        similarities = np.ma.array(
            [0.05, 0.08, 0.81, 0.9],
            mask=False,
        )

        for threshold_fn in (otsu_similarity_threshold, minimum_similarity_threshold):
            with self.subTest(threshold_fn=threshold_fn.__name__):
                threshold = threshold_fn(similarities, nbins=16)
                self.assertTrue(np.isfinite(threshold))


if __name__ == "__main__":
    unittest.main()
