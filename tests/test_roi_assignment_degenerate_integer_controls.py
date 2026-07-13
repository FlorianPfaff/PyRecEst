import unittest

from pyrecest.backend import array
from pyrecest.utils.roi_assignment import (
    minimum_similarity_threshold,
    otsu_similarity_threshold,
)


class TestRoiAssignmentDegenerateIntegerControls(unittest.TestCase):
    def test_degenerate_threshold_inputs_reject_invalid_nbins(self):
        degenerate_scores = (
            array([]),
            array([0.5, 0.5]),
            array([float("nan")]),
        )
        invalid_nbins = (0, -1, 1.5, True, [2])

        for threshold_fn in (otsu_similarity_threshold, minimum_similarity_threshold):
            for scores in degenerate_scores:
                for nbins in invalid_nbins:
                    with self.subTest(
                        threshold_fn=threshold_fn.__name__,
                        scores=scores,
                        nbins=nbins,
                    ):
                        with self.assertRaisesRegex(ValueError, "nbins"):
                            threshold_fn(scores, nbins=nbins)


if __name__ == "__main__":
    unittest.main()
