import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest import backend
from pyrecest.backend import array
from pyrecest.utils.roi_assignment import (
    assign_by_similarity_matrix,
    minimum_similarity_threshold,
    otsu_similarity_threshold,
)


class TestRoiAssignmentIntegerControls(unittest.TestCase):
    def test_thresholds_reject_invalid_nbins(self):
        scores = array([0.05, 0.08, 0.1, 0.81, 0.84, 0.9])

        for threshold_fn in (otsu_similarity_threshold, minimum_similarity_threshold):
            for nbins in (0, -1, 1.5, True, [2]):
                with self.subTest(threshold_fn=threshold_fn.__name__, nbins=nbins):
                    with self.assertRaisesRegex(ValueError, "nbins"):
                        threshold_fn(scores, nbins=nbins)

    @unittest.skipIf(
        backend.__backend_name__ == "jax",
        reason="Not supported on the jax backend",
    )
    def test_assignment_rejects_invalid_num_dummy(self):
        similarity_matrix = array([[1.0]])

        for num_dummy in (-1, 1.5, True, [1]):
            with self.subTest(num_dummy=num_dummy):
                with self.assertRaisesRegex(ValueError, "num_dummy"):
                    assign_by_similarity_matrix(
                        similarity_matrix,
                        num_dummy=num_dummy,
                    )


if __name__ == "__main__":
    unittest.main()
