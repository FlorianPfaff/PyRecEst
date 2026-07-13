import unittest

import numpy as np
from pyrecest.backend import array
from pyrecest.smoothers import SlidingWindowManifoldMeanSmoother


class SlidingWindowManifoldMeanSmootherNonfiniteWeightsTest(unittest.TestCase):
    def test_rejects_nonfinite_window_weights(self):
        invalid_weight_vectors = (
            array([1.0, np.nan, 1.0]),
            array([1.0, np.inf, 1.0]),
            array([1.0, -np.inf, 1.0]),
        )

        for window_weights in invalid_weight_vectors:
            with self.subTest(window_weights=window_weights):
                with self.assertRaisesRegex(ValueError, "finite"):
                    SlidingWindowManifoldMeanSmoother(
                        window_size=3,
                        window_weights=window_weights,
                    )


if __name__ == "__main__":
    unittest.main()
