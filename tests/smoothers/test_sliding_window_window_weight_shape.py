import unittest

from pyrecest.backend import array
from pyrecest.smoothers import SlidingWindowManifoldMeanSmoother


class SlidingWindowWindowWeightShapeTest(unittest.TestCase):
    def test_rejects_matrix_shaped_window_weights(self):
        matrix_weights = (
            array([[1.0, 2.0, 1.0]]),
            array([[1.0], [2.0], [1.0]]),
        )

        for window_weights in matrix_weights:
            with self.subTest(shape=window_weights.shape):
                with self.assertRaisesRegex(ValueError, "one-dimensional"):
                    SlidingWindowManifoldMeanSmoother(
                        window_size=3,
                        window_weights=window_weights,
                    )

    def test_scalar_weight_remains_supported_for_singleton_window(self):
        smoother = SlidingWindowManifoldMeanSmoother(
            window_size=1,
            window_weights=array(2.0),
        )

        self.assertEqual(smoother.window_weights.shape, (1,))


if __name__ == "__main__":
    unittest.main()
