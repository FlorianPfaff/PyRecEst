import unittest

import numpy as np
from pyrecest.filters import normalize_log_weights


class GaussianHypothesisLogWeightShapeTest(unittest.TestCase):
    def test_matrix_log_weights_are_rejected(self):
        for log_weights in (
            np.array([[0.0, 1.0]]),
            np.array([[0.0], [1.0]]),
        ):
            with self.subTest(shape=log_weights.shape):
                with self.assertRaisesRegex(ValueError, "one-dimensional"):
                    normalize_log_weights(log_weights)

    def test_scalar_and_vector_log_weights_remain_supported(self):
        self.assertTrue(np.allclose(normalize_log_weights(0.0), np.array([1.0])))
        self.assertTrue(
            np.allclose(
                normalize_log_weights(np.array([0.0, 0.0])),
                np.array([0.5, 0.5]),
            )
        )


if __name__ == "__main__":
    unittest.main()
