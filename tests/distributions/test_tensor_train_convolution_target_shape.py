import unittest

import numpy as np

from pyrecest.distributions.hypertorus._tensor_train import TensorTrain


class TestTensorTrainConvolutionTargetShape(unittest.TestCase):
    def setUp(self):
        coefficients = np.array([1.0, 2.0, 3.0])
        self.left = TensorTrain.from_dense(coefficients)
        self.right = TensorTrain.from_dense(coefficients)

    def test_rejects_non_integer_target_mode_sizes(self):
        for invalid in (3.5, True, np.bool_(False), "3"):
            with self.subTest(invalid=invalid):
                with self.assertRaises(TypeError):
                    self.left.coefficient_convolution(
                        self.right, target_shape=(invalid,)
                    )

    def test_rejects_non_positive_target_mode_sizes(self):
        for invalid in (0, -1):
            with self.subTest(invalid=invalid):
                with self.assertRaises(ValueError):
                    self.left.coefficient_convolution(
                        self.right, target_shape=(invalid,)
                    )

    def test_accepts_python_and_numpy_integer_target_mode_sizes(self):
        for valid in (3, np.int64(3)):
            with self.subTest(valid=valid):
                result = self.left.coefficient_convolution(
                    self.right, target_shape=(valid,)
                )
                self.assertEqual(result.shape, (3,))


if __name__ == "__main__":
    unittest.main()
