import unittest

from pyrecest.backend import array
from pyrecest.distributions import GaussianMixture


class GaussianMixtureMomentWeightShapeTest(unittest.TestCase):
    def test_rejects_matrix_shaped_moment_weights(self):
        means = array([[0.0], [2.0]])
        covariance_matrices = array([[[1.0, 1.0]]])
        matrix_weights = (
            array([[1.0, 3.0]]),
            array([[1.0], [3.0]]),
        )

        for weights in matrix_weights:
            with self.subTest(shape=weights.shape):
                with self.assertRaisesRegex(ValueError, "one-dimensional"):
                    GaussianMixture.mixture_parameters_to_gaussian_parameters(
                        means,
                        covariance_matrices,
                        weights,
                    )

    def test_scalar_weight_remains_supported_for_single_component(self):
        mean, covariance = GaussianMixture.mixture_parameters_to_gaussian_parameters(
            array([[2.0]]),
            array([[[3.0]]]),
            array(4.0),
        )

        self.assertEqual(mean.shape, (1,))
        self.assertEqual(covariance.shape, (1, 1))
        self.assertEqual(float(mean[0]), 2.0)
        self.assertEqual(float(covariance[0, 0]), 3.0)


if __name__ == "__main__":
    unittest.main()
