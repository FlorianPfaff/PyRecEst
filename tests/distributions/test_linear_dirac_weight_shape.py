import unittest

from pyrecest.backend import array
from pyrecest.distributions import LinearDiracDistribution


class LinearDiracWeightShapeTest(unittest.TestCase):
    def test_rejects_matrix_shaped_moment_weights(self):
        samples = array([0.0, 2.0])
        matrix_weights = (
            array([[1.0, 3.0]]),
            array([[1.0], [3.0]]),
        )

        for weights in matrix_weights:
            with self.subTest(shape=weights.shape):
                with self.assertRaisesRegex(ValueError, "one-dimensional"):
                    LinearDiracDistribution.weighted_samples_to_mean_and_cov(
                        samples,
                        weights,
                    )

    def test_scalar_weight_remains_supported_for_single_sample(self):
        mean, covariance = LinearDiracDistribution.weighted_samples_to_mean_and_cov(
            array(2.0),
            array(4.0),
        )

        self.assertEqual(mean.shape, (1,))
        self.assertEqual(covariance.shape, (1, 1))
        self.assertEqual(float(mean[0]), 2.0)
        self.assertEqual(float(covariance[0, 0]), 0.0)


if __name__ == "__main__":
    unittest.main()
