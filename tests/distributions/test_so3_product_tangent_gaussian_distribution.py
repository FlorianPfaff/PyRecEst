import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import (
    array,
    diag,
    eye,
    linalg,
    ones,
    pi,
    random,
    sqrt,
)
from pyrecest.distributions import SO3ProductTangentGaussianDistribution
from tests.distributions.so3_test_helpers import (
    ATOL,
    assert_pdf_peak_matches_log_pdf,
    x_quaternion,
    z_quaternion,
    z_rotation,
)


class SO3ProductTangentGaussianDistributionTest(unittest.TestCase):
    def test_constructor_normalizes_flat_mean_and_covariance(self):
        covariance = diag(array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
        mean = array([0.0, 0.0, 0.0, -2.0, 0.0, 0.0, sqrt(0.5), sqrt(0.5)])

        dist = SO3ProductTangentGaussianDistribution(mean, covariance)

        self.assertEqual(dist.num_rotations, 2)
        self.assertEqual(dist.dim, 6)
        self.assertEqual(dist.input_dim, 8)
        npt.assert_allclose(dist.mean()[0], array([0.0, 0.0, 0.0, 1.0]), atol=ATOL)
        npt.assert_allclose(dist.covariance(), covariance, atol=ATOL)
        npt.assert_allclose(dist.get_manifold_size(), pi**4, atol=ATOL)
        self.assertTrue(dist.is_valid())

    def test_constructor_rejects_invalid_covariance(self):
        mean = array([z_quaternion(0.0), x_quaternion(0.0)])

        invalid_cases = [
            (diag(array([0.1, 0.2, 0.3])), "shape"),
            (
                diag(array([0.1, 0.2, 0.3, float("nan"), 0.5, 0.6])),
                "finite",
            ),
            (
                array(
                    [
                        [0.1, 0.2, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.2, 0.0, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.3, 0.0, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.4, 0.0, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.5, 0.0],
                        [0.0, 0.0, 0.0, 0.0, 0.0, 0.6],
                    ]
                ),
                "symmetric",
            ),
            (diag(array([0.1, 0.2, 0.3, 0.0, 0.5, 0.6])), "positive definite"),
        ]

        for covariance, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    SO3ProductTangentGaussianDistribution(mean, covariance)

    def test_exp_log_roundtrip_with_base_product_rotation(self):
        base = array(
            [
                z_quaternion(pi / 3.0),
                x_quaternion(pi / 5.0),
            ]
        )
        tangent_vectors = array(
            [
                [0.1, -0.2, 0.05, 0.02, 0.0, -0.03],
                [0.0, 0.0, 0.0, -0.1, 0.2, -0.05],
            ]
        )

        rotations = SO3ProductTangentGaussianDistribution.exp_map(
            tangent_vectors, base=base
        )
        roundtrip = SO3ProductTangentGaussianDistribution.log_map(rotations, base=base)

        self.assertEqual(rotations.shape, (2, 2, 4))
        npt.assert_allclose(roundtrip, tangent_vectors, atol=ATOL)

    def test_pdf_and_ln_pdf_peak_at_mode(self):
        covariance = diag(array([0.2, 0.3, 0.4, 0.5, 0.6, 0.7]))
        dist = SO3ProductTangentGaussianDistribution(
            array([z_quaternion(0.0), x_quaternion(0.0)]), covariance
        )
        offset = SO3ProductTangentGaussianDistribution.exp_map(
            array([0.4, 0.0, 0.0, 0.0, -0.2, 0.0]), base=dist.mean()
        )

        assert_pdf_peak_matches_log_pdf(self, dist, covariance, 6, offset)

    def test_sampling_returns_unit_product_quaternions(self):
        random.seed(0)
        dist = SO3ProductTangentGaussianDistribution.from_covariance_diagonal(
            array([z_quaternion(0.0), x_quaternion(0.0)]),
            array([0.01, 0.01, 0.01, 0.02, 0.02, 0.02]),
        )

        samples = dist.sample(8)
        single_sample = dist.sample(1)

        self.assertEqual(samples.shape, (8, 2, 4))
        self.assertEqual(single_sample.shape, (1, 2, 4))
        npt.assert_allclose(linalg.norm(samples, axis=-1), ones((8, 2)), atol=ATOL)
        npt.assert_allclose(
            linalg.norm(single_sample, axis=-1), ones((1, 2)), atol=ATOL
        )

    def test_sampling_accepts_integer_like_count(self):
        dist = SO3ProductTangentGaussianDistribution.from_covariance_diagonal(
            array([z_quaternion(0.0), x_quaternion(0.0)]),
            array([0.01, 0.01, 0.01, 0.02, 0.02, 0.02]),
        )

        samples = dist.sample(np.array(3.0))
        tangent_samples = dist.sample_tangent(np.int64(3))

        self.assertEqual(samples.shape, (3, 2, 4))
        self.assertEqual(tangent_samples.shape, (3, 6))

    def test_sampling_rejects_invalid_count(self):
        dist = SO3ProductTangentGaussianDistribution.from_covariance_diagonal(
            array([z_quaternion(0.0), x_quaternion(0.0)]),
            array([0.01, 0.01, 0.01, 0.02, 0.02, 0.02]),
        )

        for n in (0, -1, 2.5, True, [3]):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    dist.sample(n)
                with self.assertRaises(ValueError):
                    dist.sample_tangent(n)

    def test_geodesic_distance_component_and_sum(self):
        identity_product = array([z_quaternion(0.0), x_quaternion(0.0)])
        rotated_product = array([z_quaternion(pi / 2.0), x_quaternion(pi / 4.0)])

        component_distances = SO3ProductTangentGaussianDistribution.geodesic_distance(
            identity_product, rotated_product, reduce=False
        )
        summed_distances = SO3ProductTangentGaussianDistribution.geodesic_distance(
            identity_product, rotated_product
        )

        npt.assert_allclose(
            component_distances, array([[pi / 2.0, pi / 4.0]]), atol=ATOL
        )
        npt.assert_allclose(summed_distances, array([3.0 * pi / 4.0]), atol=ATOL)

    def test_marginalize_rotations(self):
        covariance = diag(array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6]))
        dist = SO3ProductTangentGaussianDistribution(
            array([z_quaternion(0.0), x_quaternion(pi / 2.0)]), covariance
        )

        marginal = dist.marginalize_rotation(1)

        self.assertIsInstance(marginal, SO3ProductTangentGaussianDistribution)
        self.assertEqual(marginal.num_rotations, 1)
        self.assertEqual(marginal.dim, 3)
        npt.assert_allclose(marginal.mean()[0], x_quaternion(pi / 2.0), atol=ATOL)
        npt.assert_allclose(
            marginal.covariance(), diag(array([0.4, 0.5, 0.6])), atol=ATOL
        )

    def test_mean_rotation_matrices(self):
        dist = SO3ProductTangentGaussianDistribution.from_covariance_diagonal(
            array([z_quaternion(0.0), z_quaternion(pi / 2.0)]),
            array([0.1, 0.1, 0.1, 0.1, 0.1, 0.1]),
        )
        expected_rotation_matrices = array([eye(3), z_rotation(pi / 2.0)])

        npt.assert_allclose(
            dist.mean_rotation_matrices(), expected_rotation_matrices, atol=ATOL
        )


if __name__ == "__main__":
    unittest.main()
