import unittest

import numpy as np
import numpy.testing as npt

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array, diag, linalg, ones, pi, random
from pyrecest.distributions import SO3TangentGaussianDistribution
from tests.distributions.so3_test_helpers import (
    ATOL,
    assert_matches_z_rotation,
    assert_pdf_peak_matches_log_pdf,
    z_quaternion,
)


class SO3TangentGaussianDistributionTest(unittest.TestCase):
    def test_constructor_normalizes_and_canonicalizes_mean(self):
        covariance = diag(array([0.1, 0.2, 0.3]))
        dist = SO3TangentGaussianDistribution(array([0.0, 0.0, 0.0, -2.0]), covariance)

        npt.assert_allclose(dist.mean(), array([0.0, 0.0, 0.0, 1.0]), atol=ATOL)
        npt.assert_allclose(dist.covariance(), covariance, atol=ATOL)
        self.assertTrue(dist.is_valid())

    def test_constructor_rejects_invalid_covariance(self):
        mean = array([0.0, 0.0, 0.0, 1.0])

        invalid_cases = [
            (array([0.1, 0.2, 0.3]), "shape"),
            (
                array([[0.1, 0.0, 0.0], [0.0, float("nan"), 0.0], [0.0, 0.0, 0.3]]),
                "finite",
            ),
            (array([[0.1, 0.2, 0.0], [0.0, 0.2, 0.0], [0.0, 0.0, 0.3]]), "symmetric"),
            (diag(array([0.1, 0.0, 0.3])), "positive definite"),
        ]

        for covariance, message in invalid_cases:
            with self.subTest(message=message):
                with self.assertRaisesRegex(ValueError, message):
                    SO3TangentGaussianDistribution(mean, covariance)

    def test_exp_log_roundtrip_with_base_rotation(self):
        base = z_quaternion(pi / 3.0)
        tangent_vectors = array([[0.1, -0.2, 0.05], [0.0, 0.0, 0.0]])

        rotations = SO3TangentGaussianDistribution.exp_map(tangent_vectors, base=base)
        roundtrip = SO3TangentGaussianDistribution.log_map(rotations, base=base)

        npt.assert_allclose(roundtrip, tangent_vectors, atol=ATOL)

    def test_pdf_and_ln_pdf_peak_at_mode(self):
        covariance = diag(array([0.2, 0.3, 0.4]))
        dist = SO3TangentGaussianDistribution(array([0.0, 0.0, 0.0, 1.0]), covariance)
        offset = SO3TangentGaussianDistribution.exp_map(array([0.4, 0.0, 0.0]))

        assert_pdf_peak_matches_log_pdf(self, dist, covariance, 3, offset)

    def test_sampling_returns_unit_quaternions(self):
        random.seed(0)
        dist = SO3TangentGaussianDistribution.from_covariance_diagonal(
            array([0.0, 0.0, 0.0, 1.0]), array([0.01, 0.01, 0.01])
        )

        samples = dist.sample(8)

        self.assertEqual(samples.shape, (8, 4))
        npt.assert_allclose(linalg.norm(samples, None, -1), ones(8), atol=ATOL)

    def test_sampling_accepts_integer_like_count(self):
        dist = SO3TangentGaussianDistribution.from_covariance_diagonal(
            array([0.0, 0.0, 0.0, 1.0]), array([0.01, 0.01, 0.01])
        )

        samples = dist.sample(np.array(3.0))
        tangent_samples = dist.sample_tangent(np.int64(3))

        self.assertEqual(samples.shape, (3, 4))
        self.assertEqual(tangent_samples.shape, (3, 3))

    def test_sampling_rejects_invalid_count(self):
        dist = SO3TangentGaussianDistribution.from_covariance_diagonal(
            array([0.0, 0.0, 0.0, 1.0]), array([0.01, 0.01, 0.01])
        )

        for n in (0, -1, 2.5, True, [3]):
            with self.subTest(n=n):
                with self.assertRaises(ValueError):
                    dist.sample(n)
                with self.assertRaises(ValueError):
                    dist.sample_tangent(n)

    def test_geodesic_distance_respects_antipodal_equivalence(self):
        identity = array([0.0, 0.0, 0.0, 1.0])
        identity_antipodal = array([0.0, 0.0, 0.0, -1.0])
        quarter_turn = z_quaternion(pi / 2.0)

        npt.assert_allclose(
            SO3TangentGaussianDistribution.geodesic_distance(
                identity, identity_antipodal
            ),
            array([0.0]),
            atol=ATOL,
        )
        npt.assert_allclose(
            SO3TangentGaussianDistribution.geodesic_distance(identity, quarter_turn),
            array([pi / 2.0]),
            atol=ATOL,
        )

    def test_mean_rotation_matrix_matches_quaternion(self):
        dist = SO3TangentGaussianDistribution(
            z_quaternion(pi / 2.0), diag(array([0.1, 0.1, 0.1]))
        )

        assert_matches_z_rotation(self, dist.mean_rotation_matrix(), pi / 2.0)


if __name__ == "__main__":
    unittest.main()
