import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, eye, random
from pyrecest.distributions import GaussianDistribution, LinearDiracDistribution
from pyrecest.distributions.conversion import (
    ConversionError,
    ConversionResult,
    can_convert,
    convert_distribution,
    register_conversion_alias,
)
from pyrecest.distributions.so3_dirac_distribution import SO3DiracDistribution
from pyrecest.distributions.so3_product_dirac_distribution import (
    SO3ProductDiracDistribution,
)
from pyrecest.distributions.so3_product_tangent_gaussian_distribution import (
    SO3ProductTangentGaussianDistribution,
)
from pyrecest.distributions.so3_tangent_gaussian_distribution import (
    SO3TangentGaussianDistribution,
)


class ConversionTest(unittest.TestCase):
    def test_convert_distribution_uses_target_from_distribution(self):
        random.seed(0)
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        particles = convert_distribution(
            gaussian, LinearDiracDistribution, n_particles=25
        )

        self.assertIsInstance(particles, LinearDiracDistribution)
        self.assertEqual(particles.d.shape[0], 25)

    def test_return_info(self):
        random.seed(0)
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        result = convert_distribution(
            gaussian,
            LinearDiracDistribution,
            n_particles=5,
            return_info=True,
        )

        self.assertIsInstance(result, ConversionResult)
        self.assertIsInstance(result.distribution, LinearDiracDistribution)
        self.assertEqual(result.source_type, GaussianDistribution)
        self.assertEqual(result.target_type, LinearDiracDistribution)
        self.assertEqual(result.method, "LinearDiracDistribution.from_distribution")
        self.assertFalse(result.exact)

    def test_identity_conversion_is_exact(self):
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        result = convert_distribution(
            gaussian, GaussianDistribution, return_info=True
        )

        self.assertIs(result.distribution, gaussian)
        self.assertTrue(result.exact)
        self.assertEqual(result.method, "identity")

    def test_can_convert_reports_route_only(self):
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        self.assertTrue(can_convert(gaussian, LinearDiracDistribution))

    def test_missing_required_conversion_argument_raises_helpful_error(self):
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        with self.assertRaises(ConversionError):
            convert_distribution(gaussian, LinearDiracDistribution)

    def test_unknown_conversion_argument_raises_helpful_error(self):
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        with self.assertRaises(ConversionError):
            convert_distribution(
                gaussian,
                LinearDiracDistribution,
                n_particles=5,
                wrong_name=True,
            )

    def test_manifold_specific_distribution_supports_approximate_as(self):
        random.seed(0)
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        particles = gaussian.approximate_as(
            LinearDiracDistribution, n_particles=25
        )

        self.assertIsInstance(particles, LinearDiracDistribution)
        self.assertEqual(particles.d.shape[0], 25)

    def test_linear_dirac_to_gaussian_uses_moment_matching(self):
        particles = LinearDiracDistribution(
            array([[0.0, 0.0], [2.0, 0.0]]), array([0.5, 0.5])
        )

        gaussian = convert_distribution(particles, GaussianDistribution)

        self.assertIsInstance(gaussian, GaussianDistribution)
        self.assertTrue(allclose(gaussian.mean(), particles.mean()))
        self.assertTrue(allclose(gaussian.covariance(), particles.covariance()))

    def test_builtin_string_alias_particles(self):
        random.seed(0)
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        particles = convert_distribution(gaussian, "particles", n_particles=25)

        self.assertIsInstance(particles, LinearDiracDistribution)
        self.assertEqual(particles.d.shape[0], 25)

    def test_builtin_string_alias_gaussian(self):
        particles = LinearDiracDistribution(
            array([[0.0, 0.0], [2.0, 0.0]]), array([0.5, 0.5])
        )

        gaussian = particles.approximate_as("gaussian")

        self.assertIsInstance(gaussian, GaussianDistribution)
        self.assertTrue(allclose(gaussian.mean(), particles.mean()))

    def test_custom_string_alias(self):
        random.seed(0)
        register_conversion_alias("test_particles", LinearDiracDistribution)
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        particles = convert_distribution(gaussian, "test_particles", n_particles=5)

        self.assertIsInstance(particles, LinearDiracDistribution)

    def test_unknown_string_alias_raises_helpful_error(self):
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        with self.assertRaises(ConversionError):
            convert_distribution(gaussian, "not_a_representation")

    def test_can_convert_supports_string_aliases(self):
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        self.assertTrue(can_convert(gaussian, "particles"))
        self.assertFalse(can_convert(gaussian, "not_a_representation"))

    def test_so3_tangent_gaussian_to_dirac_alias(self):
        random.seed(0)
        distribution = SO3TangentGaussianDistribution(
            array([0.0, 0.0, 0.0, 1.0]), 0.01 * eye(3)
        )

        particles = convert_distribution(distribution, "particles", n_particles=8)

        self.assertIsInstance(particles, SO3DiracDistribution)
        self.assertEqual(particles.d.shape, (8, 4))
        self.assertTrue(particles.is_valid())

    def test_so3_dirac_to_tangent_gaussian_alias(self):
        base = array([0.0, 0.0, 0.0, 1.0])
        rotations = SO3TangentGaussianDistribution.exp_map(
            array(
                [
                    [0.01, 0.0, 0.0],
                    [-0.01, 0.0, 0.0],
                    [0.0, 0.02, 0.0],
                    [0.0, -0.02, 0.0],
                ]
            ),
            base=base,
        )
        particles = SO3DiracDistribution(rotations, array([0.25, 0.25, 0.25, 0.25]))

        gaussian = particles.approximate_as("so3_tangent_gaussian")

        self.assertIsInstance(gaussian, SO3TangentGaussianDistribution)
        self.assertEqual(gaussian.mean().shape, (4,))
        self.assertEqual(gaussian.covariance().shape, (3, 3))
        self.assertTrue(gaussian.is_valid())

    def test_so3_generic_gaussian_alias_is_tangent_gaussian(self):
        base = array([0.0, 0.0, 0.0, 1.0])
        rotations = SO3TangentGaussianDistribution.exp_map(
            array([[0.01, 0.0, 0.0], [-0.01, 0.0, 0.0]]),
            base=base,
        )
        particles = SO3DiracDistribution(rotations, array([0.5, 0.5]))

        gaussian = convert_distribution(
            particles, "gaussian", covariance_regularization=1e-9
        )

        self.assertIsInstance(gaussian, SO3TangentGaussianDistribution)
        self.assertEqual(gaussian.covariance().shape, (3, 3))

    def test_so3_product_tangent_gaussian_to_dirac_alias(self):
        random.seed(0)
        mean = array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
        distribution = SO3ProductTangentGaussianDistribution(mean, 0.01 * eye(6))

        particles = convert_distribution(distribution, "particles", n_particles=8)

        self.assertIsInstance(particles, SO3ProductDiracDistribution)
        self.assertEqual(particles.d.shape, (8, 2, 4))
        self.assertEqual(particles.num_rotations, 2)
        self.assertTrue(particles.is_valid())

    def test_so3_product_dirac_to_tangent_gaussian_alias(self):
        mean = array([[0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0]])
        rotations = SO3ProductTangentGaussianDistribution.exp_map(
            array(
                [
                    [0.01, 0.0, 0.0, 0.0, 0.02, 0.0],
                    [-0.01, 0.0, 0.0, 0.0, -0.02, 0.0],
                    [0.0, 0.01, 0.0, 0.02, 0.0, 0.0],
                    [0.0, -0.01, 0.0, -0.02, 0.0, 0.0],
                ]
            ),
            base=mean,
            num_rotations=2,
        )
        particles = SO3ProductDiracDistribution(
            rotations, array([0.25, 0.25, 0.25, 0.25])
        )

        gaussian = convert_distribution(particles, "so3_product_tangent_gaussian")

        self.assertIsInstance(gaussian, SO3ProductTangentGaussianDistribution)
        self.assertEqual(gaussian.mean().shape, (2, 4))
        self.assertEqual(gaussian.covariance().shape, (6, 6))
        self.assertEqual(gaussian.num_rotations, 2)
        self.assertTrue(gaussian.is_valid())


if __name__ == "__main__":
    unittest.main()
