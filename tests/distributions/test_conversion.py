import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import allclose, array, eye, random
from pyrecest.distributions import (
    GaussianDistribution,
    GaussianMixture,
    LinearDiracDistribution,
)
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

        result = convert_distribution(gaussian, GaussianDistribution, return_info=True)

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

        particles = gaussian.approximate_as(LinearDiracDistribution, n_particles=25)

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

        with self.assertRaises(ConversionError) as context:
            convert_distribution(gaussian, "not_a_representation")
        message = str(context.exception)

        self.assertIn("Unknown conversion alias 'not_a_representation'", message)
        self.assertIn("Known built-in aliases", message)
        self.assertIn("Supported aliases for GaussianDistribution", message)
        self.assertIn("'particles'", message)
        self.assertIn("'gaussian'", message)

    def test_known_alias_not_supported_by_source_reports_valid_aliases(self):
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        with self.assertRaises(ConversionError) as context:
            convert_distribution(gaussian, "grid")
        message = str(context.exception)

        self.assertIn("Conversion alias 'grid' is known", message)
        self.assertIn("not supported for source type GaussianDistribution", message)
        self.assertIn("Supported aliases for GaussianDistribution", message)
        self.assertIn("'particles'", message)
        self.assertIn("'gaussian'", message)
        self.assertNotIn("Unknown conversion alias", message)

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

    def test_linear_dirac_from_distribution_accepts_n_samples_alias(self):
        random.seed(0)
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        particles = convert_distribution(
            gaussian, LinearDiracDistribution, n_samples=25
        )

        self.assertIsInstance(particles, LinearDiracDistribution)
        self.assertEqual(particles.d.shape[0], 25)

    def test_linear_dirac_rejects_conflicting_particle_count_aliases(self):
        gaussian = GaussianDistribution(array([0.0, 0.0]), eye(2))

        with self.assertRaises(ConversionError):
            convert_distribution(
                gaussian,
                LinearDiracDistribution,
                n_particles=5,
                n_samples=6,
            )

    def test_linear_dirac_set_mean_uses_current_mean_method(self):
        particles = LinearDiracDistribution(
            array([[0.0, 0.0], [2.0, 0.0]]), array([0.5, 0.5])
        )

        particles.set_mean(array([3.0, 1.0]))

        self.assertTrue(allclose(particles.mean(), array([3.0, 1.0])))

    def test_weighted_samples_default_weights_use_number_of_samples(self):
        samples = array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])

        mean, covariance = LinearDiracDistribution.weighted_samples_to_mean_and_cov(
            samples
        )

        self.assertTrue(allclose(mean, array([2.0, 0.0])))
        self.assertTrue(
            allclose(
                covariance,
                array([[8.0 / 3.0, 0.0], [0.0, 0.0]]),
            )
        )

    def test_gaussian_mixture_to_gaussian_moment_match(self):
        mixture = GaussianMixture(
            [
                GaussianDistribution(array([0.0]), array([[1.0]])),
                GaussianDistribution(array([2.0]), array([[1.0]])),
            ],
            array([0.25, 0.75]),
        )

        gaussian = convert_distribution(mixture, "gaussian")

        self.assertIsInstance(gaussian, GaussianDistribution)
        self.assertTrue(allclose(gaussian.mean(), array([1.5])))
        self.assertTrue(allclose(gaussian.covariance(), array([[1.75]])))

    def test_gaussian_mixture_to_linear_dirac_via_particles_alias(self):
        random.seed(0)
        mixture = GaussianMixture(
            [
                GaussianDistribution(array([0.0, 0.0]), eye(2)),
                GaussianDistribution(array([2.0, 0.0]), eye(2)),
            ],
            array([0.25, 0.75]),
        )

        particles = convert_distribution(mixture, "particles", n_samples=30)

        self.assertIsInstance(particles, LinearDiracDistribution)
        self.assertEqual(particles.d.shape, (30, 2))


if __name__ == "__main__":
    unittest.main()
