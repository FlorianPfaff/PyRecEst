import unittest

# pylint: disable=no-name-in-module,no-member,redefined-builtin
from pyrecest import backend
from pyrecest.backend import (
    allclose,
    array,
    diag,
    is_array,
    random,
)
from pyrecest.backend import sum as backend_sum
from pyrecest.backend import (
    to_numpy,
)
from pyrecest.distributions import GaussianDistribution, LinearDiracDistribution
from pyrecest.distributions.conversion import (
    ConversionError,
    can_convert,
    convert_distribution,
)

SUPPORTED_BACKENDS = ("numpy", "pytorch", "jax")


def _as_bool(value) -> bool:
    if isinstance(value, (bool, int, float)):
        return bool(value)
    try:
        value_np = to_numpy(value)
    except AttributeError:
        return bool(value)
    if hasattr(value_np, "item"):
        return bool(value_np.item())
    return bool(value_np)


def _as_float(value) -> float:
    if isinstance(value, (bool, int, float)):
        return float(value)
    try:
        value_np = to_numpy(value)
    except AttributeError:
        return float(value)
    if hasattr(value_np, "item"):
        return float(value_np.item())
    return float(value_np)


class ConversionBackendCompatibilityTest(unittest.TestCase):
    """Conversion tests that are intentionally run under each CI backend.

    The repository test workflow executes the complete test suite with the
    ``numpy``, ``pytorch``, and ``jax`` backends. These tests exercise the
    conversion gateway in a backend-sensitive way: converted arrays must stay in
    the active backend representation, deterministic moment matching must agree
    numerically, and argument validation must not depend on the selected
    backend.
    """

    def setUp(self):
        self.backend_name = backend.get_backend_name()
        if self.backend_name not in SUPPORTED_BACKENDS:
            self.skipTest(
                f"Conversion backend compatibility is defined for {SUPPORTED_BACKENDS}, "
                f"not {self.backend_name!r}."
            )

    def test_gaussian_to_particles_alias_preserves_active_backend_arrays(self):
        random.seed(42)
        gaussian = GaussianDistribution(
            array([0.0, 1.0]),
            diag(array([1.0, 2.0])),
            check_validity=False,
        )

        particles = convert_distribution(gaussian, "particles", n_particles=11)

        self.assertIsInstance(particles, LinearDiracDistribution)
        self.assertTrue(is_array(particles.d), self.backend_name)
        self.assertTrue(is_array(particles.w), self.backend_name)
        self.assertEqual(tuple(particles.d.shape), (11, 2))
        self.assertEqual(tuple(particles.w.shape), (11,))
        self.assertAlmostEqual(_as_float(backend_sum(particles.w)), 1.0, places=6)

    def test_particles_to_gaussian_alias_preserves_active_backend_arrays(self):
        particles = LinearDiracDistribution(
            array([[0.0, 0.0], [2.0, 0.0], [2.0, 2.0]]),
            array([0.25, 0.25, 0.5]),
        )

        gaussian = convert_distribution(particles, "gaussian")

        self.assertIsInstance(gaussian, GaussianDistribution)
        self.assertTrue(is_array(gaussian.mean()), self.backend_name)
        self.assertTrue(is_array(gaussian.covariance()), self.backend_name)
        self.assertTrue(_as_bool(allclose(gaussian.mean(), particles.mean())))
        self.assertTrue(
            _as_bool(allclose(gaussian.covariance(), particles.covariance()))
        )

    def test_class_and_alias_targets_are_consistent_on_active_backend(self):
        random.seed(7)
        gaussian = GaussianDistribution(
            array([0.0, 0.0]),
            diag(array([1.0, 1.0])),
            check_validity=False,
        )

        by_class = convert_distribution(
            gaussian, LinearDiracDistribution, n_particles=5
        )
        by_alias = convert_distribution(gaussian, "particles", n_particles=5)

        self.assertIsInstance(by_class, LinearDiracDistribution)
        self.assertIsInstance(by_alias, LinearDiracDistribution)
        self.assertEqual(tuple(by_class.d.shape), tuple(by_alias.d.shape))
        self.assertTrue(is_array(by_class.d), self.backend_name)
        self.assertTrue(is_array(by_alias.d), self.backend_name)

    def test_conversion_route_detection_is_backend_independent(self):
        gaussian = GaussianDistribution(
            array([0.0, 0.0]),
            diag(array([1.0, 1.0])),
            check_validity=False,
        )
        particles = LinearDiracDistribution(
            array([[0.0, 0.0], [1.0, 1.0]]), array([0.5, 0.5])
        )

        self.assertTrue(can_convert(gaussian, "particles"))
        self.assertTrue(can_convert(gaussian, LinearDiracDistribution))
        self.assertTrue(can_convert(particles, "gaussian"))
        self.assertTrue(can_convert(particles, GaussianDistribution))
        self.assertFalse(can_convert(gaussian, "not_a_representation"))

    def test_conversion_argument_validation_is_backend_independent(self):
        gaussian = GaussianDistribution(
            array([0.0, 0.0]),
            diag(array([1.0, 1.0])),
            check_validity=False,
        )

        with self.assertRaises(ConversionError):
            convert_distribution(gaussian, "particles")

        with self.assertRaises(ConversionError):
            convert_distribution(
                gaussian,
                "particles",
                n_particles=5,
                unsupported_argument=True,
            )


if __name__ == "__main__":
    unittest.main()
