import unittest

from pyrecest.backend import allclose, array
from pyrecest.distributions.conversion import convert_distribution
from pyrecest.distributions.nonperiodic.gaussian_distribution import (
    GaussianDistribution,
)
from pyrecest.distributions.nonperiodic.gaussian_mixture import GaussianMixture


class GaussianSubclass(GaussianDistribution):
    pass


class GaussianFactorySubclassPreservationTest(unittest.TestCase):
    def test_conversion_to_gaussian_subclass_preserves_requested_type(self):
        source = GaussianDistribution(
            array([1.0, -2.0]),
            array([[2.0, 0.25], [0.25, 1.0]]),
        )

        converted = convert_distribution(
            source,
            GaussianSubclass,
            check_validity=True,
        )

        self.assertIs(type(converted), GaussianSubclass)
        self.assertTrue(bool(allclose(converted.mu, source.mu)))
        self.assertTrue(bool(allclose(converted.C, source.C)))

    def test_mixture_conversion_to_gaussian_subclass_preserves_requested_type(self):
        mixture = GaussianMixture(
            [
                GaussianDistribution(array([0.0]), array([[1.0]])),
                GaussianDistribution(array([2.0]), array([[3.0]])),
            ],
            array([0.25, 0.75]),
        )
        expected = mixture.to_gaussian(check_validity=True)

        converted = convert_distribution(
            mixture,
            GaussianSubclass,
            check_validity=True,
        )

        self.assertIs(type(converted), GaussianSubclass)
        self.assertTrue(bool(allclose(converted.mu, expected.mu)))
        self.assertTrue(bool(allclose(converted.C, expected.C)))


if __name__ == "__main__":
    unittest.main()
