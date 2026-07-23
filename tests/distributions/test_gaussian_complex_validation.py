import unittest

from pyrecest.backend import array
from pyrecest.distributions import GaussianDistribution


class GaussianComplexValidationTest(unittest.TestCase):
    def test_constructor_rejects_complex_parameters(self):
        invalid_parameters = (
            (array([1.0 + 2.0j]), array([[1.0]]), "mu"),
            (array([0.0]), array([[1.0 + 2.0j]]), "C"),
        )

        for mean, covariance, name in invalid_parameters:
            with self.subTest(name=name):
                with self.assertRaisesRegex(
                    ValueError, rf"{name} must contain only real values"
                ):
                    GaussianDistribution(mean, covariance)

    def test_pdf_rejects_complex_evaluation_points(self):
        distribution = GaussianDistribution(array([0.0]), array([[1.0]]))

        with self.assertRaisesRegex(ValueError, "xs must contain only real values"):
            distribution.pdf(array([1.0 + 2.0j]))

    def test_mean_mutators_reject_complex_values(self):
        distribution = GaussianDistribution(array([0.0]), array([[1.0]]))

        with self.assertRaisesRegex(
            ValueError, "new_mean must contain only real values"
        ):
            distribution.set_mean(array([1.0 + 2.0j]))
        with self.assertRaisesRegex(
            ValueError, "shift_by must contain only real values"
        ):
            distribution.shift(array([1.0 + 2.0j]))


if __name__ == "__main__":
    unittest.main()
