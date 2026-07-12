import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions.abstract_orthogonal_basis_distribution import (
    AbstractOrthogonalBasisDistribution,
)


class DummyOrthogonalBasisDistribution(AbstractOrthogonalBasisDistribution):
    def __init__(self, values, transformation="sqrt"):
        self._values = values
        self.normalization_calls = 0
        super().__init__(coeff_mat=array([1.0]), transformation=transformation)

    def normalize_in_place(self):
        self.normalization_calls += 1

    def value(self, xs):  # pylint: disable=unused-argument
        return self._values


class AbstractOrthogonalBasisDistributionTest(unittest.TestCase):
    def test_identity_pdf_rejects_large_negative_imaginary_part(self):
        dist = DummyOrthogonalBasisDistribution(array([1.0 - 1j]), "identity")

        with self.assertRaises(ValueError):
            dist.pdf(array([0.0]))

    def test_sqrt_pdf_uses_complex_modulus_squared(self):
        dist = DummyOrthogonalBasisDistribution(array([2.0 - 1j]), "sqrt")

        self.assertAlmostEqual(float(dist.pdf(array([0.0]))[0]), 5.0)

    def test_normalize_returns_independent_copy_when_in_place_returns_none(self):
        dist = DummyOrthogonalBasisDistribution(array([1.0]), "identity")

        normalized = dist.normalize()

        self.assertIsInstance(normalized, DummyOrthogonalBasisDistribution)
        self.assertIsNot(normalized, dist)
        self.assertEqual(dist.normalization_calls, 1)
        self.assertEqual(normalized.normalization_calls, 2)


if __name__ == "__main__":
    unittest.main()
