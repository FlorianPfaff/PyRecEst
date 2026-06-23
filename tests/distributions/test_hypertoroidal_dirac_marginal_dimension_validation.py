import unittest

from pyrecest.backend import array
from pyrecest.distributions import HypertoroidalDiracDistribution


class TestHypertoroidalDiracMarginalDimensionValidation(unittest.TestCase):
    def setUp(self):
        self.dist = HypertoroidalDiracDistribution(
            array([[0.5, 2.0, 0.5], [3.0, 2.0, 0.2]]),
            array([0.25, 0.75]),
        )

    def test_marginalize_to_1d_rejects_invalid_dimension_indices(self):
        for dimension in (-1, self.dist.dim, True, 1.5):
            with self.subTest(dimension=dimension):
                with self.assertRaisesRegex(ValueError, "dimension"):
                    self.dist.marginalize_to_1D(dimension)

    def test_marginalize_out_rejects_invalid_dimension_indices(self):
        invalid_dimensions = (-1, self.dist.dim, True, [0, 1.5], [0, True])

        for dimensions in invalid_dimensions:
            with self.subTest(dimensions=dimensions):
                with self.assertRaisesRegex(ValueError, "dimension"):
                    self.dist.marginalize_out(dimensions)


if __name__ == "__main__":
    unittest.main()
