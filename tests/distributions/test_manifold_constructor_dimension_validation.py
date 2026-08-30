import unittest

from pyrecest.distributions.hypertorus.custom_hypertoroidal_distribution import (
    CustomHypertoroidalDistribution,
)


class TestManifoldConstructorDimensionValidation(unittest.TestCase):
    def test_custom_hypertoroidal_rejects_nonpositive_dimensions(self):
        for dim in (0, -1):
            with self.subTest(dim=dim), self.assertRaisesRegex(
                ValueError, "dim must be a positive integer"
            ):
                CustomHypertoroidalDistribution(lambda _xs: 1.0, dim)


if __name__ == "__main__":
    unittest.main()
