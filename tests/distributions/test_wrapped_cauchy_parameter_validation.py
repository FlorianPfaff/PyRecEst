import unittest

from pyrecest.backend import array
from pyrecest.distributions.circle.wrapped_cauchy_distribution import (
    WrappedCauchyDistribution,
)


class WrappedCauchyParameterValidationTest(unittest.TestCase):
    def test_rejects_non_real_scalar_parameters(self):
        invalid_values = (True, array([True]), "0.5", 0.5 + 0.0j)
        for invalid in invalid_values:
            with self.subTest(mu=invalid), self.assertRaisesRegex(ValueError, "mu"):
                WrappedCauchyDistribution(invalid, 0.5)
            with self.subTest(gamma=invalid), self.assertRaisesRegex(
                ValueError, "gamma"
            ):
                WrappedCauchyDistribution(0.0, invalid)


if __name__ == "__main__":
    unittest.main()
