import unittest

from pyrecest.backend import array
from pyrecest.distributions.circle.wrapped_normal_distribution import (
    WrappedNormalDistribution,
)
from pyrecest.distributions.conversion import ConversionError, convert_distribution
from pyrecest.distributions.nonperiodic.gaussian_distribution import (
    GaussianDistribution,
)


class GaussianConversionContractTest(unittest.TestCase):
    def test_non_linear_distribution_without_covariance_raises_conversion_error(self):
        distribution = WrappedNormalDistribution(array(0.0), 0.5)

        with self.assertRaisesRegex(
            ConversionError,
            "requires the source distribution to expose mean\(\) and covariance\(\)",
        ):
            convert_distribution(distribution, GaussianDistribution)

    def test_gaussian_alias_reports_conversion_error_not_attribute_error(self):
        distribution = WrappedNormalDistribution(array(0.0), 0.5)

        with self.assertRaises(ConversionError):
            convert_distribution(distribution, "gaussian")


if __name__ == "__main__":
    unittest.main()
