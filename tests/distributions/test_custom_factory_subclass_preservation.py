import unittest

from pyrecest.distributions.cart_prod.custom_hypercylindrical_distribution import (
    CustomHypercylindricalDistribution,
)
from pyrecest.distributions.conversion import convert_distribution
from pyrecest.distributions.hypersphere_subset.custom_hemispherical_distribution import (
    CustomHemisphericalDistribution,
)
from pyrecest.distributions.hypersphere_subset.custom_hyperhemispherical_distribution import (
    CustomHyperhemisphericalDistribution,
)
from pyrecest.distributions.hypersphere_subset.custom_hyperspherical_distribution import (
    CustomHypersphericalDistribution,
)
from pyrecest.distributions.nonperiodic.custom_linear_distribution import (
    CustomLinearDistribution,
)


def _constant_pdf(xs):
    return xs[..., 0] * 0.0 + 1.0


class _CustomLinearSubclass(CustomLinearDistribution):
    pass


class _CustomHypercylindricalSubclass(CustomHypercylindricalDistribution):
    pass


class _CustomHypersphericalSubclass(CustomHypersphericalDistribution):
    pass


class _CustomHyperhemisphericalSubclass(CustomHyperhemisphericalDistribution):
    pass


class _CustomHemisphericalSubclass(CustomHemisphericalDistribution):
    pass


class CustomFactorySubclassPreservationTest(unittest.TestCase):
    def test_custom_linear_conversion_preserves_requested_subclass(self):
        source = CustomLinearDistribution(_constant_pdf, dim=1)

        converted = convert_distribution(source, _CustomLinearSubclass)

        self.assertIsInstance(converted, _CustomLinearSubclass)

    def test_custom_hypercylindrical_conversion_preserves_requested_subclass(self):
        source = CustomHypercylindricalDistribution(
            _constant_pdf, bound_dim=1, lin_dim=1
        )

        converted = convert_distribution(source, _CustomHypercylindricalSubclass)

        self.assertIsInstance(converted, _CustomHypercylindricalSubclass)

    def test_custom_hyperspherical_conversion_preserves_requested_subclass(self):
        source = CustomHypersphericalDistribution(_constant_pdf, dim=2)

        converted = convert_distribution(source, _CustomHypersphericalSubclass)

        self.assertIsInstance(converted, _CustomHypersphericalSubclass)

    def test_custom_hyperhemispherical_conversion_preserves_requested_subclass(self):
        source = CustomHyperhemisphericalDistribution(_constant_pdf, dim=2)

        converted = convert_distribution(source, _CustomHyperhemisphericalSubclass)

        self.assertIsInstance(converted, _CustomHyperhemisphericalSubclass)

    def test_custom_hemispherical_conversion_preserves_requested_subclass(self):
        source = CustomHemisphericalDistribution(_constant_pdf)

        converted = convert_distribution(source, _CustomHemisphericalSubclass)

        self.assertIsInstance(converted, _CustomHemisphericalSubclass)


if __name__ == "__main__":
    unittest.main()
