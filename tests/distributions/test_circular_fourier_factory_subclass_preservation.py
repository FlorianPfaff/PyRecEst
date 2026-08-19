import unittest

# pylint: disable=no-name-in-module,no-member
from pyrecest.backend import array
from pyrecest.distributions.circle.circular_dirac_distribution import (
    CircularDiracDistribution,
)
from pyrecest.distributions.circle.circular_fourier_distribution import (
    CircularFourierDistribution,
)
from pyrecest.distributions.circle.von_mises_distribution import VonMisesDistribution
from pyrecest.distributions.conversion import convert_distribution


class _CircularFourierSubclass(CircularFourierDistribution):
    pass


class CircularFourierFactorySubclassPreservationTest(unittest.TestCase):
    def test_conversion_factory_preserves_requested_subclass_for_density_source(self):
        source = VonMisesDistribution(0.3, 2.0)

        converted = convert_distribution(
            source,
            _CircularFourierSubclass,
            n=9,
            transformation="identity",
            store_values_multiplied_by_n=False,
        )

        self.assertIsInstance(converted, _CircularFourierSubclass)

    def test_conversion_factory_preserves_requested_subclass_for_dirac_source(self):
        source = CircularDiracDistribution(array([0.0, 1.0]))

        converted = convert_distribution(
            source,
            _CircularFourierSubclass,
            n=9,
            transformation="identity",
            store_values_multiplied_by_n=False,
        )

        self.assertIsInstance(converted, _CircularFourierSubclass)

    def test_function_value_factory_preserves_requested_subclass(self):
        function_values = array([1.0, 0.9, 0.8, 0.7, 0.6, 0.7, 0.8, 0.9, 1.0])

        converted = _CircularFourierSubclass.from_function_values(
            function_values,
            transformation="identity",
            store_values_multiplied_by_n=False,
        )

        self.assertIsInstance(converted, _CircularFourierSubclass)


if __name__ == "__main__":
    unittest.main()
