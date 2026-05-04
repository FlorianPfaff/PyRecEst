import unittest

from pyrecest import distributions

CONVERSION_EXPORTS = (
    "ConversionError",
    "ConversionResult",
    "can_convert",
    "convert_distribution",
    "register_conversion",
    "register_conversion_alias",
    "registered_conversion_aliases",
    "registered_conversions",
)


class ConversionExportTest(unittest.TestCase):
    def test_conversion_api_is_reexported_from_distributions_package(self):
        for name in CONVERSION_EXPORTS:
            with self.subTest(name=name):
                self.assertTrue(hasattr(distributions, name))
                self.assertIn(name, distributions.__all__)


if __name__ == "__main__":
    unittest.main()
