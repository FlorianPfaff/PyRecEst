import unittest

import pytest
from pyrecest import backend, distributions
from pyrecest.distributions import conversion
from pyrecest.distributions.nonperiodic import conversion as legacy_conversion
from tests.support.backend_runner import run_backend_code

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

    def test_legacy_conversion_api_reexports_canonical_objects(self):
        for name in CONVERSION_EXPORTS:
            with self.subTest(name=name):
                self.assertTrue(hasattr(legacy_conversion, name))
                self.assertIn(name, legacy_conversion.__all__)
                self.assertIs(
                    getattr(legacy_conversion, name), getattr(conversion, name)
                )

    @pytest.mark.backend_portable
    def test_legacy_registrations_are_visible_to_canonical_conversion_api(self):
        code = """
from pyrecest.distributions import conversion
from pyrecest.distributions.nonperiodic import conversion as legacy_conversion

class Source:
    def __init__(self, value):
        self.value = value

class Target:
    def __init__(self, value):
        self.value = value

def convert_source(source, *, increment=0):
    return Target(source.value + increment)

legacy_conversion.register_conversion(
    Source, Target, convert_source, exact=True, method="legacy_export_converter"
)
legacy_conversion.register_conversion_alias(
    "legacy_export_target",
    Target,
    default_kwargs={"increment": 3},
    description="Temporary legacy export test target",
)

source = Source(7)
assert conversion.can_convert(source, Target)
assert conversion.can_convert(source, "legacy_export_target")

for target, expected_value in ((Target, 7), ("legacy_export_target", 10)):
    result = conversion.convert_distribution(source, target, return_info=True)
    assert type(result) is conversion.ConversionResult
    assert isinstance(result.distribution, Target)
    assert result.distribution.value == expected_value
    assert result.source_type is Source
    assert result.target_type is Target
    assert result.exact is True

assert (Source, Target, "legacy_export_converter", True) in conversion.registered_conversions()
assert (
    "legacy_export_target", "Temporary legacy export test target"
) in conversion.registered_conversion_aliases()
assert legacy_conversion.registered_conversions() == conversion.registered_conversions()
assert legacy_conversion.registered_conversion_aliases() == conversion.registered_conversion_aliases()
"""
        result = run_backend_code(backend.get_backend_name(), code, timeout=60)
        self.assertEqual(result.returncode, 0, result.stderr)


if __name__ == "__main__":
    unittest.main()
