from fractions import Fraction

import pytest

from pyrecest.evaluation.get_distance_function import _validate_symmetry_count


def test_symmetry_count_rejects_fraction_rounded_to_integer_by_float():
    fractionally_rounded = Fraction(2**54 + 1, 2)

    assert float(fractionally_rounded).is_integer()
    with pytest.raises(
        ValueError,
        match="nSymm must be a finite positive integer",
    ):
        _validate_symmetry_count(fractionally_rounded)


def test_symmetry_count_preserves_exact_large_integer_values():
    exact_integer = Fraction(2**54, 2)

    assert _validate_symmetry_count(exact_integer) == 2**53
