"""Regression tests for exact multi-session integer validation."""

from fractions import Fraction

import pytest

from pyrecest.utils.multisession_assignment_score import (
    _normalize_index_matrix_fill_value,
    _normalize_max_gap,
)


@pytest.mark.parametrize(
    ("normalizer", "value", "message"),
    [
        (
            _normalize_max_gap,
            Fraction(2**54 + 1, 2),
            "max_gap must be a non-negative integer",
        ),
        (
            _normalize_index_matrix_fill_value,
            -Fraction(2**54 + 1, 2),
            "fill_value must be a negative integer",
        ),
    ],
)
def test_rejects_fraction_rounded_to_integer_by_binary64(normalizer, value, message):
    assert float(value).is_integer()

    with pytest.raises(ValueError, match=message):
        normalizer(value)


def test_preserves_exact_large_integer_values():
    exact_integer = Fraction(2**54, 2)

    assert _normalize_max_gap(exact_integer) == 2**53
    assert _normalize_index_matrix_fill_value(-exact_integer) == -(2**53)
