"""Regression tests for exact Gauss-von-Mises sample-count validation."""

from fractions import Fraction

import pytest
from pyrecest.distributions.cart_prod.gauss_von_mises_distribution import (
    _validate_positive_sample_count,
)


def test_rejects_fraction_rounded_to_integer_by_binary64():
    rounded_half_integer = Fraction(2**54 + 1, 2)

    with pytest.raises(ValueError, match="finite integer"):
        _validate_positive_sample_count(rounded_half_integer)


def test_preserves_exact_large_integer():
    exact_integer = Fraction(2**54 + 2, 2)

    assert _validate_positive_sample_count(exact_integer) == 2**53 + 1
