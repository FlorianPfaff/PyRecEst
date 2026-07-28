"""Regression tests for exact partially wrapped count validation."""

from fractions import Fraction

import numpy as np
import pytest
from pyrecest.distributions.cart_prod.partially_wrapped_normal_distribution import (
    _validate_nonnegative_wrap_count,
    _validate_positive_sample_count,
)


@pytest.mark.parametrize(
    ("validator", "message"),
    [
        (_validate_positive_sample_count, "n must be a finite integer"),
        (_validate_nonnegative_wrap_count, "m must be a finite integer"),
    ],
)
def test_rejects_fraction_rounded_to_integer_by_binary64(validator, message):
    rounded_half_integer = Fraction(2**54 + 1, 2)

    with pytest.raises(ValueError, match=message):
        validator(rounded_half_integer)


@pytest.mark.parametrize(
    "validator",
    [_validate_positive_sample_count, _validate_nonnegative_wrap_count],
)
def test_preserves_exact_large_integer(validator):
    exact_integer = Fraction(2**54 + 2, 2)

    assert validator(exact_integer) == 2**53 + 1


@pytest.mark.parametrize(
    "validator",
    [_validate_positive_sample_count, _validate_nonnegative_wrap_count],
)
@pytest.mark.parametrize(
    "invalid_count",
    [
        np.timedelta64(3, "ns"),
        np.datetime64(3, "ns"),
        np.array(np.timedelta64(3, "ns"), dtype=object),
        np.array(np.datetime64(3, "ns"), dtype=object),
        np.array(np.complex128(3 + 0j), dtype=object),
    ],
)
def test_rejects_nonreal_or_temporal_counts(validator, invalid_count):
    with pytest.raises(ValueError, match="finite integer"):
        validator(invalid_count)


@pytest.mark.parametrize(
    "validator",
    [_validate_positive_sample_count, _validate_nonnegative_wrap_count],
)
def test_preserves_object_wrapped_integer_counts(validator):
    assert validator(np.array(3, dtype=object)) == 3
