from fractions import Fraction

import pytest

from pyrecest.distributions.cart_prod.mardia_sutton_distribution import (
    _validate_positive_sample_count,
)


def test_mardia_sutton_sample_count_rejects_fraction_rounded_by_float():
    count = Fraction(2**54 + 1, 2)

    assert float(count).is_integer()
    with pytest.raises(ValueError, match="finite integer"):
        _validate_positive_sample_count(count)


def test_mardia_sutton_sample_count_accepts_exact_integral_fraction():
    assert _validate_positive_sample_count(Fraction(6, 3)) == 2
