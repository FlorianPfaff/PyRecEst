from fractions import Fraction

import pytest
from pyrecest.distributions.cart_prod.cart_prod_stacked_distribution import (
    _validate_positive_sample_count,
)


def test_cart_prod_sample_count_rejects_fraction_rounded_by_float():
    count = Fraction(2**54 + 1, 2)

    assert float(count).is_integer()
    with pytest.raises(ValueError, match="finite integer"):
        _validate_positive_sample_count(count)


def test_cart_prod_sample_count_preserves_large_exact_integer():
    count = Fraction(2**54 + 2, 2)

    assert _validate_positive_sample_count(count) == 2**53 + 1
