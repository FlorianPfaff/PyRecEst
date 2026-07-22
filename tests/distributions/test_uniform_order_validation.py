from fractions import Fraction

import pytest

from pyrecest.backend import array
from pyrecest.distributions.hypertorus.hypertoroidal_uniform_distribution import (
    HypertoroidalUniformDistribution,
    _validate_positive_sample_count,
)


_INEXACT_INTEGER_AFTER_FLOAT_ROUNDING = Fraction(2**54 + 1, 2)


def test_uniform_moment_rejects_invalid_order():
    dist = HypertoroidalUniformDistribution(2)

    for order in (
        "0",
        True,
        array([0]),
        1.5,
        _INEXACT_INTEGER_AFTER_FLOAT_ROUNDING,
    ):
        with pytest.raises(ValueError):
            dist.trigonometric_moment(order)


def test_uniform_sample_count_rejects_fraction_rounded_to_integer_float():
    with pytest.raises(ValueError):
        _validate_positive_sample_count(_INEXACT_INTEGER_AFTER_FLOAT_ROUNDING)


def test_uniform_integer_validation_accepts_exact_fraction():
    exact_integer = Fraction(6, 3)

    assert _validate_positive_sample_count(exact_integer) == 2
    moment = HypertoroidalUniformDistribution(2).trigonometric_moment(exact_integer)
    assert moment.shape == (2,)
