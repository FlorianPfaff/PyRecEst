from fractions import Fraction

import pytest
from pyrecest.filters.euclidean_particle_filter import EuclideanParticleFilter
from pyrecest.filters.hypertoroidal_particle_filter import _validate_positive_integer


def _validate_euclidean(value):
    return EuclideanParticleFilter._validate_positive_int(value, "count")


def _validate_hypertoroidal(value):
    return _validate_positive_integer(value, "count")


@pytest.mark.parametrize(
    "validator",
    [_validate_euclidean, _validate_hypertoroidal],
    ids=["euclidean", "hypertoroidal"],
)
def test_particle_filter_count_validation_is_exact(validator):
    rounded_half_integer = Fraction(2**54 + 1, 2)
    exact_large_integer = Fraction(2**54 + 2, 2)

    with pytest.raises(ValueError, match="positive integer"):
        validator(rounded_half_integer)

    assert validator(exact_large_integer) == 2**53 + 1
