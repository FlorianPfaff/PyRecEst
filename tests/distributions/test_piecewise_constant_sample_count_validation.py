"""Regression tests for piecewise-constant sample-count validation."""

import numpy as np
import pytest
from pyrecest.distributions.circle.piecewise_constant_distribution import (
    _validate_positive_sample_count,
)


@pytest.mark.parametrize(
    "count",
    [
        "2",
        b"2",
        np.str_("2"),
        np.bytes_(b"2"),
        np.timedelta64(2, "ns"),
        np.datetime64("1970-01-01T00:00:00.000000002"),
        np.asarray(np.timedelta64(2, "ns")),
        np.asarray(np.datetime64("1970-01-01T00:00:00.000000002")),
        np.array(np.timedelta64(2, "ns"), dtype=object),
        np.array(
            np.datetime64("1970-01-01T00:00:00.000000002"),
            dtype=object,
        ),
    ],
)
def test_piecewise_sample_count_rejects_text_and_temporal_values(count):
    with pytest.raises(ValueError, match="integer"):
        _validate_positive_sample_count(count)


def test_piecewise_sample_count_keeps_integer_like_numeric_values():
    assert _validate_positive_sample_count(np.array(2.0)) == 2
    assert _validate_positive_sample_count(np.int64(3)) == 3
