import numpy as np
import pytest
from pyrecest.distributions.circle.piecewise_constant_distribution import (
    PiecewiseConstantDistribution,
)


@pytest.mark.parametrize(
    "helper_name",
    ("left_border", "right_border", "interval_center"),
)
@pytest.mark.parametrize(
    ("m", "n"),
    (
        (np.timedelta64(1, "ns"), 4),
        (1, np.timedelta64(4, "ns")),
    ),
)
def test_interval_helpers_reject_temporal_indices(helper_name, m, n):
    helper = getattr(PiecewiseConstantDistribution, helper_name)

    with pytest.raises(ValueError, match="m and n must be integers"):
        helper(m, n)


def test_interval_helpers_preserve_numpy_integer_indices():
    assert PiecewiseConstantDistribution.left_border(np.int64(1), np.int64(4)) == 0.0
