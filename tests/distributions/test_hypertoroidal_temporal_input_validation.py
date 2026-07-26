import datetime

import numpy as np
import pytest
from pyrecest.distributions.circle.circular_uniform_distribution import (
    CircularUniformDistribution,
)
from pyrecest.distributions.hypertorus._input_validation import (
    as_hypertoroidal_points,
    as_shift_vector,
)


@pytest.mark.parametrize(
    "value",
    [
        np.datetime64(1, "ns"),
        np.timedelta64(1, "ns"),
        datetime.datetime(2026, 1, 1),
        datetime.timedelta(seconds=1),
        np.array([np.datetime64(1, "ns")], dtype=object),
    ],
)
@pytest.mark.parametrize(
    "validate",
    [
        pytest.param(lambda value: as_shift_vector(value, 1), id="shift-vector"),
        pytest.param(
            lambda value: as_hypertoroidal_points(value, 1),
            id="evaluation-points",
        ),
    ],
)
def test_hypertoroidal_angle_helpers_reject_temporal_values(value, validate) -> None:
    with pytest.raises(ValueError, match="temporal"):
        validate(value)


def test_circular_uniform_shift_rejects_temporal_angle() -> None:
    distribution = CircularUniformDistribution()

    with pytest.raises(ValueError, match="temporal"):
        distribution.shift(np.timedelta64(1, "ns"))
