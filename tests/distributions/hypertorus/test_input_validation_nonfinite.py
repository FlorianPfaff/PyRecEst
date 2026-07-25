import numpy as np
import pytest

from pyrecest.distributions.hypertorus._input_validation import (
    as_hypertoroidal_points,
    as_shift_vector,
)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
@pytest.mark.parametrize(
    "validate",
    [
        pytest.param(
            lambda value: as_shift_vector([0.0, value], dim=2),
            id="shift-vector",
        ),
        pytest.param(
            lambda value: as_hypertoroidal_points(
                [[0.0, 1.0], [value, 2.0]],
                dim=2,
            ),
            id="evaluation-points",
        ),
    ],
)
def test_angle_inputs_reject_nonfinite_values(invalid, validate) -> None:
    with pytest.raises(ValueError, match="finite real angles"):
        validate(invalid)


def test_finite_inputs_keep_existing_shapes() -> None:
    shift = as_shift_vector([0.1, 0.2], dim=2)
    points = as_hypertoroidal_points([[0.1, 0.2], [0.3, 0.4]], dim=2)

    assert shift.shape == (2,)
    assert points.shape == (2, 2)
