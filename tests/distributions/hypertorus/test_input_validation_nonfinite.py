import numpy as np
import pytest

from pyrecest.distributions.hypertorus._input_validation import (
    as_hypertoroidal_points,
    as_shift_vector,
)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_shift_vector_rejects_nonfinite_angles(invalid) -> None:
    with pytest.raises(ValueError, match="finite real angles"):
        as_shift_vector([0.0, invalid], dim=2)


@pytest.mark.parametrize("invalid", [np.nan, np.inf, -np.inf])
def test_hypertoroidal_points_reject_nonfinite_angles(invalid) -> None:
    with pytest.raises(ValueError, match="finite real angles"):
        as_hypertoroidal_points([[0.0, 1.0], [invalid, 2.0]], dim=2)


def test_finite_inputs_keep_existing_shapes() -> None:
    shift = as_shift_vector([0.1, 0.2], dim=2)
    points = as_hypertoroidal_points([[0.1, 0.2], [0.3, 0.4]], dim=2)

    assert shift.shape == (2,)
    assert points.shape == (2, 2)
