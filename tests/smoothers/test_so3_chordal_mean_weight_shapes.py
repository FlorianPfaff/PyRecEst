"""Regression coverage for SO(3) chordal smoother weight-vector shapes."""

import pytest
from pyrecest.backend import eye
from pyrecest.smoothers import SO3ChordalMeanSmoother


@pytest.mark.parametrize(
    "invalid_weights",
    [
        [[1.0, 1.0, 1.0]],
        [[1.0], [1.0], [1.0]],
    ],
)
def test_rejects_matrix_shaped_weights_at_all_entry_points(invalid_weights):
    rotations = [eye(3), eye(3), eye(3)]

    with pytest.raises(ValueError, match="one-dimensional"):
        SO3ChordalMeanSmoother(
            window_size=3,
            kernel_weights=invalid_weights,
        )

    with pytest.raises(ValueError, match="one-dimensional"):
        SO3ChordalMeanSmoother.chordal_mean(
            rotations,
            weights=invalid_weights,
        )

    with pytest.raises(ValueError, match="one-dimensional"):
        SO3ChordalMeanSmoother(window_size=3).smooth(
            rotations,
            weights=invalid_weights,
        )
