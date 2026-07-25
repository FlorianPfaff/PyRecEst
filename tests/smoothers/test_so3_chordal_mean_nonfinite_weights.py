"""Regression coverage for non-finite SO(3) chordal smoother weights."""

import pytest
from pyrecest.backend import eye
from pyrecest.smoothers import SO3ChordalMeanSmoother


@pytest.mark.parametrize(
    "invalid_weight",
    [float("nan"), float("inf"), -float("inf")],
)
def test_rejects_nonfinite_weights_at_all_entry_points(invalid_weight):
    rotations = [eye(3), eye(3), eye(3)]

    with pytest.raises(ValueError, match="finite"):
        SO3ChordalMeanSmoother(
            window_size=3,
            kernel_weights=[1.0, invalid_weight, 1.0],
        )

    with pytest.raises(ValueError, match="finite"):
        SO3ChordalMeanSmoother.chordal_mean(
            rotations[:2],
            weights=[1.0, invalid_weight],
        )

    with pytest.raises(ValueError, match="finite"):
        SO3ChordalMeanSmoother(window_size=3).smooth(
            rotations,
            weights=[1.0, invalid_weight, 1.0],
        )
