"""Regression tests for masked SO(3) chordal smoother inputs."""

import numpy as np
import pytest

import pyrecest.backend
from pyrecest.smoothers import SO3ChordalMeanSmoother


def _masked_rotation():
    values = np.eye(3)
    values[0, 0] = 999.0
    mask = np.zeros((3, 3), dtype=bool)
    mask[0, 0] = True
    return np.ma.array(values, mask=mask)


def test_rejects_masked_window_size_at_both_public_paths():
    masked_window_size = np.ma.array(3, mask=True)

    with pytest.raises(ValueError, match="window_size"):
        SO3ChordalMeanSmoother(window_size=masked_window_size)

    smoother = SO3ChordalMeanSmoother(window_size=1)
    with pytest.raises(ValueError, match="window_size"):
        smoother.smooth([np.eye(3)], window_size=masked_window_size)


@pytest.mark.parametrize(
    "call",
    (
        lambda weights: SO3ChordalMeanSmoother(
            window_size=2,
            kernel_weights=weights,
        ),
        lambda weights: SO3ChordalMeanSmoother.chordal_mean(
            [np.eye(3), np.eye(3)],
            weights=weights,
        ),
        lambda weights: SO3ChordalMeanSmoother(window_size=2).smooth(
            [np.eye(3), np.eye(3)],
            weights=weights,
        ),
    ),
)
def test_rejects_masked_weight_payloads(call):
    masked_weights = np.ma.array(
        [1.0, 999.0],
        mask=[False, True],
    )

    with pytest.raises(ValueError, match="masked"):
        call(masked_weights)


@pytest.mark.parametrize(
    "call",
    (
        lambda rotation: SO3ChordalMeanSmoother.chordal_mean(
            [np.eye(3), rotation]
        ),
        lambda rotation: SO3ChordalMeanSmoother(window_size=2).smooth(
            [np.eye(3), rotation]
        ),
        SO3ChordalMeanSmoother.project_to_so3,
        lambda rotation: SO3ChordalMeanSmoother.chordal_distance(
            np.eye(3),
            rotation,
        ),
    ),
)
def test_rejects_masked_rotation_payloads(call):
    with pytest.raises(ValueError, match="masked"):
        call(_masked_rotation())


def test_clear_mask_window_size_remains_supported():
    smoother = SO3ChordalMeanSmoother(
        window_size=np.ma.array(3, mask=False),
    )

    assert smoother.window_size == 3


@pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="NumPy masked-array compatibility is specific to the NumPy backend",
)
def test_clear_mask_weights_and_rotations_remain_supported():
    rotations = np.ma.array(
        np.stack([np.eye(3), np.eye(3)]),
        mask=False,
    )
    weights = np.ma.array([1.0, 1.0], mask=False)

    mean = SO3ChordalMeanSmoother.chordal_mean(rotations, weights=weights)

    np.testing.assert_allclose(mean, np.eye(3))
