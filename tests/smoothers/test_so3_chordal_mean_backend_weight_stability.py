"""Cross-backend regressions for extreme SO(3) chordal smoother weights."""

import numpy as np
import numpy.testing as npt
import pytest
from pyrecest.backend import array, cos, eye, sin, to_numpy
from pyrecest.smoothers import SO3ChordalMeanSmoother


def _active_dtype():
    return to_numpy(array([0.0], dtype=float)).dtype


def _z_rotation(angle):
    return array(
        [
            [cos(angle), -sin(angle), 0.0],
            [sin(angle), cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
    )


def test_chordal_mean_preserves_largest_finite_weight_ratio_across_backends():
    dtype = _active_dtype()
    largest = np.finfo(dtype).max
    weights = array(np.asarray([largest, largest / 2.0], dtype=dtype), dtype=float)

    mean_rotation = SO3ChordalMeanSmoother.chordal_mean(
        [eye(3), _z_rotation(0.5 * np.pi)],
        weights=weights,
    )

    npt.assert_allclose(
        to_numpy(mean_rotation),
        to_numpy(_z_rotation(np.arctan(0.5))),
        atol=1.0e-6,
    )


def test_weight_scaling_preserves_positive_subnormal_ratio_across_backends():
    dtype = _active_dtype()
    smallest = np.finfo(dtype).smallest_subnormal
    weights = array(
        np.asarray([2.0 * smallest, smallest], dtype=dtype),
        dtype=float,
    )

    scaled_weights = SO3ChordalMeanSmoother._normalize_weight_vector(
        weights,
        2,
        "weights",
        normalize=False,
    )
    normalized_weights = SO3ChordalMeanSmoother._normalize_weight_vector(
        weights,
        2,
        "weights",
        normalize=True,
    )

    npt.assert_allclose(to_numpy(scaled_weights), [1.0, 0.5])
    npt.assert_allclose(to_numpy(normalized_weights), [2.0 / 3.0, 1.0 / 3.0])


def test_rejects_negative_subnormal_weight_across_backends():
    dtype = _active_dtype()
    smallest = np.finfo(dtype).smallest_subnormal
    weights = array(
        np.asarray([1.0, -smallest], dtype=dtype),
        dtype=float,
    )

    with pytest.raises(ValueError, match="nonnegative"):
        SO3ChordalMeanSmoother._normalize_weight_vector(
            weights,
            2,
            "weights",
        )
