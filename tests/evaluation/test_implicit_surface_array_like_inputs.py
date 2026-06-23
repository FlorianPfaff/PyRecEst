"""Regression tests for implicit-surface helpers with plain array-like inputs."""

import numpy as np

from pyrecest.evaluation import surface_band_probability_from_signed_distance


def test_surface_band_probability_accepts_plain_array_like_inputs() -> None:
    probability = surface_band_probability_from_signed_distance(
        [0.0, 0.2],
        [0.05, 0.05],
        0.1,
    )

    assert probability.shape == (2,)
    assert np.all(np.isfinite(probability))
    assert probability[0] > 0.9
    assert 0.0 <= probability[1] < probability[0]
