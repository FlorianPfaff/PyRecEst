"""Regression tests for extreme finite rejection-sampling bounds."""

import numpy as np
import numpy.testing as npt

from pyrecest.sampling import FibonacciGridSampler, FibonacciRejectionSampler


def test_extreme_finite_bounding_box_maps_without_overflow() -> None:
    """Finite endpoints must not produce non-finite mapped candidates."""
    max_float = np.finfo(np.float64).max
    bounding_box = np.array([[-max_float, max_float]])
    n_candidates = 9

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        samples, info = FibonacciRejectionSampler().sample_rejection(
            lambda values: np.ones(values.shape[0]),
            n_candidates=n_candidates,
            dim=1,
            max_density=1.0,
            bounding_box=bounding_box,
        )

    unit_samples = FibonacciGridSampler().get_uniform_samples(n_candidates, 2)[:, :1]
    expected = (1.0 - unit_samples) * (-max_float) + unit_samples * max_float

    assert np.all(np.isfinite(samples))
    npt.assert_array_equal(samples, expected)
    assert info["n_accepted"] == n_candidates
    assert np.all(samples >= -max_float)
    assert np.all(samples <= max_float)
