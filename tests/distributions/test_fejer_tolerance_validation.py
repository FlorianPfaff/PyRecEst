import numpy as np
import pytest

from pyrecest.distributions.hypertorus.fejer import (
    adaptive_kernel_reduce_coefficients,
    minimum_on_fft_grid,
)


@pytest.mark.parametrize(
    "invalid_tolerance",
    [
        np.nan,
        np.inf,
        -np.inf,
        -1.0e-12,
        True,
        "1e-12",
        1.0e-12 + 0.0j,
        np.array([1.0e-12]),
        np.datetime64("2026-01-01"),
    ],
)
def test_adaptive_reduction_rejects_invalid_min_value_tolerance(
    invalid_tolerance,
):
    with pytest.raises(
        ValueError,
        match="min_value_tolerance must be a finite nonnegative real scalar",
    ):
        adaptive_kernel_reduce_coefficients(
            np.ones(3),
            min_value_tolerance=invalid_tolerance,
        )


def test_adaptive_reduction_accepts_numpy_scalar_tolerance():
    coefficients = np.array(
        [0.0, -0.1, 1.0 / (2.0 * np.pi), -0.1, 0.0]
    )

    reduced, exponent = adaptive_kernel_reduce_coefficients(
        coefficients,
        (3,),
        min_value_tolerance=np.float64(0.0),
        return_exponent=True,
    )

    assert exponent > 0.0
    assert minimum_on_fft_grid(reduced) >= -1.0e-12
