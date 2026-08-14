"""Regression tests for extreme finite point-estimate metrics."""

import numpy as np
import numpy.testing as npt
from pyrecest.utils.metrics import mae, rmse


def test_rmse_and_mae_preserve_representable_extreme_values() -> None:
    magnitude = 1.0e308
    estimates = np.full((2, 2), magnitude)
    groundtruths = np.zeros_like(estimates)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        scalar_rmse = rmse(estimates, groundtruths)
        scalar_mae = mae(estimates, groundtruths)
        component_rmse = rmse(estimates, groundtruths, axis=0)
        component_mae = mae(estimates, groundtruths, axis=0)

    npt.assert_allclose(scalar_rmse, magnitude, rtol=0.0, atol=0.0)
    npt.assert_allclose(scalar_mae, magnitude, rtol=0.0, atol=0.0)
    npt.assert_allclose(
        component_rmse,
        np.full(2, magnitude),
        rtol=0.0,
        atol=0.0,
    )
    npt.assert_allclose(
        component_mae,
        np.full(2, magnitude),
        rtol=0.0,
        atol=0.0,
    )
