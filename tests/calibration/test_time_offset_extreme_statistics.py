"""Regression tests for extreme finite time-offset residual statistics."""

from __future__ import annotations

import numpy as np
import numpy.testing as npt
from pyrecest.calibration import (
    aggregate_time_offset_sweeps,
    time_offset_error_summary,
)


def test_time_offset_summary_preserves_extreme_finite_residual_norm() -> None:
    magnitude = 1.0e308
    expected_error = np.hypot(magnitude, magnitude)

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        summary = time_offset_error_summary(
            np.array([0.0]),
            np.array([[magnitude, magnitude]]),
            np.array([0.0, 1.0]),
            np.zeros((2, 2)),
            0.0,
        )

    assert summary["count"] == 1.0
    assert summary["coverage"] == 1.0
    assert summary["std"] == 0.0
    for key in ("mean", "rmse", "p95", "max"):
        npt.assert_allclose(summary[key], expected_error, rtol=1.0e-15)


def test_time_offset_sweep_aggregation_scales_extreme_finite_metrics() -> None:
    magnitude = 1.0e308
    row = {
        "time_offset_s": 0.0,
        "count": 1.0,
        "mean": magnitude,
        "std": 0.0,
        "rmse": magnitude,
        "p95": magnitude,
        "max": magnitude,
    }

    with np.errstate(over="raise", invalid="raise", divide="raise"):
        aggregated = aggregate_time_offset_sweeps([[row], [row]])

    assert len(aggregated) == 1
    result = aggregated[0]
    assert result["count"] == 2.0
    assert result["std"] == 0.0
    for key in ("mean", "rmse", "max"):
        npt.assert_allclose(result[key], magnitude, rtol=1.0e-15)
