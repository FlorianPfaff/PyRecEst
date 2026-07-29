from __future__ import annotations

import numpy as np
import pyrecest.calibration as calibration
import pytest


def test_calibration_scalar_helpers_reject_masked_values() -> None:
    masked = np.ma.array(0.5, mask=True)

    with pytest.raises(ValueError, match="offset_s must be a finite scalar"):
        calibration._as_finite_float(masked, "offset_s")
    with pytest.raises(ValueError, match="max_time_delta_s must be nonnegative"):
        calibration._as_nonnegative_time_delta(masked, "max_time_delta_s")
    with pytest.raises(ValueError, match="time_offset_s must be a real scalar"):
        calibration._as_summary_scalar(masked, "time_offset_s")


def test_apply_time_offset_rejects_partially_masked_times() -> None:
    times = np.ma.array([0.0, 1.0], mask=[False, True])

    with pytest.raises(ValueError, match="times_s must contain real numeric values"):
        calibration.apply_time_offset(times, 0.25)


def test_bias_training_rejects_partially_masked_measurements() -> None:
    measurements = np.ma.array(
        [[1.0], [2.0]],
        mask=[[False], [True]],
    )

    with pytest.raises(
        ValueError,
        match="measurement_values must contain numeric values",
    ):
        calibration.make_bias_training_examples(
            measurement_times_s=[0.0, 1.0],
            measurement_values=measurements,
            reference_times_s=[0.0, 1.0],
            reference_values=[[1.0], [2.0]],
        )


def test_bias_scalar_helpers_reject_masked_values() -> None:
    masked = np.ma.array(1, mask=True)

    with pytest.raises(ValueError, match="target_dim must be a nonnegative integer"):
        calibration._bias_module._as_nonnegative_int(masked, "target_dim")
    with pytest.raises(
        ValueError,
        match="ridge_alpha must be a nonnegative finite scalar",
    ):
        calibration._bias_module._as_nonnegative_finite_float(masked, "ridge_alpha")


def test_fully_unmasked_masked_arrays_remain_supported() -> None:
    times = np.ma.array([0.0, 1.0], mask=False)
    offset = np.ma.array(0.25, mask=False)

    np.testing.assert_allclose(
        calibration.apply_time_offset(times, offset),
        [0.25, 1.25],
    )
