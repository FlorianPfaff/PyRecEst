from __future__ import annotations

import numpy as np
import pyrecest.calibration as calibration
import pytest

_OBJECT_TEMPORAL_SCALARS = (
    np.array(np.timedelta64(1, "ns"), dtype=object),
    np.array(np.datetime64("1970-01-01T00:00:00.000000001"), dtype=object),
)

_OBJECT_TEMPORAL_ARRAYS = (
    *_OBJECT_TEMPORAL_SCALARS,
    np.array([np.timedelta64(1, "ns")], dtype=object),
    np.array([np.datetime64("1970-01-01T00:00:00.000000001")], dtype=object),
)


@pytest.mark.parametrize("value", _OBJECT_TEMPORAL_SCALARS)
def test_calibration_scalar_helpers_reject_object_temporal_values(value) -> None:
    with pytest.raises(ValueError, match="offset_s must be a finite scalar"):
        calibration._as_finite_float(value, "offset_s")
    with pytest.raises(ValueError, match="max_time_delta_s must be nonnegative"):
        calibration._as_nonnegative_time_delta(value, "max_time_delta_s")
    with pytest.raises(ValueError, match="time_offset_s must be a real scalar"):
        calibration._as_summary_scalar(value, "time_offset_s")


@pytest.mark.parametrize("value", _OBJECT_TEMPORAL_ARRAYS)
def test_calibration_numeric_array_helper_rejects_object_temporal_values(value) -> None:
    with pytest.raises(
        ValueError,
        match="measurement_times_s must contain real numeric values",
    ):
        calibration._as_real_numeric_array(value, "measurement_times_s")
