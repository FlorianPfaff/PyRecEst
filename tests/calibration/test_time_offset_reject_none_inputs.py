import numpy as np
import pytest

from pyrecest.calibration.time_offset import (
    apply_time_offset,
    interpolate_reference_values,
    nearest_time_indices,
    time_offset_error_summary,
)


def _assert_rejects_none_payload(func, expected_name, *args):
    with pytest.raises(
        ValueError,
        match=f"{expected_name} must contain real numeric values",
    ):
        func(*args)


def test_apply_time_offset_rejects_none_time_payloads():
    _assert_rejects_none_payload(
        apply_time_offset,
        "times_s",
        np.array([0.0, None], dtype=object),
        0.0,
    )


def test_nearest_time_indices_rejects_none_query_payloads():
    _assert_rejects_none_payload(
        nearest_time_indices,
        "query_times_s",
        np.array([0.0, 1.0]),
        np.array([None], dtype=object),
    )


def test_interpolation_rejects_none_reference_values():
    _assert_rejects_none_payload(
        interpolate_reference_values,
        "reference_values",
        np.array([0.0, 1.0]),
        np.array([[0.0], [None]], dtype=object),
        np.array([0.25]),
    )


def test_time_offset_summary_rejects_none_measurement_values():
    _assert_rejects_none_payload(
        time_offset_error_summary,
        "measurement_values",
        np.array([0.0]),
        np.array([None], dtype=object),
        np.array([0.0, 1.0]),
        np.array([[0.0], [1.0]]),
        0.0,
    )
