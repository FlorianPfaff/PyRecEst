import numpy as np
import pytest
from pyrecest.filters.update_diagnostics import MeasurementUpdateDiagnostics


def test_active_measurement_indices_reject_non_integer_values():
    invalid_indices = (
        True,
        False,
        "1",
        b"1",
        1.5,
        -1,
        np.bool_(True),
        np.array([1]),
    )

    for invalid_index in invalid_indices:
        with pytest.raises(ValueError, match="active_measurement_indices"):
            MeasurementUpdateDiagnostics(active_measurement_indices=(invalid_index,))


def test_active_measurement_indices_reject_non_sequence_values():
    with pytest.raises(ValueError, match="active_measurement_indices"):
        MeasurementUpdateDiagnostics(active_measurement_indices=1)  # type: ignore[arg-type]


def test_measurement_count_rejects_non_integer_values():
    invalid_counts = (
        True,
        False,
        "3",
        b"3",
        2.0,
        -1,
        np.bool_(False),
        np.array([1]),
    )

    for invalid_count in invalid_counts:
        with pytest.raises(ValueError, match="measurement_count"):
            MeasurementUpdateDiagnostics(measurement_count=invalid_count)


def test_diagnostics_accept_integer_like_numpy_scalars():
    diagnostics = MeasurementUpdateDiagnostics(
        active_measurement_indices=(0, np.int64(2), np.array(3)),
        measurement_count=np.int64(4),
    )

    assert diagnostics.active_measurement_indices == (0, 2, 3)
    assert diagnostics.measurement_count == 4
