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


def test_active_measurement_indices_reject_masked_integer_values():
    masked_indices = (
        np.ma.array(0, mask=True),
        np.ma.array(7, mask=True),
        np.ma.masked,
    )

    for masked_index in masked_indices:
        with pytest.raises(ValueError, match="active_measurement_indices"):
            MeasurementUpdateDiagnostics(active_measurement_indices=(masked_index,))


def test_active_measurement_indices_reject_non_sequence_values():
    with pytest.raises(ValueError, match="active_measurement_indices"):
        MeasurementUpdateDiagnostics(active_measurement_indices=1)  # type: ignore[arg-type]


def test_active_measurement_indices_reject_text_like_top_level_sequences():
    invalid_sequences = (
        "01",
        b"\x00\x01",
        bytearray([0, 1]),
    )

    for invalid_sequence in invalid_sequences:
        with pytest.raises(ValueError, match="active_measurement_indices"):
            MeasurementUpdateDiagnostics(
                active_measurement_indices=invalid_sequence,  # type: ignore[arg-type]
                measurement_count=2,
            )


def test_active_measurement_indices_must_fit_measurement_count():
    with pytest.raises(
        ValueError,
        match="active_measurement_indices.*measurement_count",
    ):
        MeasurementUpdateDiagnostics(
            active_measurement_indices=(0, 2), measurement_count=2
        )


def test_active_measurement_indices_reject_duplicates():
    with pytest.raises(ValueError, match="active_measurement_indices.*duplicate"):
        MeasurementUpdateDiagnostics(
            active_measurement_indices=(0, np.int64(0)),
            measurement_count=1,
        )


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


def test_measurement_count_rejects_masked_integer_values():
    for masked_count in (np.ma.array(0, mask=True), np.ma.array(3, mask=True)):
        with pytest.raises(ValueError, match="measurement_count"):
            MeasurementUpdateDiagnostics(measurement_count=masked_count)


def test_diagnostics_accept_integer_like_numpy_scalars():
    diagnostics = MeasurementUpdateDiagnostics(
        active_measurement_indices=(
            0,
            np.ma.array(1, mask=False),
            np.int64(2),
            np.array(3),
        ),
        measurement_count=np.ma.array(4, mask=False),
    )

    assert diagnostics.active_measurement_indices == (0, 1, 2, 3)
    assert diagnostics.measurement_count == 4
