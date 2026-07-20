from __future__ import annotations

import numpy as np
import pytest
from pyrecest.tracking import diagnostic_from_record, innovation_diagnostic

_TEMPORAL_SCALARS = (
    np.timedelta64(2, "ns"),
    np.datetime64("1970-01-01T00:00:00.000000002"),
    np.asarray(np.timedelta64(2, "ns")),
    np.array(np.datetime64("1970-01-01T00:00:00.000000002"), dtype=object),
)


@pytest.mark.parametrize("threshold", _TEMPORAL_SCALARS)
def test_innovation_diagnostic_rejects_temporal_gate_threshold(threshold) -> None:
    with pytest.raises(ValueError, match="gate_threshold"):
        innovation_diagnostic(np.array([1.0]), np.eye(1), gate_threshold=threshold)


@pytest.mark.parametrize("measurement_dim", _TEMPORAL_SCALARS)
def test_diagnostic_from_record_rejects_temporal_measurement_dim(
    measurement_dim,
) -> None:
    with pytest.raises(ValueError, match="measurement_dim"):
        diagnostic_from_record({"measurement_dim": measurement_dim})


@pytest.mark.parametrize(
    "accepted",
    (
        np.timedelta64(0, "ns"),
        np.timedelta64(1, "ns"),
        np.datetime64("1970-01-01T00:00:00.000000001"),
        np.array(np.timedelta64(1, "ns"), dtype=object),
    ),
)
def test_diagnostic_from_record_rejects_temporal_accepted_flags(accepted) -> None:
    with pytest.raises(ValueError, match="accepted"):
        diagnostic_from_record({"accepted": accepted})


def test_diagnostic_from_record_treats_temporal_float_fields_as_missing() -> None:
    diagnostic = diagnostic_from_record(
        {
            "nis": np.timedelta64(2, "ns"),
            "residual_norm_m": np.datetime64("1970-01-01T00:00:00.000000003"),
            "gate_threshold": np.array(np.timedelta64(4, "ns"), dtype=object),
            "time_s": np.asarray(np.datetime64("1970-01-01T00:00:00.000000005")),
        }
    )

    assert diagnostic.nis is None
    assert diagnostic.residual_norm is None
    assert diagnostic.gate_threshold is None
    assert diagnostic.time is None
