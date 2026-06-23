from __future__ import annotations

import pytest
from pyrecest.tracking import diagnostic_from_record


def test_diagnostic_from_record_accepts_integral_measurement_dim_strings() -> None:
    assert diagnostic_from_record({"measurement_dim": "2"}).measurement_dim == 2
    assert diagnostic_from_record({"measurement_dim": "2.0"}).measurement_dim == 2


def test_diagnostic_from_record_infers_measurement_dim_from_residual() -> None:
    diagnostic = diagnostic_from_record({"residual": [1.0, -2.0, 3.0]})

    assert diagnostic.measurement_dim == 3


@pytest.mark.parametrize(
    "measurement_dim",
    [True, -1, 2.5, "2.5", float("nan")],
)
def test_diagnostic_from_record_rejects_invalid_measurement_dim(
    measurement_dim,
) -> None:
    with pytest.raises(ValueError, match="measurement_dim"):
        diagnostic_from_record({"measurement_dim": measurement_dim})
