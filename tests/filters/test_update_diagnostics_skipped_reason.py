import pytest
from pyrecest.filters.update_diagnostics import MeasurementUpdateDiagnostics


@pytest.mark.parametrize("skipped_reason", ["", "   ", 0, object()])
def test_skipped_reason_rejects_non_text_or_empty_values(skipped_reason):
    with pytest.raises(ValueError, match="skipped_reason"):
        MeasurementUpdateDiagnostics(skipped_reason=skipped_reason)


def test_skipped_reason_rejects_active_measurements():
    with pytest.raises(
        ValueError,
        match="skipped diagnostics must not contain active_measurement_indices",
    ):
        MeasurementUpdateDiagnostics(
            active_measurement_indices=[0],
            measurement_count=1,
            skipped_reason="gated",
        )


def test_skipped_constructor_accepts_non_empty_reason():
    diagnostics = MeasurementUpdateDiagnostics.skipped("gated")

    assert diagnostics.skipped_reason == "gated"
    assert not diagnostics.updated
