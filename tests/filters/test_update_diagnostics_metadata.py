import pytest
from pyrecest.filters.update_diagnostics import MeasurementUpdateDiagnostics


def test_metadata_rejects_non_mapping_pair_sequences():
    with pytest.raises(ValueError, match="metadata"):
        MeasurementUpdateDiagnostics(metadata=(("a", 1),))  # type: ignore[arg-type]
