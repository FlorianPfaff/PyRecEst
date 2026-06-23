from __future__ import annotations

import numpy as np
from pyrecest.tracking import (
    diagnostic_from_record,
    diagnostics_from_records,
    summarize_innovation_diagnostics,
)


def test_diagnostic_from_record_parses_serialized_accepted_values() -> None:
    false_record = diagnostic_from_record({"accepted": "False"})
    true_record = diagnostic_from_record({"accepted": "true"})
    zero_record = diagnostic_from_record({"accepted": "0"})
    missing_record = diagnostic_from_record({"accepted": np.nan})

    assert false_record.accepted is False
    assert true_record.accepted is True
    assert zero_record.accepted is False
    assert missing_record.accepted is None

    diagnostics = diagnostics_from_records(
        [
            {"source": "rf", "accepted": "False", "nis": 9.0},
            {"source": "rf", "accepted": "true", "nis": 1.0},
        ]
    )
    summary = summarize_innovation_diagnostics(diagnostics)[0]

    assert summary.accepted_count == 1
    assert summary.rejected_count == 1
    assert summary.acceptance_rate == 0.5
