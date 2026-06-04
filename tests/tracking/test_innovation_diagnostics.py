from __future__ import annotations

import numpy as np
from pyrecest.tracking import (
    diagnostic_from_record,
    diagnostics_from_records,
    innovation_diagnostic,
    innovation_gate_threshold,
    linear_innovation_diagnostic,
    normalized_innovation_squared,
    summarize_innovation_diagnostics,
)


def test_innovation_diagnostic_computes_nis_and_gate_decision() -> None:
    diagnostic = innovation_diagnostic(
        np.array([2.0, 0.0]),
        np.eye(2),
        gate_probability=0.95,
        source="radar",
        time=12.0,
    )

    assert diagnostic.measurement_dim == 2
    assert diagnostic.nis == 4.0
    assert diagnostic.residual_norm == 2.0
    assert diagnostic.gate_threshold == innovation_gate_threshold(0.95, 2)
    assert diagnostic.accepted is True
    assert diagnostic.source == "radar"
    assert diagnostic.time == 12.0


def test_innovation_diagnostic_rejects_when_over_gate() -> None:
    diagnostic = innovation_diagnostic(
        np.array([10.0]),
        np.eye(1),
        gate_threshold=9.0,
    )

    assert diagnostic.nis == 100.0
    assert diagnostic.accepted is False


def test_normalized_innovation_squared_reexport() -> None:
    assert normalized_innovation_squared([2.0, 1.0], np.diag([4.0, 1.0])) == 2.0


def test_linear_innovation_diagnostic_uses_measurement_model() -> None:
    diagnostic = linear_innovation_diagnostic(
        mean=np.array([1.0, 2.0, 3.0]),
        covariance=np.eye(3),
        measurement=np.array([2.0, 4.0]),
        measurement_matrix=np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        measurement_covariance=np.eye(2),
        gate_threshold=10.0,
        source="rf",
    )

    assert np.allclose(diagnostic.residual, [1.0, 2.0])
    assert np.isclose(diagnostic.nis, 2.5)
    assert diagnostic.accepted is True
    assert diagnostic.source == "rf"


def test_diagnostic_from_record_preserves_fields() -> None:
    diagnostic = diagnostic_from_record(
        {
            "time_s": 3.0,
            "source": "rf",
            "update_action": "rejected",
            "accepted": False,
            "measurement_dim": 2,
            "nis": 12.5,
            "residual_norm_m": 50.0,
            "extra": "kept",
        }
    )

    assert diagnostic.time == 3.0
    assert diagnostic.source == "rf"
    assert diagnostic.action == "rejected"
    assert diagnostic.accepted is False
    assert diagnostic.nis == 12.5
    assert diagnostic.residual_norm == 50.0
    assert diagnostic.metadata["extra"] == "kept"


def test_summarize_innovation_diagnostics_by_source() -> None:
    diagnostics = diagnostics_from_records(
        [
            {"source": "rf", "accepted": True, "nis": 1.0, "residual_norm_m": 2.0},
            {"source": "rf", "accepted": False, "nis": 9.0, "residual_norm_m": 6.0},
            {"source": "radar", "accepted": True, "nis": 4.0, "residual_norm_m": 3.0},
        ]
    )
    summaries = {
        item.group: item for item in summarize_innovation_diagnostics(diagnostics)
    }

    assert summaries["rf"].count == 2
    assert summaries["rf"].accepted_count == 1
    assert summaries["rf"].rejected_count == 1
    assert summaries["rf"].acceptance_rate == 0.5
    assert summaries["rf"].nis_mean == 5.0
    assert summaries["radar"].count == 1
