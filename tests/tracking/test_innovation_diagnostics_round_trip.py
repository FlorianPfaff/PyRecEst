from __future__ import annotations

import json

import numpy as np
from pyrecest.tracking import diagnostic_from_record, innovation_diagnostic


def test_diagnostic_record_round_trip_preserves_canonical_fields() -> None:
    original = innovation_diagnostic(
        np.array([1.0, -2.0]),
        np.array([[2.0, 0.25], [0.25, 1.5]]),
        gate_threshold=10.0,
        action="updated",
        source="radar",
        time=3.5,
        metadata={"sensor_id": 7},
    )
    serialized = original.to_dict(include_arrays=True)
    serialized["extra"] = "kept"

    restored = diagnostic_from_record(serialized)

    assert restored.measurement_dim == original.measurement_dim
    assert restored.nis == original.nis
    assert restored.residual_norm == original.residual_norm
    assert restored.gate_threshold == original.gate_threshold
    assert restored.accepted == original.accepted
    assert restored.action == original.action
    assert restored.source == original.source
    assert restored.time == original.time
    assert np.array_equal(restored.residual, original.residual)
    assert np.array_equal(
        restored.innovation_covariance,
        original.innovation_covariance,
    )
    assert restored.metadata == {"sensor_id": 7, "extra": "kept"}

    # The deserialized object remains serializable instead of leaking NumPy arrays
    # through a nested metadata field.
    json.dumps(restored.to_dict(include_arrays=True))
