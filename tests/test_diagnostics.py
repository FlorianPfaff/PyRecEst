from math import isclose, log

import numpy as np
import pytest
from pyrecest.diagnostics import (
    AssociationDiagnostics,
    FilterDiagnostics,
    ParticleDiagnostics,
)


def test_diagnostics_are_dict_serializable_containers():
    filter_diag = FilterDiagnostics(nis=1.5, covariance_trace=0.2)
    particle_diag = ParticleDiagnostics(effective_sample_size=42.0, resampled=True)
    association_diag = AssociationDiagnostics(selected_assignments=[(0, 1)])

    assert filter_diag.to_dict()["nis"] == 1.5
    assert particle_diag.to_dict()["resampled"] is True
    assert association_diag.to_dict()["selected_assignments"] == [(0, 1)]


def test_diagnostics_mapping_exposes_metadata_entries():
    diagnostics = FilterDiagnostics.from_mapping({"nis": 1.5, "custom_score": 7.0})

    assert diagnostics["custom_score"] == 7.0
    assert diagnostics.get("custom_score") == 7.0
    assert "custom_score" in diagnostics
    assert dict(diagnostics.items())["custom_score"] == 7.0
    assert diagnostics.to_dict()["metadata"] == {"custom_score": 7.0}


def test_diagnostics_mapping_stores_method_name_keys_in_metadata():
    diagnostics = FilterDiagnostics()

    diagnostics["items"] = "custom-value"

    assert diagnostics["items"] == "custom-value"
    assert diagnostics.metadata == {"items": "custom-value"}
    assert callable(diagnostics.items)


def test_particle_diagnostics_clips_negative_weights_before_normalizing():
    diagnostics = ParticleDiagnostics.from_weights([2.0, -1.0, 2.0])

    assert isclose(diagnostics.effective_sample_size, 2.0)
    assert isclose(diagnostics.weight_entropy, log(2.0))


def test_particle_diagnostics_stabilizes_extreme_finite_weights():
    maximum = np.finfo(float).max

    diagnostics = ParticleDiagnostics.from_weights([maximum, maximum / 2.0])

    expected_entropy = -(2.0 / 3.0 * log(2.0 / 3.0) + 1.0 / 3.0 * log(1.0 / 3.0))
    assert isclose(diagnostics.effective_sample_size, 1.8)
    assert isclose(diagnostics.weight_entropy, expected_entropy)


def test_particle_diagnostics_rejects_nonfinite_weights():
    for weights in ([float("nan"), 1.0], [float("inf"), 1.0]):
        with pytest.raises(ValueError, match="Particle weights must be finite"):
            ParticleDiagnostics.from_weights(weights)


def test_particle_diagnostics_rejects_text_weight_sequences():
    text_bytes = bytes([49, 50])
    mutable_text_bytes = bytearray([49, 50])
    for weights in ("12", text_bytes, mutable_text_bytes):
        with pytest.raises(ValueError, match="Particle weights must be numeric"):
            ParticleDiagnostics.from_weights(weights)


def test_particle_diagnostics_rejects_boolean_weights():
    invalid_cases = (True, [True, False], np.array([True, False]))
    for weights in invalid_cases:
        with pytest.raises(ValueError, match="Particle weights must be numeric"):
            ParticleDiagnostics.from_weights(weights)


def test_particle_diagnostics_rejects_complex_weights():
    invalid_cases = (
        1.0 + 2.0j,
        np.complex64(1.0 + 2.0j),
        np.complex128(1.0 + 0.0j),
        [np.complex64(1.0 + 2.0j), 1.0],
        np.array([1.0 + 2.0j, 1.0], dtype=np.complex64),
        np.array([np.complex64(1.0 + 2.0j), 1.0], dtype=object),
    )
    for weights in invalid_cases:
        with pytest.raises(ValueError, match="Particle weights must be numeric"):
            ParticleDiagnostics.from_weights(weights)


def test_particle_diagnostics_accepts_real_numpy_scalar_weights():
    diagnostics = ParticleDiagnostics.from_weights(
        [np.float32(1.0), np.float64(3.0)]
    )

    assert isclose(diagnostics.effective_sample_size, 1.6)
