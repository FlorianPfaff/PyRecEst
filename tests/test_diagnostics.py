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
