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
