import numpy as np
import pytest

from pyrecest.tracking import InnovationDiagnostic


@pytest.mark.parametrize("nis", [np.nan, np.inf, -np.inf])
def test_nonfinite_nis_has_no_mahalanobis_distance(nis):
    diagnostic = InnovationDiagnostic(measurement_dim=1, nis=nis)

    assert diagnostic.mahalanobis_distance is None
    assert diagnostic.to_dict()["mahalanobis_distance"] is None


def test_finite_nis_keeps_clamped_square_root_behavior():
    assert InnovationDiagnostic(
        measurement_dim=1, nis=4.0
    ).mahalanobis_distance == pytest.approx(2.0)
    assert InnovationDiagnostic(
        measurement_dim=1, nis=-1e-12
    ).mahalanobis_distance == pytest.approx(0.0)
