import numpy as np
from pyrecest.filters.measurement_scoring import MeasurementScore


def _score(active_measurement_indices):
    return MeasurementScore(
        measurement_jacobian=None,
        predicted_measurements=None,
        innovation_covariance=None,
        residual=None,
        active_measurement_indices=active_measurement_indices,
    )


def test_measurement_score_is_active_handles_numpy_index_arrays():
    assert _score(np.array([0, 2])).is_active
    assert not _score(np.array([], dtype=int)).is_active
