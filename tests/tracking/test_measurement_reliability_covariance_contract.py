from __future__ import annotations

import numpy as np
import pytest
from pyrecest.tracking.measurement_reliability import (
    ReliabilityWeightedMeasurement,
    apply_measurement_reliability,
    scale_covariance_by_reliability,
)


@pytest.mark.parametrize(
    ("covariance", "message"),
    [
        (np.array([[1.0, 0.5], [0.0, 1.0]]), "symmetric"),
        (np.diag([1.0, -0.25]), "positive semidefinite"),
    ],
)
def test_reliability_helpers_reject_invalid_covariance_geometry(
    covariance: np.ndarray, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        scale_covariance_by_reliability(covariance, 0.5)

    with pytest.raises(ValueError, match=message):
        apply_measurement_reliability(covariance, reliability=0.5)

    with pytest.raises(ValueError, match=message):
        ReliabilityWeightedMeasurement(
            measurement=np.array([1.0, 2.0]),
            covariance=covariance,
            reliability=0.5,
        )


def test_reliability_helpers_accept_singular_psd_covariance() -> None:
    covariance = np.array([[1.0, 1.0], [1.0, 1.0]])

    scaled, scale = scale_covariance_by_reliability(covariance, 0.5)

    assert scale == 2.0
    assert np.allclose(scaled, 2.0 * covariance)
