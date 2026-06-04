from __future__ import annotations

import numpy as np
import pytest
from pyrecest.tracking import (
    MeasurementReliabilityConfig,
    ReliabilityWeightedMeasurement,
    apply_measurement_reliability,
    reliability_to_covariance_scale,
    scale_covariance_by_reliability,
)


def test_reliability_to_covariance_scale_uses_inverse_probability() -> None:
    assert reliability_to_covariance_scale(1.0) == 1.0
    assert reliability_to_covariance_scale(0.5) == 2.0
    assert reliability_to_covariance_scale(0.01, floor=0.1) == 10.0


def test_scale_covariance_by_reliability_returns_scaled_copy() -> None:
    cov = np.diag([4.0, 9.0])
    scaled, scale = scale_covariance_by_reliability(cov, 0.25)

    assert scale == 4.0
    assert np.allclose(scaled, np.diag([16.0, 36.0]))
    assert np.allclose(cov, np.diag([4.0, 9.0]))


def test_hard_mode_rejects_low_reliability_measurement() -> None:
    result = apply_measurement_reliability(
        np.eye(2),
        reliability=0.4,
        mode="hard",
        threshold=0.5,
    )

    assert not result.accepted
    assert result.action == "reliability_rejected"
    assert result.covariance_scale == 1.0
    assert np.allclose(result.covariance, np.eye(2))


def test_inflate_mode_can_also_apply_threshold() -> None:
    rejected = apply_measurement_reliability(
        np.eye(2),
        reliability=0.2,
        mode="inflate",
        threshold=0.25,
    )
    accepted = apply_measurement_reliability(
        np.eye(2),
        reliability=0.5,
        mode="inflate",
        threshold=0.25,
    )

    assert not rejected.accepted
    assert rejected.action == "reliability_rejected"
    assert accepted.accepted
    assert accepted.action == "reliability_inflated"
    assert accepted.covariance_scale == 2.0


def test_reliability_weighted_measurement_validates_covariance_dimension() -> None:
    measurement = ReliabilityWeightedMeasurement(
        measurement=np.array([1.0, 2.0]),
        covariance=np.eye(2),
        reliability=0.5,
        source="rf",
        metadata={"row": 3},
    )
    result = measurement.apply_reliability(MeasurementReliabilityConfig(mode="inflate"))

    assert measurement.source == "rf"
    assert measurement.metadata == {"row": 3}
    assert result.covariance_scale == 2.0
    assert np.allclose(result.covariance, 2.0 * np.eye(2))

    with pytest.raises(ValueError, match="covariance must have shape"):
        ReliabilityWeightedMeasurement(
            measurement=np.array([1.0, 2.0]),
            covariance=np.eye(3),
            reliability=0.5,
        )


def test_invalid_reliability_raises() -> None:
    with pytest.raises(ValueError, match="reliability"):
        reliability_to_covariance_scale(1.5)
