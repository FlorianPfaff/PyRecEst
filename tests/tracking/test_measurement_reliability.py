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


def test_max_scale_caps_at_or_above_nominal_scale() -> None:
    assert reliability_to_covariance_scale(0.25, max_scale=2.0) == 2.0

    with pytest.raises(ValueError, match="max_scale must be at least 1"):
        reliability_to_covariance_scale(0.25, max_scale=0.5)
    with pytest.raises(ValueError, match="max_scale must be at least 1"):
        MeasurementReliabilityConfig(max_scale=0.5)


def test_scalar_reliability_inputs_reject_text_and_object_booleans() -> None:
    invalid_values = (
        "0.5",
        b"0.5",
        np.array("0.5"),
        np.array("0.5", dtype=object),
        np.array(True, dtype=object),
    )

    for value in invalid_values:
        with pytest.raises(ValueError, match="finite scalar"):
            reliability_to_covariance_scale(value)


def test_reliability_config_rejects_text_and_object_boolean_scalars() -> None:
    invalid_configs = (
        {"threshold": "0.5"},
        {"floor": np.array("0.1")},
        {"exponent": "2.0"},
        {"max_scale": np.array(True, dtype=object)},
    )

    for kwargs in invalid_configs:
        with pytest.raises(ValueError):
            MeasurementReliabilityConfig(**kwargs)


def test_reliability_weighted_measurement_rejects_malformed_measurements() -> None:
    invalid_measurements = (
        np.array([True]),
        np.array(["1.0"]),
        np.array([True], dtype=object),
        np.array([1.0 + 0.0j]),
    )

    for measurement in invalid_measurements:
        with pytest.raises(
            ValueError, match="measurement must contain real numeric values"
        ):
            ReliabilityWeightedMeasurement(
                measurement=measurement,
                covariance=np.eye(1),
                reliability=0.5,
            )


def test_reliability_helpers_reject_malformed_covariance_arrays() -> None:
    invalid_covariances = (
        np.array([[True]]),
        np.array([["1.0"]]),
        np.array([[True]], dtype=object),
        np.array([[1.0 + 0.0j]]),
    )

    for covariance in invalid_covariances:
        with pytest.raises(
            ValueError, match="covariance must contain real numeric values"
        ):
            scale_covariance_by_reliability(covariance, 0.5)
        with pytest.raises(
            ValueError, match="covariance must contain real numeric values"
        ):
            apply_measurement_reliability(covariance, reliability=0.5)


def test_scale_covariance_by_reliability_returns_scaled_copy() -> None:
    cov = np.diag([4.0, 9.0])
    scaled, scale = scale_covariance_by_reliability(cov, 0.25)

    assert scale == 4.0
    assert np.allclose(scaled, np.diag([16.0, 36.0]))
    assert np.allclose(cov, np.diag([4.0, 9.0]))


def test_covariance_validation_rejects_bool_and_text_values() -> None:
    invalid_covariances = (
        [[True]],
        [["1.0"]],
        [[b"1.0"]],
        np.array([[False]], dtype=object),
        np.array([["1.0"]], dtype=object),
    )

    for covariance in invalid_covariances:
        with pytest.raises(ValueError, match="real numeric values"):
            scale_covariance_by_reliability(covariance, 0.5)
        with pytest.raises(ValueError, match="real numeric values"):
            ReliabilityWeightedMeasurement(
                measurement=np.array([1.0]),
                covariance=covariance,
                reliability=0.5,
            )


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
