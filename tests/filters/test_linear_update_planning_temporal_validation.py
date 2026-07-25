from dataclasses import dataclass

import numpy as np
import pytest
from pyrecest.filters.linear_update_planning import (
    gate_threshold_for_measurement,
    huber_covariance_scale,
    normalized_innovation_squared,
    plan_linear_measurement_update,
    robust_update_covariance_scale,
    source_float_value,
    student_t_covariance_scale,
)


@dataclass(frozen=True)
class _Measurement:
    source: str
    vector: object


_TEMPORAL_SCALARS = (
    np.timedelta64(2, "ns"),
    np.datetime64("1970-01-01T00:00:00.000000002"),
    np.asarray(np.timedelta64(2, "ns")),
    np.asarray(np.datetime64("1970-01-01T00:00:00.000000002")),
    np.asarray(np.timedelta64(2, "ns"), dtype=object),
    np.asarray(np.datetime64("1970-01-01T00:00:00.000000002"), dtype=object),
)


def _base_plan_kwargs():
    return {
        "mean": np.zeros(1),
        "covariance_matrix": np.eye(1),
        "measurement_vector": np.array([1.0]),
        "measurement_covariance": np.eye(1),
        "observation_matrix": np.eye(1),
    }


@pytest.mark.parametrize("temporal", _TEMPORAL_SCALARS)
def test_linear_update_scalar_controls_reject_temporal_payloads(temporal):
    with pytest.raises(ValueError, match="nis"):
        student_t_covariance_scale(temporal, measurement_dim=2)

    with pytest.raises(ValueError, match="measurement_dim"):
        student_t_covariance_scale(1.0, measurement_dim=temporal)

    with pytest.raises(ValueError, match="threshold"):
        huber_covariance_scale(4.0, threshold=temporal)

    with pytest.raises(ValueError, match="gate_threshold"):
        robust_update_covariance_scale(
            "nis-inflate",
            nis=4.0,
            measurement_dim=2,
            gate_threshold=temporal,
        )

    with pytest.raises(ValueError, match="gate_threshold"):
        plan_linear_measurement_update(**_base_plan_kwargs(), gate_threshold=temporal)


@pytest.mark.parametrize("temporal", _TEMPORAL_SCALARS)
def test_source_specific_linear_update_helpers_reject_temporal_scalars(temporal):
    measurement = _Measurement(source="radar", vector=np.array([0.0, 1.0]))

    with pytest.raises(ValueError, match="gate_threshold"):
        gate_threshold_for_measurement(
            measurement,
            gate_thresholds_by_source={"radar": temporal},
        )

    with pytest.raises(ValueError, match="source value"):
        source_float_value(measurement, {"radar": temporal})


@pytest.mark.parametrize(
    "temporal_array",
    (
        np.array([np.timedelta64(1, "ns")]),
        np.array([np.datetime64("1970-01-01T00:00:00.000000001")]),
        np.array([np.timedelta64(1, "ns")], dtype=object),
        np.array([np.datetime64("1970-01-01T00:00:00.000000001")], dtype=object),
    ),
)
def test_linear_update_array_inputs_reject_temporal_values(temporal_array):
    with pytest.raises(ValueError, match="residual"):
        normalized_innovation_squared(temporal_array, np.eye(1))

    kwargs = _base_plan_kwargs()
    kwargs["measurement_vector"] = temporal_array
    with pytest.raises(ValueError, match="measurement_vector"):
        plan_linear_measurement_update(**kwargs)


def test_numeric_numpy_scalars_remain_accepted():
    assert huber_covariance_scale(4.0, threshold=np.asarray(2.0)) == 1.0

    plan = plan_linear_measurement_update(
        **_base_plan_kwargs(),
        gate_threshold=np.asarray(2.0),
        max_residual_norm=np.asarray(10.0),
    )
    assert plan.gate_threshold == 2.0
    assert plan.residual_threshold == 10.0
