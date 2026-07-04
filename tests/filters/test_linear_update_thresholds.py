from dataclasses import dataclass

import numpy as np
from pyrecest.filters.linear_update_planning import (
    gate_threshold_for_measurement,
    plan_linear_measurement_update,
)


@dataclass
class _Measurement:
    source: str
    vector: np.ndarray


def _base_plan_kwargs(measurement_value=1.0):
    return dict(
        mean=np.array([0.0]),
        covariance_matrix=np.array([[1.0]]),
        measurement_vector=np.array([measurement_value]),
        measurement_covariance=np.array([[1.0]]),
        observation_matrix=np.array([[1.0]]),
    )


def test_zero_gate_threshold_rejects_nonzero_nis():
    kwargs = _base_plan_kwargs(measurement_value=1.0)
    kwargs["gate_threshold"] = 0.0

    plan = plan_linear_measurement_update(**kwargs)

    assert plan.gate_threshold == 0.0
    assert plan.accepted is False
    assert plan.action == "rejected"


def test_zero_safety_threshold_rejects_nonzero_nis():
    kwargs = _base_plan_kwargs(measurement_value=1.0)
    kwargs["safety_gate_threshold"] = 0.0

    plan = plan_linear_measurement_update(**kwargs)

    assert plan.safety_gate_threshold == 0.0
    assert plan.accepted is False
    assert plan.action == "safety_rejected"


def test_zero_residual_threshold_rejects_nonzero_residual():
    kwargs = _base_plan_kwargs(measurement_value=1.0)
    kwargs["max_residual_norm"] = 0.0

    plan = plan_linear_measurement_update(**kwargs)

    assert plan.residual_threshold == 0.0
    assert plan.accepted is False
    assert plan.action == "residual_rejected"


def test_zero_source_gate_threshold_is_valid_hard_gate():
    measurement = _Measurement(source="radar", vector=np.array([1.0]))

    threshold = gate_threshold_for_measurement(
        measurement,
        gate_thresholds_by_source={"radar": 0.0},
    )

    assert threshold == 0.0
