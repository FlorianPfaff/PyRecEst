from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest
from pyrecest.filters.linear_update_planning import (
    chi_square_gate_threshold,
    huber_covariance_scale,
    normalized_innovation_squared,
    plan_linear_measurement_update,
    source_float_value,
    student_t_covariance_scale,
)


@dataclass(frozen=True)
class _Measurement:
    source: str
    vector: object


def _base_plan_kwargs(dim: int = 1):
    return {
        "mean": np.zeros(dim),
        "covariance_matrix": np.eye(dim),
        "measurement_vector": np.ones(dim),
        "measurement_covariance": np.eye(dim),
        "observation_matrix": np.eye(dim),
    }


@pytest.mark.parametrize(
    "invalid",
    ("0.95", b"0.95", np.str_("0.95"), np.bytes_("0.95")),
)
def test_linear_update_scalar_controls_reject_text_values(invalid) -> None:
    with pytest.raises(ValueError, match="probability"):
        chi_square_gate_threshold(invalid, 1)
    with pytest.raises(ValueError, match="threshold"):
        huber_covariance_scale(4.0, threshold=invalid)
    with pytest.raises(ValueError, match="nis"):
        student_t_covariance_scale(invalid, measurement_dim=1)
    with pytest.raises(ValueError, match="gate_threshold"):
        plan_linear_measurement_update(
            **_base_plan_kwargs(),
            gate_threshold=invalid,
        )

    measurement = _Measurement(source="radar", vector=np.array([0.0]))
    with pytest.raises(ValueError, match="source value"):
        source_float_value(measurement, {"radar": invalid})


@pytest.mark.parametrize(
    "invalid",
    (
        ["1.0"],
        [b"1.0"],
        [1.0 + 2.0j],
        np.array([np.complex128(1.0 + 2.0j)], dtype=object),
        [1.0, "2.0"],
    ),
)
def test_linear_update_vectors_reject_text_and_complex_values(invalid) -> None:
    dim = np.asarray(invalid).size
    with pytest.raises(ValueError, match="residual"):
        normalized_innovation_squared(invalid, np.eye(dim))

    kwargs = _base_plan_kwargs(dim)
    kwargs["measurement_vector"] = invalid
    with pytest.raises(ValueError, match="measurement_vector"):
        plan_linear_measurement_update(**kwargs)


@pytest.mark.parametrize(
    "invalid",
    (
        [["1.0"]],
        [[b"1.0"]],
        [[1.0 + 2.0j]],
        np.array([[np.complex128(1.0 + 2.0j)]], dtype=object),
        [[1.0, 0.0], [0.0, "2.0"]],
    ),
)
def test_linear_update_matrices_reject_text_and_complex_values(invalid) -> None:
    dim = np.asarray(invalid).shape[0]

    kwargs = _base_plan_kwargs(dim)
    kwargs["measurement_covariance"] = invalid
    with pytest.raises(ValueError, match="measurement_covariance"):
        plan_linear_measurement_update(**kwargs)

    kwargs = _base_plan_kwargs(dim)
    kwargs["observation_matrix"] = invalid
    with pytest.raises(ValueError, match="observation_matrix"):
        plan_linear_measurement_update(**kwargs)


def test_real_numeric_linear_update_inputs_remain_supported() -> None:
    threshold = chi_square_gate_threshold(np.float64(0.95), np.int64(2))
    assert threshold is not None and np.isfinite(threshold)

    plan = plan_linear_measurement_update(
        mean=[0, 0.0],
        covariance_matrix=[[1, 0.0], [0, 1.0]],
        measurement_vector=[1, 2.5],
        measurement_covariance=[[1, 0.0], [0, 2.0]],
        observation_matrix=[[1, 0.0], [0, 1.0]],
        gate_threshold=np.float64(10.0),
    )
    assert np.allclose(plan.vector, [1.0, 2.5])
    assert np.allclose(plan.covariance, [[1.0, 0.0], [0.0, 2.0]])
    assert np.allclose(plan.observation, np.eye(2))
