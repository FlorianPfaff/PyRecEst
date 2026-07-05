import numpy as np
import pytest
from pyrecest.filters.linear_update_planning import (
    chi_square_gate_threshold,
    normalized_innovation_squared,
    plan_linear_measurement_update,
    robust_update_covariance_scale,
    student_t_covariance_scale,
)


def test_chi_square_gate_threshold_matches_known_2d_95_percent_gate():
    assert np.isclose(chi_square_gate_threshold(0.95, 2), 5.991464547107979)


def test_hard_gate_rejects_without_changing_effective_covariance():
    plan = plan_linear_measurement_update(
        mean=np.zeros(2),
        covariance_matrix=np.eye(2),
        measurement_vector=np.array([10.0, 0.0]),
        measurement_covariance=np.eye(2),
        observation_matrix=np.eye(2),
        gate_threshold=1.0,
        robust_update="none",
    )
    assert not plan.accepted
    assert plan.action == "rejected"
    assert plan.covariance_scale == 1.0


def test_student_t_plan_inflates_outlier_covariance():
    plan = plan_linear_measurement_update(
        mean=np.zeros(2),
        covariance_matrix=np.eye(2),
        measurement_vector=np.array([100.0, 100.0]),
        measurement_covariance=np.eye(2),
        observation_matrix=np.eye(2),
        robust_update="student-t",
        student_t_dof=4.0,
    )
    assert plan.accepted
    assert plan.action == "student_t"
    assert plan.covariance_scale > 1.0
    assert np.allclose(plan.covariance, np.eye(2) * plan.covariance_scale)


def test_nis_inflate_uses_gate_ratio():
    scale, action = robust_update_covariance_scale(
        "nis-inflate",
        nis=10.0,
        measurement_dim=2,
        gate_threshold=2.5,
        inflation_alpha=0.5,
    )
    assert action == "inflated"
    assert np.isclose(scale, 2.0)


def test_chi_square_gate_threshold_rejects_invalid_measurement_dim():
    invalid_dimensions = (True, 1.5, np.nan, np.inf, np.array([2]))

    for measurement_dim in invalid_dimensions:
        with pytest.raises(ValueError, match="measurement_dim"):
            chi_square_gate_threshold(0.95, measurement_dim)


def test_plan_rejects_nonfinite_threshold_parameters():
    base_kwargs = {
        "mean": np.zeros(1),
        "covariance_matrix": np.eye(1),
        "measurement_vector": np.array([1.0]),
        "measurement_covariance": np.eye(1),
        "observation_matrix": np.eye(1),
    }
    invalid_overrides = (
        {"gate_threshold": np.nan},
        {"safety_gate_threshold": np.inf},
        {"max_residual_norm": np.nan},
        {"inflation_alpha": np.nan},
    )

    for overrides in invalid_overrides:
        with pytest.raises(ValueError, match=next(iter(overrides))):
            plan_linear_measurement_update(**base_kwargs, **overrides)


def test_plan_rejects_nonfinite_array_inputs():
    base_kwargs = {
        "mean": np.zeros(1),
        "covariance_matrix": np.eye(1),
        "measurement_vector": np.array([1.0]),
        "measurement_covariance": np.eye(1),
        "observation_matrix": np.eye(1),
    }
    invalid_overrides = (
        {"measurement_vector": np.array([np.nan])},
        {"measurement_covariance": np.array([[np.inf]])},
        {"observation_matrix": np.array([[np.nan]])},
        {"mean": np.array([np.inf])},
        {"covariance_matrix": np.array([[np.nan]])},
    )

    for overrides in invalid_overrides:
        invalid_name = next(iter(overrides))
        kwargs = {**base_kwargs, **overrides}
        with pytest.raises(ValueError, match=invalid_name):
            plan_linear_measurement_update(**kwargs)


def test_normalized_innovation_squared_rejects_nonfinite_inputs():
    with pytest.raises(ValueError, match="residual"):
        normalized_innovation_squared(np.array([np.nan]), np.eye(1))

    with pytest.raises(ValueError, match="innovation_covariance"):
        normalized_innovation_squared(np.array([0.0]), np.array([[np.inf]]))


def test_robust_scale_rejects_nonfinite_parameters():
    invalid_cases = (
        {
            "robust_update": "nis-inflate",
            "nis": np.nan,
            "measurement_dim": 1,
            "gate_threshold": 1.0,
            "inflation_alpha": 1.0,
        },
        {
            "robust_update": "nis-inflate",
            "nis": 2.0,
            "measurement_dim": 1,
            "gate_threshold": np.nan,
            "inflation_alpha": 1.0,
        },
        {
            "robust_update": "nis-inflate",
            "nis": 2.0,
            "measurement_dim": 1,
            "gate_threshold": 1.0,
            "inflation_alpha": np.inf,
        },
        {
            "robust_update": "huber",
            "nis": 2.0,
            "measurement_dim": 1,
            "gate_threshold": None,
            "huber_threshold": np.nan,
        },
    )

    for kwargs in invalid_cases:
        with pytest.raises(ValueError):
            robust_update_covariance_scale(**kwargs)


def test_student_t_covariance_scale_uses_sanitized_degrees_of_freedom():
    expected = student_t_covariance_scale(
        10.0,
        measurement_dim=2,
        degrees_of_freedom=4.0,
    )

    actual = student_t_covariance_scale(
        10.0,
        measurement_dim=2,
        degrees_of_freedom="4.0",
    )

    assert np.isclose(actual, expected)


def test_student_t_covariance_scale_rejects_nonfinite_dof():
    with pytest.raises(ValueError, match="degrees_of_freedom"):
        student_t_covariance_scale(1.0, measurement_dim=2, degrees_of_freedom=np.nan)
