import numpy as np

from pyrecest.filters.linear_update_planning import (
    chi_square_gate_threshold,
    plan_linear_measurement_update,
    robust_update_covariance_scale,
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
