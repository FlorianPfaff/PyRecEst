import numpy as np
import pyrecest.backend
import pytest
from pyrecest.filters.dirichlet_process_birth_tracker import (
    DirichletProcessBirthMultiBernoulliTracker,
)


pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="DP birth multi-Bernoulli tracker inherits the NumPy-only MultiBernoulliTracker.",
)


def _tracker_with_active_birth_atom(survival_probability):
    measurement_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    measurement_covariance = np.eye(2) * 0.2
    tracker = DirichletProcessBirthMultiBernoulliTracker(
        tracker_param={
            "birth_covariance": np.diag([1.0, 1.0, 4.0, 4.0]),
            "birth_existence_probability": 0.8,
            "clutter_intensity": 1e-6,
            "dp_concentration": 0.05,
            "dp_birth_threshold": 1.0,
            "dp_birth_atom_survival_probability": survival_probability,
            "measurement_to_state_matrix": np.array(
                [
                    [1.0, 0.0],
                    [0.0, 1.0],
                    [0.0, 0.0],
                    [0.0, 0.0],
                ]
            ),
        }
    )
    tracker.update_linear(
        np.array([[2.0], [3.0]]),
        measurement_matrix,
        measurement_covariance,
    )
    return tracker


@pytest.mark.parametrize(
    "invalid_survival_probability", [-0.1, 1.1, np.nan]
)
def test_invalid_birth_atom_survival_is_rejected_before_prediction(
    invalid_survival_probability,
):
    tracker = _tracker_with_active_birth_atom(
        invalid_survival_probability
    )
    label = tracker.get_component_labels()[0]
    component_before = tracker.get_component_by_label(label)
    existence_before = component_before.existence_probability
    estimate_before = component_before.get_point_estimate().copy()
    atom_before = tracker.get_birth_atoms()[0]

    transition = np.eye(4)
    transition[0, 0] = 2.0

    with pytest.raises(
        ValueError,
        match="dp_birth_atom_survival_probability must be in \[0, 1\]",
    ):
        tracker.predict_linear(
            transition,
            np.zeros((4, 4)),
            inputs=np.ones(4),
        )

    component_after = tracker.get_component_by_label(label)
    np.testing.assert_allclose(
        component_after.get_point_estimate(), estimate_before
    )
    assert component_after.existence_probability == pytest.approx(
        existence_before
    )
    np.testing.assert_allclose(
        tracker.birth_atoms[0].mean, atom_before.mean
    )
    np.testing.assert_allclose(
        tracker.birth_atoms[0].covariance, atom_before.covariance
    )
    assert tracker.birth_atoms[0].count == pytest.approx(
        atom_before.count
    )
