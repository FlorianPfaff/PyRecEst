import numpy as np
import pyrecest.backend
import pytest
from pyrecest.filters.dirichlet_process_birth_tracker import (
    DirichletProcessBirthMultiBernoulliTracker,
    DPBirthMultiBernoulliTracker,
)

pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="DP birth multi-Bernoulli tracker inherits the NumPy-only MultiBernoulliTracker.",
)


def _tracker(**overrides):
    tracker_param = {
        "birth_covariance": np.diag([1.0, 1.0, 4.0, 4.0]),
        "birth_existence_probability": 0.8,
        "clutter_intensity": 1e-6,
        "dp_concentration": 0.05,
        "dp_birth_threshold": 1.0,
        "measurement_to_state_matrix": np.array(
            [
                [1.0, 0.0],
                [0.0, 1.0],
                [0.0, 0.0],
                [0.0, 0.0],
            ]
        ),
    }
    tracker_param.update(overrides)
    measurement_matrix = np.array(
        [
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ]
    )
    measurement_covariance = np.eye(2) * 0.2
    return (
        DirichletProcessBirthMultiBernoulliTracker(tracker_param=tracker_param),
        measurement_matrix,
        measurement_covariance,
    )


def test_unassigned_measurement_creates_birth_atom_and_component():
    tracker, measurement_matrix, measurement_covariance = _tracker()

    tracker.update_linear(
        np.array([[2.0], [3.0]]), measurement_matrix, measurement_covariance
    )

    assert len(tracker.birth_atoms) == 1
    assert tracker.get_number_of_components() == 1
    assert tracker.get_expected_number_of_targets() > 0.0
    assert tracker.last_birth_diagnostics[0]["action"] == "new_atom"


def test_high_clutter_suppresses_birth_creation():
    tracker, measurement_matrix, measurement_covariance = _tracker(
        clutter_intensity=1e6,
        dp_birth_threshold=10.0,
    )

    tracker.update_linear(
        np.array([[2.0], [3.0]]), measurement_matrix, measurement_covariance
    )

    assert len(tracker.birth_atoms) == 0
    assert tracker.get_number_of_components() == 0
    assert tracker.last_birth_diagnostics[0]["action"] == "clutter"


def test_nearby_unassigned_measurement_reuses_existing_birth_atom():
    tracker, measurement_matrix, measurement_covariance = _tracker()

    first_component = tracker._create_birth_component_from_measurement(
        np.array([0.0, 0.0]),
        measurement_matrix,
        measurement_covariance,
    )
    second_component = tracker._create_birth_component_from_measurement(
        np.array([0.1, -0.1]),
        measurement_matrix,
        measurement_covariance,
    )

    assert first_component is not None
    assert second_component is not None
    assert len(tracker.birth_atoms) == 1
    assert tracker.birth_atoms[0].count > 1.0
    assert tracker.last_birth_diagnostics[-1]["action"] == "existing_atom"


def test_alias_export_matches_long_class_name():
    assert DPBirthMultiBernoulliTracker is DirichletProcessBirthMultiBernoulliTracker
