import numpy as np
import pyrecest.backend
import pytest
from pyrecest.filters.dirichlet_process_birth_tracker import (
    DirichletProcessBirthMultiBernoulliTracker,
)

pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="DP birth multi-Bernoulli tracker is NumPy-only.",
)


def _tracker(**overrides):
    birth_atoms = overrides.pop("birth_atoms", None)
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
        DirichletProcessBirthMultiBernoulliTracker(
            tracker_param=tracker_param,
            birth_atoms=birth_atoms,
        ),
        measurement_matrix,
        measurement_covariance,
    )


def _assert_atom_unchanged(atom, reference):
    np.testing.assert_allclose(atom.mean, reference.mean)
    np.testing.assert_allclose(atom.covariance, reference.covariance)
    assert atom.count == reference.count


def test_new_birth_rejects_invalid_atom_cap_without_partial_state():
    tracker, measurement_matrix, measurement_covariance = _tracker(
        maximum_number_of_birth_atoms=1.5
    )

    with pytest.raises(ValueError, match="maximum_number_of_birth_atoms"):
        tracker._create_birth_component_from_measurement(
            np.array([2.0, 3.0]),
            measurement_matrix,
            measurement_covariance,
        )

    assert tracker.birth_atoms == []
    assert tracker.last_birth_diagnostics == []


def test_existing_birth_rejects_invalid_pruning_without_mutating_atom():
    tracker, measurement_matrix, measurement_covariance = _tracker(
        birth_atoms=[(np.zeros(4), np.eye(4), 2.0)],
        dp_birth_atom_pruning_threshold=np.nan,
    )
    atom_before = tracker.get_birth_atoms()[0]

    with pytest.raises(ValueError, match="dp_birth_atom_pruning_threshold"):
        tracker._create_birth_component_from_measurement(
            np.array([0.1, -0.1]),
            measurement_matrix,
            measurement_covariance,
        )

    assert len(tracker.birth_atoms) == 1
    _assert_atom_unchanged(tracker.birth_atoms[0], atom_before)
    assert tracker.last_birth_diagnostics == []


def test_failed_birth_component_construction_does_not_mutate_existing_atom():
    tracker, measurement_matrix, measurement_covariance = _tracker(
        birth_atoms=[(np.zeros(4), np.eye(4), 2.0)],
        birth_existence_probability=np.nan,
    )
    atom_before = tracker.get_birth_atoms()[0]

    with pytest.raises(ValueError, match="existence_probability"):
        tracker._create_birth_component_from_measurement(
            np.array([0.1, -0.1]),
            measurement_matrix,
            measurement_covariance,
        )

    assert len(tracker.birth_atoms) == 1
    _assert_atom_unchanged(tracker.birth_atoms[0], atom_before)
    assert tracker.last_birth_diagnostics == []
