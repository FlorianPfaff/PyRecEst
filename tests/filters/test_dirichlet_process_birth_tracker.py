import numpy as np
import pyrecest.backend
import pytest
from pyrecest.filters.dirichlet_process_birth_tracker import (
    DirichletProcessBirthMultiBernoulliTracker,
    DPBirthAtom,
    DPBirthMultiBernoulliTracker,
)

pytestmark = pytest.mark.skipif(
    pyrecest.backend.__backend_name__ != "numpy",
    reason="DP birth multi-Bernoulli tracker inherits the NumPy-only MultiBernoulliTracker.",
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


@pytest.mark.parametrize("invalid_count", [np.nan, np.inf, -np.inf])
def test_birth_atom_rejects_nonfinite_count(invalid_count):
    with pytest.raises(ValueError, match="count must be finite and positive"):
        DPBirthAtom(np.zeros(2), np.eye(2), invalid_count)


@pytest.mark.parametrize("invalid_concentration", [np.nan, np.inf, -np.inf])
def test_tracker_rejects_nonfinite_dp_concentration(invalid_concentration):
    tracker, measurement_matrix, measurement_covariance = _tracker(
        dp_concentration=invalid_concentration
    )

    with pytest.raises(
        ValueError, match="dp_concentration must be finite and positive"
    ):
        tracker._create_birth_component_from_measurement(
            np.zeros(2),
            measurement_matrix,
            measurement_covariance,
        )


def test_alias_export_matches_long_class_name():
    assert DPBirthMultiBernoulliTracker is DirichletProcessBirthMultiBernoulliTracker


@pytest.mark.parametrize(
    "invalid_limit",
    [
        -1,
        1.5,
        True,
        np.array([1]),
        np.nan,
        np.inf,
    ],
)
def test_birth_atom_cap_rejects_invalid_limits(invalid_limit):
    tracker, _, _ = _tracker(maximum_number_of_birth_atoms=invalid_limit)

    with pytest.raises(ValueError, match="maximum_number_of_birth_atoms"):
        tracker._prune_and_cap_birth_atoms()


@pytest.mark.parametrize("valid_limit", [0, 1, 2.0, np.int64(3)])
def test_birth_atom_cap_accepts_nonnegative_integer_limits(valid_limit):
    tracker, _, _ = _tracker(
        maximum_number_of_birth_atoms=valid_limit,
        birth_atoms=[
            (np.zeros(4), np.eye(4), 1.0),
            (np.ones(4), np.eye(4), 2.0),
        ],
    )

    tracker._prune_and_cap_birth_atoms()

    assert len(tracker.birth_atoms) == min(int(valid_limit), 2)
