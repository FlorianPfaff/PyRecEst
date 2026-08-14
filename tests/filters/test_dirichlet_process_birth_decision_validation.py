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
    return DirichletProcessBirthMultiBernoulliTracker(tracker_param=tracker_param)


def _try_birth(tracker):
    return tracker._create_birth_component_from_measurement(
        np.array([2.0, 3.0]),
        np.array(
            [
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
            ]
        ),
        np.eye(2) * 0.2,
    )


@pytest.mark.parametrize(
    "invalid_concentration",
    [True, np.array([0.05]), np.ma.array(0.05, mask=True)],
)
def test_rejects_non_scalar_or_semantic_dp_concentration(invalid_concentration):
    tracker = _tracker(dp_concentration=invalid_concentration)

    with pytest.raises(
        ValueError,
        match="dp_concentration must be finite and positive",
    ):
        _try_birth(tracker)

    assert tracker.birth_atoms == []
    assert tracker.last_birth_diagnostics == []


@pytest.mark.parametrize(
    "invalid_threshold",
    [np.nan, np.inf, -1.0, True, np.ma.array(1.0, mask=True)],
)
def test_rejects_invalid_dp_birth_threshold(invalid_threshold):
    tracker = _tracker(dp_birth_threshold=invalid_threshold)

    with pytest.raises(
        ValueError,
        match="dp_birth_threshold must be finite and non-negative",
    ):
        _try_birth(tracker)

    assert tracker.birth_atoms == []
    assert tracker.last_birth_diagnostics == []


@pytest.mark.parametrize(
    "invalid_intensity",
    [np.nan, np.inf, -1.0, True, np.ma.array(1e-6, mask=True)],
)
def test_rejects_invalid_explicit_dp_birth_clutter_intensity(invalid_intensity):
    tracker = _tracker(dp_birth_clutter_intensity=invalid_intensity)

    with pytest.raises(
        ValueError,
        match="dp_birth_clutter_intensity must be finite and non-negative",
    ):
        _try_birth(tracker)

    assert tracker.birth_atoms == []
    assert tracker.last_birth_diagnostics == []


@pytest.mark.parametrize("invalid_intensity", [np.nan, np.inf, -1.0, True])
def test_rejects_invalid_fallback_clutter_intensity(invalid_intensity):
    tracker = _tracker(clutter_intensity=invalid_intensity)

    with pytest.raises(
        ValueError,
        match="clutter_intensity must be finite and non-negative",
    ):
        _try_birth(tracker)

    assert tracker.birth_atoms == []
    assert tracker.last_birth_diagnostics == []


def test_zero_threshold_and_zero_clutter_intensity_remain_supported():
    tracker = _tracker(
        dp_concentration=np.float64(0.05),
        dp_birth_threshold=0.0,
        dp_birth_clutter_intensity=0.0,
    )

    component = _try_birth(tracker)

    assert component is not None
    assert len(tracker.birth_atoms) == 1
    assert tracker.last_birth_diagnostics[0]["action"] == "new_atom"
