import numpy as np
import numpy.testing as npt
from pyrecest.backend import array, diag, eye
from pyrecest.filters.mem_qkf_tracker import MEMQKFTracker
from pyrecest.smoothers import (
    FixedIntervalMEMQKFSmoother,
    FixedIntervalMemQkfSmoother,
    FixedLagFreeMEMQKFSmoother,
    FixedLagMEMQKFSmoother,
    FixedLagMemQkfSmoother,
    FLMEMQKFSmoother,
    FullIntervalMEMQKFSmoother,
    MEMQKFTrackerState,
)


def _state(
    kinematic_state,
    covariance,
    shape_state,
    shape_covariance,
    *,
    minimum_axis_length=1e-9,
    minimum_covariance_eigenvalue=0.0,
):
    return MEMQKFTrackerState(
        array(kinematic_state),
        array(covariance),
        array(shape_state),
        array(shape_covariance),
        minimum_axis_length=minimum_axis_length,
        minimum_covariance_eigenvalue=minimum_covariance_eigenvalue,
    )


def test_aliases():
    assert FixedLagMemQkfSmoother is FixedLagMEMQKFSmoother
    assert FixedLagFreeMEMQKFSmoother is FixedLagMEMQKFSmoother
    assert FLMEMQKFSmoother is FixedLagMEMQKFSmoother
    assert FixedIntervalMemQkfSmoother is FixedIntervalMEMQKFSmoother
    assert FullIntervalMEMQKFSmoother is FixedIntervalMEMQKFSmoother


def test_lag_one_smooths_kinematics_and_mem_qkf_shape_state():
    smoother = FixedLagMEMQKFSmoother(lag=1)
    filtered_states = [
        _state(
            [0.0, 0.0, 1.0, 0.0],
            diag(array([0.5, 0.5, 0.5, 0.5])),
            [0.1, 2.0, 1.0],
            diag(array([0.2, 0.2, 0.2])),
        ),
        _state(
            [1.0, -1.0, 3.0, 0.0],
            diag(array([0.4, 0.4, 0.4, 0.4])),
            [0.5, 3.0, 2.0],
            diag(array([0.1, 0.1, 0.1])),
        ),
    ]
    predicted_states = [
        _state(
            [0.0, 0.0, 1.0, 0.0],
            diag(array([1.0, 1.0, 1.0, 1.0])),
            [0.1, 2.0, 1.0],
            diag(array([0.4, 0.4, 0.4])),
        )
    ]

    smoothed_states, smoother_gains = smoother.smooth(
        filtered_states=filtered_states,
        predicted_states=predicted_states,
        system_matrices=eye(4),
        shape_system_matrices=eye(3),
    )

    npt.assert_allclose(
        smoothed_states[0].kinematic_state,
        array([0.5, -0.5, 2.0, 0.0]),
    )
    npt.assert_allclose(
        smoothed_states[0].shape_state,
        array([0.3, 2.5, 1.5]),
        rtol=1e-6,
        atol=1e-7,
    )
    npt.assert_allclose(smoother_gains[0][0].kinematic, 0.5 * eye(4))
    npt.assert_allclose(smoother_gains[0][0].shape, 0.5 * eye(3))


def test_shape_smoothing_none_only_postprocesses_filtered_shape_state():
    smoother = FixedLagMEMQKFSmoother(lag=1, shape_smoothing="none")
    filtered_states = [
        _state(
            [0.0, 0.0, 1.0, 0.0],
            diag(array([0.5, 0.5, 0.5, 0.5])),
            [0.1, 2.0, 1.0],
            diag(array([0.2, 0.2, 0.2])),
        ),
        _state(
            [1.0, -1.0, 3.0, 0.0],
            diag(array([0.4, 0.4, 0.4, 0.4])),
            [0.5, 3.0, 2.0],
            diag(array([0.1, 0.1, 0.1])),
        ),
    ]
    predicted_states = [
        _state(
            [0.0, 0.0, 1.0, 0.0],
            diag(array([1.0, 1.0, 1.0, 1.0])),
            [0.1, 2.0, 1.0],
            diag(array([0.4, 0.4, 0.4])),
        )
    ]

    smoothed_states, smoother_gains = smoother.smooth(
        filtered_states,
        predicted_states,
        system_matrices=eye(4),
        shape_system_matrices=eye(3),
    )

    npt.assert_allclose(smoothed_states[0].shape_state, filtered_states[0].shape_state)
    assert smoother_gains[0][0].shape is None


def test_axial_orientation_residual_wraps_across_pi_period():
    smoother = FixedLagMEMQKFSmoother(lag=1)
    reference = np.pi - 0.05
    filtered_states = [
        _state(
            [0.0, 0.0, 0.0, 0.0],
            eye(4),
            [reference, 2.0, 1.0],
            diag(array([0.2, 0.2, 0.2])),
        ),
        _state(
            [0.0, 0.0, 0.0, 0.0],
            eye(4),
            [0.03, 2.0, 1.0],
            diag(array([0.1, 0.1, 0.1])),
        ),
    ]
    predicted_states = [
        _state(
            [0.0, 0.0, 0.0, 0.0],
            2.0 * eye(4),
            [reference, 2.0, 1.0],
            diag(array([0.4, 0.4, 0.4])),
        )
    ]

    smoothed_states, _ = smoother.smooth(
        filtered_states,
        predicted_states,
        system_matrices=eye(4),
        shape_system_matrices=eye(3),
    )

    expected_delta = ((0.03 - reference + np.pi / 2.0) % np.pi) - np.pi / 2.0
    npt.assert_allclose(
        smoothed_states[0].shape_state[0], reference + 0.5 * expected_delta
    )


def test_state_snapshot_round_trips_to_mem_qkf_tracker():
    tracker = MEMQKFTracker(
        kinematic_state=array([0.0, 0.0, 1.0, 0.0]),
        covariance=eye(4),
        shape_state=array([0.1, 2.0, 1.0]),
        shape_covariance=diag(array([0.2, 0.1, 0.1])),
        measurement_matrix=eye(2, 4),
        multiplicative_noise_cov=0.25 * eye(2),
        default_meas_noise_cov=0.05 * eye(2),
    )

    state = MEMQKFTrackerState.from_tracker(tracker)
    round_tripped = state.to_tracker()

    assert isinstance(round_tripped, MEMQKFTracker)
    npt.assert_allclose(round_tripped.kinematic_state, tracker.kinematic_state)
    npt.assert_allclose(round_tripped.covariance, tracker.covariance)
    npt.assert_allclose(round_tripped.shape_state, tracker.shape_state)
    npt.assert_allclose(round_tripped.shape_covariance, tracker.shape_covariance)


def test_append_and_flush_emit_fixed_lag_sequence():
    smoother = FixedLagMEMQKFSmoother(lag=1)
    first = _state(
        [0.0, 0.0, 1.0, 0.0],
        diag(array([0.5, 0.5, 0.5, 0.5])),
        [0.1, 2.0, 1.0],
        diag(array([0.2, 0.2, 0.2])),
    )
    second = _state(
        [1.0, -1.0, 3.0, 0.0],
        diag(array([0.4, 0.4, 0.4, 0.4])),
        [0.5, 3.0, 2.0],
        diag(array([0.1, 0.1, 0.1])),
    )
    predicted_second = _state(
        [0.0, 0.0, 1.0, 0.0],
        diag(array([1.0, 1.0, 1.0, 1.0])),
        [0.1, 2.0, 1.0],
        diag(array([0.4, 0.4, 0.4])),
    )

    assert smoother.append(first) is None
    emitted = smoother.append(predicted_state=predicted_second, filtered_state=second)
    assert emitted is not None
    npt.assert_allclose(emitted.kinematic_state, array([0.5, -0.5, 2.0, 0.0]))

    remaining = smoother.flush()
    assert len(remaining) == 1
    npt.assert_allclose(remaining[0].kinematic_state, second.kinematic_state)


def test_fixed_interval_smoother_runs_full_suffix_windows():
    smoother = FixedIntervalMEMQKFSmoother()
    filtered_states = [
        _state([0.0], array([[0.5]]), [0.1, 2.0, 1.0], diag(array([0.2, 0.2, 0.2]))),
        _state([1.0], array([[0.4]]), [0.3, 3.0, 2.0], diag(array([0.1, 0.1, 0.1]))),
        _state([2.0], array([[0.3]]), [0.4, 4.0, 3.0], diag(array([0.1, 0.1, 0.1]))),
    ]
    predicted_states = [
        _state([0.0], array([[1.0]]), [0.1, 2.0, 1.0], diag(array([0.4, 0.4, 0.4]))),
        _state([1.0], array([[1.0]]), [0.3, 3.0, 2.0], diag(array([0.4, 0.4, 0.4]))),
    ]

    smoothed_states, smoother_gains = smoother.smooth(
        filtered_states,
        predicted_states,
        system_matrices=array([[[1.0]], [[1.0]]]),
        shape_system_matrices=array([eye(3), eye(3)]),
    )

    assert len(smoothed_states) == 3
    assert len(smoother_gains) == 3
    assert len(smoother_gains[0]) == 2
    assert len(smoother_gains[1]) == 1
    assert smoother_gains[2] == []
    npt.assert_allclose(
        smoothed_states[-1].kinematic_state, filtered_states[-1].kinematic_state
    )
