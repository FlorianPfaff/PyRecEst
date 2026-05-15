import numpy as np
import numpy.testing as npt
from pyrecest.backend import array, diag, eye
from pyrecest.filters.velocity_locked_mem_qkf_tracker import VelocityLockedMEMQKFTracker
from pyrecest.smoothers import (
    FixedLagVelocityLockedMEMQKFSmoother,
    FixedLagVLMEMQKFSmoother,
    FLVLMEMQKFSmoother,
    VelocityLockedMEMQKFTrackerState,
)


def _state(
    kinematic_state,
    covariance,
    shape_state,
    shape_covariance,
    *,
    speed_threshold=1e-9,
):
    return VelocityLockedMEMQKFTrackerState(
        array(kinematic_state),
        array(covariance),
        array(shape_state),
        array(shape_covariance),
        speed_threshold=speed_threshold,
    )


def test_aliases():
    assert FixedLagVLMEMQKFSmoother is FixedLagVelocityLockedMEMQKFSmoother
    assert FLVLMEMQKFSmoother is FixedLagVelocityLockedMEMQKFSmoother


def test_lag_one_smooths_kinematics_and_relocks_orientation_to_velocity():
    smoother = FixedLagVelocityLockedMEMQKFSmoother(lag=1)
    initial_heading = np.arctan2(4.0, 3.0)
    filtered_states = [
        _state(
            [0.0, 0.0, 3.0, 4.0],
            diag(array([0.5, 0.5, 0.04, 0.04])),
            [initial_heading, 2.0, 1.0],
            diag(array([0.02, 0.2, 0.2])),
        ),
        _state(
            [1.0, -1.0, 4.0, 0.0],
            diag(array([0.4, 0.4, 0.03, 0.03])),
            [0.0, 3.0, 2.0],
            diag(array([0.02, 0.1, 0.1])),
        ),
    ]
    predicted_states = [
        _state(
            [0.0, 0.0, 3.0, 4.0],
            diag(array([1.0, 1.0, 0.08, 0.08])),
            [initial_heading, 2.0, 1.0],
            diag(array([0.04, 0.4, 0.4])),
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
        array([0.5, -0.5, 3.5, 2.0]),
    )
    npt.assert_allclose(smoothed_states[0].shape_state[1:], array([2.5, 1.5]))
    npt.assert_allclose(
        smoothed_states[0].shape_state[0],
        np.arctan2(2.0, 3.5),
    )
    npt.assert_allclose(smoother_gains[0][0].kinematic, 0.5 * eye(4))
    npt.assert_allclose(smoother_gains[0][0].shape, 0.5 * eye(3))


def test_stationary_state_uses_shape_rts_orientation_instead_of_velocity_lock():
    smoother = FixedLagVelocityLockedMEMQKFSmoother(lag=1)
    filtered_states = [
        _state(
            [0.0, 0.0, 0.1, 0.0],
            diag(array([0.5, 0.5, 0.04, 0.04])),
            [0.1, 2.0, 1.0],
            diag(array([0.2, 0.2, 0.2])),
            speed_threshold=10.0,
        ),
        _state(
            [1.0, 0.0, 0.1, 0.0],
            diag(array([0.4, 0.4, 0.03, 0.03])),
            [0.5, 2.0, 1.0],
            diag(array([0.1, 0.1, 0.1])),
            speed_threshold=10.0,
        ),
    ]
    predicted_states = [
        _state(
            [0.0, 0.0, 0.1, 0.0],
            diag(array([1.0, 1.0, 0.08, 0.08])),
            [0.1, 2.0, 1.0],
            diag(array([0.4, 0.4, 0.4])),
            speed_threshold=10.0,
        )
    ]

    smoothed_states, _ = smoother.smooth(
        filtered_states,
        predicted_states,
        system_matrices=eye(4),
        shape_system_matrices=eye(3),
    )

    npt.assert_allclose(smoothed_states[0].shape_state[0], array(0.3))


def test_state_snapshot_round_trips_to_velocity_locked_tracker():
    tracker = VelocityLockedMEMQKFTracker(
        kinematic_state=array([0.0, 0.0, 3.0, 4.0]),
        covariance=diag(array([1.0, 1.0, 0.04, 0.04])),
        shape_state=array([0.0, 2.0, 1.0]),
        shape_covariance=diag(array([0.5, 0.2, 0.2])),
        default_meas_noise_cov=0.05 * eye(2),
    )

    state = VelocityLockedMEMQKFTrackerState.from_tracker(tracker)
    round_tripped = state.to_tracker()

    assert isinstance(round_tripped, VelocityLockedMEMQKFTracker)
    npt.assert_allclose(round_tripped.kinematic_state, tracker.kinematic_state)
    npt.assert_allclose(round_tripped.covariance, tracker.covariance)
    npt.assert_allclose(round_tripped.shape_state, tracker.shape_state)
    npt.assert_allclose(round_tripped.shape_covariance, tracker.shape_covariance)


def test_append_and_flush_emit_fixed_lag_sequence():
    smoother = FixedLagVelocityLockedMEMQKFSmoother(lag=1)
    heading = np.arctan2(4.0, 3.0)
    first = _state(
        [0.0, 0.0, 3.0, 4.0],
        diag(array([0.5, 0.5, 0.04, 0.04])),
        [heading, 2.0, 1.0],
        diag(array([0.02, 0.2, 0.2])),
    )
    second = _state(
        [1.0, -1.0, 4.0, 0.0],
        diag(array([0.4, 0.4, 0.03, 0.03])),
        [0.0, 3.0, 2.0],
        diag(array([0.02, 0.1, 0.1])),
    )
    predicted_second = _state(
        [0.0, 0.0, 3.0, 4.0],
        diag(array([1.0, 1.0, 0.08, 0.08])),
        [heading, 2.0, 1.0],
        diag(array([0.04, 0.4, 0.4])),
    )

    assert smoother.append(first) is None
    emitted = smoother.append(second, predicted_second, eye(4), eye(3))
    assert emitted is not None
    npt.assert_allclose(emitted.kinematic_state, array([0.5, -0.5, 3.5, 2.0]))

    remaining = smoother.flush()
    assert len(remaining) == 1
    npt.assert_allclose(remaining[0].kinematic_state, second.kinematic_state)


def test_lag_zero_returns_postprocessed_filtered_states():
    smoother = FixedLagVelocityLockedMEMQKFSmoother(lag=0)
    state = _state(
        [0.0, 0.0, 3.0, 4.0],
        diag(array([0.5, 0.5, 0.04, 0.04])),
        [0.0, 2.0, 1.0],
        diag(array([0.02, 0.2, 0.2])),
    )

    smoothed_states, smoother_gains = smoother.smooth([state])

    assert len(smoothed_states) == 1
    assert smoother_gains == [[]]
    npt.assert_allclose(smoothed_states[0].shape_state[0], np.arctan2(4.0, 3.0))
